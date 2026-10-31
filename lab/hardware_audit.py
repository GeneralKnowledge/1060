#!/usr/bin/env python3
"""M0 hardware audit: a read-only inventory of the GPU research machine.

Writes research/hardware_profile.json and keeps every raw command output in
research/audit_raw/<timestamp>/ so any fact in the profile can be traced back
to the evidence it came from.

Safety contract (do not weaken):
  * Only commands in READ_ONLY_COMMANDS are executed, never through a shell.
  * Nothing is written outside the output directory. No sysfs/proc writes,
    no nvidia-smi setters, no module loading, no driver/package changes.
  * Privileged reads (dmidecode) run only with --sudo, and then only through
    `sudo -n` (non-interactive; fails instead of prompting).
  * The optional idle telemetry sample is passive: it puts no load on the GPU.

Every fact is recorded as {"value", "status", "source"} where status is one of
  verified      read directly from the machine
  derived       computed from verified facts
  not_detected  the probe ran and found nothing
  unavailable   the probe could not run (tool missing, no permission, error)
Stdlib only, Python >= 3.8.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA_VERSION = "m0.1"

# What the project spec *claims* about the machine. The audit checks each one;
# none of these are trusted as facts.
SPEC_CLAIMS = {
    "machine": "Dell G3 17",
    "gpu_name_contains": "1060",
    "architecture": "Pascal",
    "compute_capability": "6.1",
    "system_ram_gb": 8,
    "os": "Ubuntu",
}

# argv tuples only. Anything not listed here is never executed.
READ_ONLY_COMMANDS = {
    "nvidia_smi_xml": ["nvidia-smi", "-q", "-x"],
    "nvidia_smi_query": [
        "nvidia-smi",
        "--query-gpu=name,pci.device_id,pci.sub_device_id,compute_cap,vbios_version,"
        "driver_version,memory.total,power.limit,power.default_limit,power.min_limit,"
        "power.max_limit,clocks.max.sm,clocks.max.mem,temperature.gpu",
        "--format=csv",
    ],
    "nvidia_smi_supported_clocks": ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
    "lspci_nvidia": ["lspci", "-nn", "-d", "10de:"],
    "lspci_vga": ["lspci", "-nn", "-d", "::0300"],
    "lspci_3d": ["lspci", "-nn", "-d", "::0302"],
    "lscpu_json": ["lscpu", "-J"],
    "lsblk_json": ["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,ROTA,MODEL,TRAN,MOUNTPOINT,FSTYPE"],
    "df_root": ["df", "-B1", "--output=source,fstype,size,used,avail,target", "/"],
    "lsmod": ["lsmod"],
    "modinfo_nvidia": ["modinfo", "-F", "version", "nvidia"],
    "nvcc_version": ["nvcc", "--version"],
    "ldconfig": ["ldconfig", "-p"],
    "vulkaninfo_summary": ["vulkaninfo", "--summary"],
    "sensors": ["sensors", "-j"],
    "prime_select": ["prime-select", "query"],
    "gcc_version": ["gcc", "--version"],
    "cmake_version": ["cmake", "--version"],
    "git_version": ["git", "--version"],
    "swapon": ["swapon", "--show", "--bytes", "--noheadings"],
    "dmidecode_memory": ["sudo", "-n", "dmidecode", "-t", "memory"],
}

TOOLS_OF_INTEREST = [
    "nvidia-smi", "nvidia-settings", "nvtop", "nvidia-persistenced", "dcgmi",
    "nvcc", "vulkaninfo", "glslc", "sensors", "lspci", "dmidecode", "cmake",
    "gcc", "g++", "git", "llama-cli", "llama-bench", "llama-server",
]

READABLE_FILES = {
    "os_release": "/etc/os-release",
    "proc_meminfo": "/proc/meminfo",
    "proc_cpuinfo": "/proc/cpuinfo",
    "proc_cmdline": "/proc/cmdline",
    "nvidia_proc_version": "/proc/driver/nvidia/version",
}

# Non-sensitive DMI fields (serials and UUIDs are deliberately excluded).
DMI_FIELDS = ["sys_vendor", "product_name", "product_family", "board_name",
              "bios_vendor", "bios_version", "bios_date"]


def fact(value, status, source):
    return {"value": value, "status": status, "source": source}


def missing(source, reason="unavailable"):
    return fact(None, reason, source)


class Collector:
    """Runs whitelisted probes and keeps the raw evidence."""

    def __init__(self, raw_dir: Path | None, timeout: float = 30.0, allow_sudo: bool = False):
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.allow_sudo = allow_sudo
        self.log: list[dict] = []
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, key: str, text: str):
        if self.raw_dir and text:
            (self.raw_dir / f"{key}.txt").write_text(text)

    def run(self, key: str) -> str | None:
        argv = READ_ONLY_COMMANDS[key]
        if argv[0] == "sudo" and not self.allow_sudo:
            self.log.append({"probe": key, "result": "skipped (needs --sudo)"})
            return None
        exe = argv[2] if argv[0] == "sudo" else argv[0]
        if shutil.which(exe) is None and not os.path.exists(f"/usr/sbin/{exe}"):
            self.log.append({"probe": key, "result": f"{exe} not installed"})
            return None
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout,
                               stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log.append({"probe": key, "result": f"error: {e}"})
            return None
        self._save(key, p.stdout + (("\n--- stderr ---\n" + p.stderr) if p.stderr else ""))
        if p.returncode != 0:
            self.log.append({"probe": key, "result": f"exit {p.returncode}: {p.stderr.strip()[:200]}"})
            return None
        self.log.append({"probe": key, "result": "ok"})
        return p.stdout

    def read(self, key: str, path: str) -> str | None:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError as e:
            self.log.append({"probe": key, "result": f"unreadable: {e.strerror}"})
            return None
        self._save(key, text)
        self.log.append({"probe": key, "result": "ok"})
        return text


# ---------------------------------------------------------------- parsers

def parse_os_release(text: str | None) -> dict:
    out = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v.strip().strip('"')
    return out


def parse_meminfo(text: str | None) -> dict:
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def parse_cpu_flags(text: str | None) -> list[str]:
    for line in (text or "").splitlines():
        if line.startswith("flags"):
            return sorted(set(line.split(":", 1)[1].split()))
    return []


def parse_lscpu(text: str | None) -> dict:
    if not text:
        return {}
    try:
        rows = json.loads(text)["lscpu"]
    except (ValueError, KeyError):
        return {}
    return {r["field"].rstrip(":"): r.get("data") for r in rows}


def _num(s):
    """'1911 MHz' -> 1911.0, '80.00 W' -> 80.0, 'N/A' -> None."""
    if s is None:
        return None
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m and "N/A" not in s else None


def _txt(node, path):
    if node is None:
        return None
    el = node.find(path)
    if el is None or el.text is None:
        return None
    t = el.text.strip()
    return None if t in ("N/A", "", "[N/A]", "Not Supported", "[Not Supported]") else t


def parse_nvidia_smi_xml(text: str | None) -> dict:
    """Extract the facts M0 needs from `nvidia-smi -q -x`. Tolerates field
    renames across driver branches by trying alternatives."""
    if not text:
        return {}
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    res = {"driver_version": _txt(root, "driver_version"),
           "cuda_version": _txt(root, "cuda_version"), "gpus": []}
    for g in root.findall("gpu"):
        pwr = g.find("gpu_power_readings")
        if pwr is None:
            pwr = g.find("power_readings")
        events = g.find("clocks_event_reasons")
        if events is None:
            events = g.find("clocks_throttle_reasons")
        reasons = {}
        if events is not None:
            for child in events:
                if child.text and child.text.strip() in ("Active", "Not Active"):
                    reasons[child.tag] = child.text.strip()
        supported = []
        for mem in g.findall("supported_clocks/supported_mem_clock"):
            supported.append({
                "mem_mhz": _num(_txt(mem, "value")),
                "graphics_mhz": [_num(c.text) for c in mem.findall("supported_graphics_clock")],
            })
        temp = g.find("temperature")
        res["gpus"].append({
            "pci_bus_id": g.get("id"),
            "product_name": _txt(g, "product_name"),
            "product_brand": _txt(g, "product_brand"),
            "product_architecture": _txt(g, "product_architecture"),
            "vbios_version": _txt(g, "vbios_version"),
            "pci_device_id": _txt(g, "pci/pci_device_id"),
            "pci_sub_system_id": _txt(g, "pci/pci_sub_system_id"),
            "pcie_link": {
                "max_gen": _txt(g, "pci/pci_gpu_link_info/pcie_gen/max_link_gen"),
                "current_gen": _txt(g, "pci/pci_gpu_link_info/pcie_gen/current_link_gen"),
                "max_width": _txt(g, "pci/pci_gpu_link_info/link_widths/max_link_width"),
                "current_width": _txt(g, "pci/pci_gpu_link_info/link_widths/current_link_width"),
            },
            "fb_memory_total_mib": _num(_txt(g, "fb_memory_usage/total")),
            "fb_memory_used_mib": _num(_txt(g, "fb_memory_usage/used")),
            "persistence_mode": _txt(g, "persistence_mode"),
            "compute_mode": _txt(g, "compute_mode"),
            "performance_state": _txt(g, "performance_state"),
            "fan_speed": _txt(g, "fan_speed"),
            "temperature_c": _num(_txt(temp, "gpu_temp")),
            "temp_slowdown_c": _num(_txt(temp, "gpu_temp_slow_threshold")),
            "temp_shutdown_c": _num(_txt(temp, "gpu_temp_max_threshold")),
            "temp_max_gpu_c": _num(_txt(temp, "gpu_temp_max_gpu_threshold")),
            "target_temp_c": _num(_txt(temp, "gpu_target_temperature")),
            "power": {
                "management": _txt(pwr, "power_management"),
                "draw_w": _num(_txt(pwr, "power_draw") or _txt(pwr, "instant_power_draw")),
                "limit_w": _num(_txt(pwr, "power_limit") or _txt(pwr, "current_power_limit")),
                "default_limit_w": _num(_txt(pwr, "default_power_limit")),
                "enforced_limit_w": _num(_txt(pwr, "enforced_power_limit")),
                "min_limit_w": _num(_txt(pwr, "min_power_limit")),
                "max_limit_w": _num(_txt(pwr, "max_power_limit")),
            },
            "clocks_mhz": {
                "graphics": _num(_txt(g, "clocks/graphics_clock")),
                "sm": _num(_txt(g, "clocks/sm_clock")),
                "mem": _num(_txt(g, "clocks/mem_clock")),
            },
            "max_clocks_mhz": {
                "graphics": _num(_txt(g, "max_clocks/graphics_clock")),
                "sm": _num(_txt(g, "max_clocks/sm_clock")),
                "mem": _num(_txt(g, "max_clocks/mem_clock")),
            },
            "default_applications_clocks_mhz": {
                "graphics": _num(_txt(g, "default_applications_clocks/graphics_clock")),
                "mem": _num(_txt(g, "default_applications_clocks/mem_clock")),
            },
            "clock_event_reasons": reasons,
            "supported_clocks": supported,
        })
    return res


def parse_vulkaninfo_summary(text: str | None) -> dict:
    """Pull instance version and per-device rows from `vulkaninfo --summary`."""
    if not text:
        return {}
    out = {"instance_version": None, "devices": []}
    m = re.search(r"Vulkan Instance Version:\s*(\S+)", text)
    if m:
        out["instance_version"] = m.group(1)
    for block in re.split(r"\n(?=GPU\d+:)", text):
        if not re.match(r"GPU\d+:", block):
            continue
        dev = {}
        for key in ("apiVersion", "driverVersion", "vendorID", "deviceID", "deviceType",
                    "deviceName", "driverID", "driverName", "driverInfo", "conformanceVersion"):
            m = re.search(rf"^\s*{key}\s*=\s*(.+)$", block, re.M)
            if m:
                dev[key] = m.group(1).strip()
        out["devices"].append(dev)
    return out


def parse_dmidecode_memory(text: str | None) -> list[dict]:
    dimms = []
    for block in (text or "").split("\n\n"):
        if "Memory Device" not in block:
            continue
        d = {}
        for key in ("Size", "Form Factor", "Locator", "Bank Locator", "Type", "Speed",
                    "Configured Memory Speed", "Manufacturer", "Part Number", "Rank"):
            m = re.search(rf"^\s*{key}:\s*(.+)$", block, re.M)
            if m:
                d[key] = m.group(1).strip()
        if d:
            d["populated"] = not d.get("Size", "").lower().startswith("no module")
            dimms.append(d)
    return dimms


def parse_ldconfig(text: str | None) -> dict:
    libs = {}
    for name in ("libcuda.so", "libnvidia-ml.so", "libcudart.so", "libcublas.so",
                 "libvulkan.so", "libnvidia-glvkspirv.so"):
        hits = sorted({l.split("=>")[-1].strip() for l in (text or "").splitlines() if name in l})
        libs[name] = hits
    return libs


# ------------------------------------------------- CUDA driver API (ctypes)

CU_ATTRS = {  # CUdevice_attribute enum values from cuda.h
    "max_threads_per_block": 1, "max_shared_memory_per_block": 8, "warp_size": 10,
    "clock_rate_khz": 13, "multiprocessor_count": 16, "compute_capability_major": 75,
    "compute_capability_minor": 76, "memory_clock_rate_khz": 36,
    "global_memory_bus_width_bits": 37, "l2_cache_size": 38,
    "max_threads_per_multiprocessor": 39, "max_shared_memory_per_multiprocessor": 81,
    "max_registers_per_multiprocessor": 82, "managed_memory": 83,
    "concurrent_managed_access": 89, "single_to_double_precision_perf_ratio": 87,
}


def query_cuda_driver() -> dict:
    """Ask libcuda directly. cuInit + attribute reads only: no context is
    created and no memory is allocated."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return {"error": f"libcuda.so.1 not loadable: {e}"}
    if cuda.cuInit(0) != 0:
        return {"error": "cuInit failed (driver not loaded or no device)"}
    ver = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(ver))
    count = ctypes.c_int()
    cuda.cuDeviceGetCount(ctypes.byref(count))
    devices = []
    for i in range(count.value):
        dev = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(dev), i) != 0:
            continue
        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, dev)
        mem = ctypes.c_size_t()
        cuda.cuDeviceTotalMem_v2(ctypes.byref(mem), dev)
        d = {"index": i, "name": name.value.decode(errors="replace"), "total_mem_bytes": mem.value}
        for k, a in CU_ATTRS.items():
            v = ctypes.c_int()
            if cuda.cuDeviceGetAttribute(ctypes.byref(v), a, dev) == 0:
                d[k] = v.value
        if "compute_capability_major" in d:
            d["compute_capability"] = f"{d['compute_capability_major']}.{d.get('compute_capability_minor', 0)}"
        if "memory_clock_rate_khz" in d and "global_memory_bus_width_bits" in d:
            # DDR: 2 transfers per clock as reported by the driver.
            d["theoretical_bandwidth_gbs"] = round(
                2 * d["memory_clock_rate_khz"] * 1e3 * d["global_memory_bus_width_bits"] / 8 / 1e9, 1)
        devices.append(d)
    v = ver.value
    return {"driver_api_version": f"{v // 1000}.{(v % 1000) // 10}", "devices": devices}


# ------------------------------------------------------------- sysfs probes

def pci_devices_sysfs(vendor="0x10de") -> list[dict]:
    out = []
    base = Path("/sys/bus/pci/devices")
    if not base.is_dir():
        return out
    for dev in sorted(base.iterdir()):
        try:
            if (dev / "vendor").read_text().strip() != vendor:
                continue
        except OSError:
            continue
        info = {"slot": dev.name}
        for f in ("device", "subsystem_vendor", "subsystem_device", "class",
                  "current_link_speed", "current_link_width", "max_link_speed",
                  "max_link_width", "power/runtime_status", "power_state"):
            try:
                info[f.replace("/", "_")] = (dev / f).read_text().strip()
            except OSError:
                pass
        drv = dev / "driver"
        info["driver"] = os.path.basename(os.readlink(drv)) if drv.is_symlink() else None
        out.append(info)
    return out


def hwmon_inventory() -> list[dict]:
    """List sensors and *report* whether fan/pwm controls exist and look
    writable. Nothing is written."""
    out = []
    base = Path("/sys/class/hwmon")
    if not base.is_dir():
        return out
    for h in sorted(base.iterdir()):
        try:
            name = (h / "name").read_text().strip()
        except OSError:
            continue
        files = sorted(p.name for p in h.iterdir() if p.is_file())
        temps = {}
        for f in files:
            if re.fullmatch(r"temp\d+_input", f):
                try:
                    temps[f] = int((h / f).read_text()) / 1000
                except (OSError, ValueError):
                    pass
        pwm = [f for f in files if re.fullmatch(r"pwm\d+(_enable)?", f)]
        out.append({
            "hwmon": h.name, "name": name, "temps_c": temps,
            "fan_inputs": [f for f in files if re.fullmatch(r"fan\d+_input", f)],
            "pwm_controls": pwm,
            "pwm_writable_by_current_user": any(os.access(h / f, os.W_OK) for f in pwm),
        })
    return out


def thermal_zones() -> list[dict]:
    out = []
    base = Path("/sys/class/thermal")
    for z in sorted(base.glob("thermal_zone*")) if base.is_dir() else []:
        try:
            out.append({"zone": z.name, "type": (z / "type").read_text().strip(),
                        "temp_c": int((z / "temp").read_text()) / 1000})
        except (OSError, ValueError):
            pass
    return out


def dmi_info() -> dict:
    out = {}
    for f in DMI_FIELDS:
        try:
            out[f] = Path(f"/sys/class/dmi/id/{f}").read_text().strip()
        except OSError:
            pass
    return out


def idle_telemetry(seconds: int, collector: Collector) -> dict:
    """Passive 1 Hz sample of an idle GPU. Puts no load on the device."""
    if seconds <= 0 or shutil.which("nvidia-smi") is None:
        return {}
    fields = "timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate,utilization.gpu,fan.speed"
    try:
        p = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
                            "-l", "1"], capture_output=True, text=True, timeout=seconds + 2,
                           stdin=subprocess.DEVNULL)
        text = p.stdout
    except subprocess.TimeoutExpired as e:
        text = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    except OSError:
        return {}
    collector._save("idle_telemetry", text)
    rows = [[c.strip() for c in l.split(",")] for l in text.strip().splitlines() if l.strip()]
    if not rows:
        return {}

    def col(i):
        vals = [_num(r[i]) for r in rows if len(r) > i]
        return [v for v in vals if v is not None]

    summary = {"samples": len(rows), "fields": fields.split(",")}
    for i, name in [(1, "temperature_c"), (2, "power_w"), (3, "sm_mhz"), (4, "mem_mhz")]:
        v = col(i)
        if v:
            summary[name] = {"min": min(v), "max": max(v), "mean": round(sum(v) / len(v), 2)}
    summary["pstates_seen"] = sorted({r[5] for r in rows if len(r) > 5})
    return summary


# ------------------------------------------------------------ assembly

def build_profile(c: Collector, idle_seconds: int) -> dict:
    os_rel = parse_os_release(c.read("os_release", READABLE_FILES["os_release"]))
    meminfo = parse_meminfo(c.read("proc_meminfo", READABLE_FILES["proc_meminfo"]))
    cpuinfo = c.read("proc_cpuinfo", READABLE_FILES["proc_cpuinfo"])
    cmdline = c.read("proc_cmdline", READABLE_FILES["proc_cmdline"])
    nv_proc = c.read("nvidia_proc_version", READABLE_FILES["nvidia_proc_version"])
    lscpu = parse_lscpu(c.run("lscpu_json"))
    smi = parse_nvidia_smi_xml(c.run("nvidia_smi_xml"))
    smi_query = c.run("nvidia_smi_query")
    c.run("nvidia_smi_supported_clocks")
    lspci_nv = c.run("lspci_nvidia")
    lspci_disp = "\n".join(filter(None, [c.run("lspci_vga"), c.run("lspci_3d")]))
    lsmod = c.run("lsmod") or ""
    modinfo = c.run("modinfo_nvidia")
    nvcc = c.run("nvcc_version")
    ldconf = parse_ldconfig(c.run("ldconfig"))
    vk = parse_vulkaninfo_summary(c.run("vulkaninfo_summary"))
    sensors = c.run("sensors")
    prime = c.run("prime_select")
    dimms = parse_dmidecode_memory(c.run("dmidecode_memory"))
    lsblk_raw = c.run("lsblk_json")
    df_raw = c.run("df_root")
    swap = c.run("swapon")
    cuda = query_cuda_driver()
    if c.raw_dir:
        (c.raw_dir / "cuda_driver_api.json").write_text(json.dumps(cuda, indent=2))

    gpu = (smi.get("gpus") or [None])[0]
    cu_dev = (cuda.get("devices") or [None])[0]
    pci_nv = pci_devices_sysfs()
    loaded = {l.split()[0] for l in lsmod.splitlines()[1:] if l.strip()}

    def from_gpu(key, src="nvidia-smi -q -x"):
        if gpu is None:
            return missing(src)
        v = gpu.get(key)
        return fact(v, "verified" if v is not None else "not_detected", src)

    p: dict = {"schema_version": SCHEMA_VERSION}
    p["audit"] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "auditor": "lab/hardware_audit.py",
        "read_only": True,
        "sudo_used": c.allow_sudo,
        "raw_evidence_dir": str(c.raw_dir) if c.raw_dir else None,
    }

    p["machine"] = {
        "dmi": fact(dmi_info() or None, "verified" if dmi_info() else "unavailable", "/sys/class/dmi/id"),
    }
    p["os"] = {
        "distribution": fact(os_rel.get("PRETTY_NAME"), "verified" if os_rel else "unavailable", "/etc/os-release"),
        "kernel": fact(platform.release(), "verified", "uname -r"),
        "kernel_cmdline": fact(cmdline.strip() if cmdline else None,
                               "verified" if cmdline else "unavailable", "/proc/cmdline"),
        "headless_hint": fact(not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
                              "verified", "DISPLAY/WAYLAND_DISPLAY env of audit process"),
    }

    # --- GPU identity
    nv_pci = [d for d in pci_nv if d.get("class", "").startswith("0x03")]
    p["gpu"] = {
        "pci_devices_sysfs": fact(nv_pci or None, "verified" if nv_pci else "not_detected",
                                  "/sys/bus/pci/devices/*"),
        "lspci": fact(lspci_nv.strip() if lspci_nv else None,
                      "verified" if lspci_nv else "unavailable", "lspci -nn -d 10de:"),
        "all_display_controllers": fact(lspci_disp.strip() or None,
                                        "verified" if lspci_disp.strip() else "unavailable",
                                        "lspci -nn -d ::0300/::0302 (hybrid graphics check)"),
        "name": from_gpu("product_name"),
        "pci_device_id": from_gpu("pci_device_id"),
        "pci_subsystem_id": from_gpu("pci_sub_system_id"),
        "architecture": from_gpu("product_architecture"),
        "vbios_version": from_gpu("vbios_version"),
        "pcie_link": from_gpu("pcie_link"),
        "compute_capability": (fact(cu_dev["compute_capability"], "verified", "CUDA driver API cuDeviceGetAttribute")
                               if cu_dev and "compute_capability" in cu_dev else missing("CUDA driver API")),
        "cuda_device_attributes": (fact(cu_dev, "verified", "CUDA driver API (libcuda.so.1)")
                                   if cu_dev else fact(cuda.get("error"), "unavailable", "CUDA driver API")),
        "vram_total_mib": from_gpu("fb_memory_total_mib"),
        "vram_used_at_idle_mib": from_gpu("fb_memory_used_mib"),
        "persistence_mode": from_gpu("persistence_mode"),
        "compute_mode": from_gpu("compute_mode"),
        "runtime_power_state": fact([{"slot": d["slot"], "runtime_status": d.get("power_runtime_status"),
                                      "power_state": d.get("power_state")} for d in nv_pci] or None,
                                    "verified" if nv_pci else "not_detected", "sysfs power/runtime_status"),
    }

    # --- Driver / CUDA / Vulkan
    nvidia_loaded = "nvidia" in loaded
    p["driver"] = {
        "nvidia_driver_version": fact(smi.get("driver_version") or (modinfo or "").strip() or None,
                                      "verified" if (smi.get("driver_version") or modinfo) else "not_detected",
                                      "nvidia-smi / modinfo"),
        "proc_driver_version": fact(nv_proc.strip() if nv_proc else None,
                                    "verified" if nv_proc else "not_detected", "/proc/driver/nvidia/version"),
        "kernel_modules_loaded": fact(sorted(m for m in loaded if m.startswith(("nvidia", "nouveau", "i915", "drm"))),
                                      "verified" if lsmod else "unavailable", "lsmod"),
        "nvidia_module_loaded": fact(nvidia_loaded, "verified" if lsmod else "unavailable", "lsmod"),
        "nouveau_module_loaded": fact("nouveau" in loaded, "verified" if lsmod else "unavailable", "lsmod"),
        "prime_profile": fact(prime.strip() if prime else None, "verified" if prime else "unavailable",
                              "prime-select query"),
    }
    p["cuda"] = {
        "max_cuda_supported_by_driver": fact(smi.get("cuda_version") or cuda.get("driver_api_version"),
                                             "verified" if (smi.get("cuda_version") or cuda.get("driver_api_version"))
                                             else "unavailable", "nvidia-smi / cuDriverGetVersion"),
        "nvcc": fact(re.search(r"release ([\d.]+)", nvcc).group(1) if nvcc and "release" in nvcc else None,
                     "verified" if nvcc else "not_detected", "nvcc --version"),
        "toolkit_dirs": fact(sorted(str(x) for x in Path("/usr/local").glob("cuda*")) or None,
                             "verified", "/usr/local/cuda*"),
        "libraries": fact(ldconf, "verified" if any(ldconf.values()) else "not_detected", "ldconfig -p"),
        "toolkit_note": fact("CUDA 13.x cannot compile for sm_61; Pascal needs a 12.x toolkit.",
                             "derived", "docs/assessment.md (developer-reported, NVIDIA release notes)"),
    }
    p["vulkan"] = {
        "summary": fact(vk or None, "verified" if vk else "unavailable", "vulkaninfo --summary"),
        "icd_files": fact(sorted(str(x) for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d")
                                 for x in Path(d).glob("*.json")) or None, "verified", "vulkan icd.d"),
    }

    # --- Clocks / power / thermal
    p["clocks"] = {
        "current_mhz": from_gpu("clocks_mhz"),
        "max_mhz": from_gpu("max_clocks_mhz"),
        "default_application_clocks_mhz": from_gpu("default_applications_clocks_mhz"),
        "supported_clocks": from_gpu("supported_clocks"),
        "performance_state_at_audit": from_gpu("performance_state"),
        "clock_event_reasons_at_audit": from_gpu("clock_event_reasons"),
        "behaviour_under_load": missing("M1 load test", "unavailable"),
    }
    p["power"] = {"readings_and_limits": from_gpu("power"),
                  "query_csv": fact(smi_query.strip() if smi_query else None,
                                    "verified" if smi_query else "unavailable", "nvidia-smi --query-gpu")}
    p["thermal"] = {
        "gpu_temperature_c": from_gpu("temperature_c"),
        "gpu_slowdown_threshold_c": from_gpu("temp_slowdown_c"),
        "gpu_shutdown_threshold_c": from_gpu("temp_shutdown_c"),
        "gpu_max_operating_c": from_gpu("temp_max_gpu_c"),
        "gpu_target_temp_c": from_gpu("target_temp_c"),
        "thermal_zones": fact(thermal_zones() or None, "verified" if thermal_zones() else "unavailable",
                              "/sys/class/thermal"),
        "idle_telemetry": fact(idle_telemetry(idle_seconds, c) or None,
                               "verified" if idle_seconds > 0 and gpu else "unavailable",
                               f"nvidia-smi -l 1 for {idle_seconds}s (passive)"),
        "sustained_load_behaviour": missing("M1 load test", "unavailable"),
    }
    hw = hwmon_inventory()
    p["cooling"] = {
        "gpu_fan_reading": from_gpu("fan_speed"),
        "hwmon": fact(hw or None, "verified" if hw else "unavailable", "/sys/class/hwmon"),
        "dell_smm_present": fact(any(h["name"] == "dell_smm" for h in hw), "verified", "/sys/class/hwmon"),
        "sensors_json_available": fact(bool(sensors), "verified", "sensors -j"),
    }

    # --- CPU / memory / storage
    flags = parse_cpu_flags(cpuinfo)
    p["cpu"] = {
        "model": fact(lscpu.get("Model name"), "verified" if lscpu else "unavailable", "lscpu -J"),
        "lscpu": fact(lscpu or None, "verified" if lscpu else "unavailable", "lscpu -J"),
        "simd_flags": fact([f for f in flags if f in ("sse4_2", "avx", "avx2", "fma", "f16c", "avx512f",
                                                       "avx512bw", "avx512_vnni", "avx_vnni")],
                           "verified" if flags else "unavailable", "/proc/cpuinfo flags"),
    }
    p["memory"] = {
        "total_bytes": fact(meminfo.get("MemTotal"), "verified" if meminfo else "unavailable", "/proc/meminfo"),
        "available_bytes_at_audit": fact(meminfo.get("MemAvailable"), "verified" if meminfo else "unavailable",
                                         "/proc/meminfo"),
        "swap_total_bytes": fact(meminfo.get("SwapTotal"), "verified" if meminfo else "unavailable", "/proc/meminfo"),
        "swap_devices": fact(swap.strip() if swap else None, "verified" if swap is not None else "unavailable",
                             "swapon --show"),
        "dimms": fact(dimms or None, "verified" if dimms else ("unavailable" if not c.allow_sudo else "not_detected"),
                      "sudo -n dmidecode -t memory" + ("" if c.allow_sudo else " (skipped: rerun with --sudo)")),
        "channel_config": fact(
            (f"{sum(1 for d in dimms if d['populated'])} of {len(dimms)} slots populated") if dimms else None,
            "derived" if dimms else "unavailable", "dmidecode"),
    }
    try:
        blk = json.loads(lsblk_raw)["blockdevices"] if lsblk_raw else None
    except (ValueError, KeyError):
        blk = None
    p["storage"] = {
        "block_devices": fact(blk, "verified" if blk else "unavailable", "lsblk -J"),
        "root_fs": fact(df_raw.strip() if df_raw else None, "verified" if df_raw else "unavailable", "df /"),
    }

    p["software"] = {
        "tools_present": fact({t: shutil.which(t) for t in TOOLS_OF_INTEREST}, "verified", "PATH lookup"),
        "python": fact(sys.version.split()[0], "verified", "sys.version"),
    }
    p["monitoring_interfaces"] = fact({
        "nvml": bool(ldconf.get("libnvidia-ml.so")),
        "nvidia_smi": bool(shutil.which("nvidia-smi")),
        "cuda_driver_api": "devices" in cuda,
        "hwmon_sensors": [h["name"] for h in hw],
        "thermal_zones": len(thermal_zones()),
        "lm_sensors": bool(shutil.which("sensors")),
        "nvtop": bool(shutil.which("nvtop")),
        "dcgm": bool(shutil.which("dcgmi")),
    }, "verified", "derived from probes above")

    p["spec_claims_check"] = check_claims(p)
    p["unknowns"] = [k for k, v in walk_facts(p) if v["status"] in ("unavailable", "not_detected")]
    p["probe_log"] = c.log
    return p


def walk_facts(d, prefix=""):
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and set(v) == {"value", "status", "source"}:
            yield path, v
        elif isinstance(v, dict) and k not in ("spec_claims_check",):
            yield from walk_facts(v, path)


def check_claims(p: dict) -> dict:
    """Compare the project spec's assumptions with what was observed."""
    def verdict(observed, ok):
        if observed is None:
            return {"observed": None, "result": "unverified"}
        return {"observed": observed, "result": "confirmed" if ok else "contradicted"}

    dmi = p["machine"]["dmi"]["value"] or {}
    product = dmi.get("product_name")
    cu = p["gpu"]["cuda_device_attributes"]["value"]
    name = p["gpu"]["name"]["value"] or (cu.get("name") if isinstance(cu, dict) else None)
    arch = p["gpu"]["architecture"]["value"]
    cc = p["gpu"]["compute_capability"]["value"]
    ram = p["memory"]["total_bytes"]["value"]
    dist = p["os"]["distribution"]["value"]
    return {
        "machine": {"claim": SPEC_CLAIMS["machine"],
                    **verdict(product, bool(product) and "G3" in product and "17" in product)},
        "gpu_model": {"claim": "GeForce GTX 1060",
                      **verdict(name, bool(name) and SPEC_CLAIMS["gpu_name_contains"] in name)},
        "architecture": {"claim": SPEC_CLAIMS["architecture"],
                         **verdict(arch, bool(arch) and arch.lower() == "pascal")},
        "compute_capability": {"claim": SPEC_CLAIMS["compute_capability"],
                               **verdict(cc, cc == SPEC_CLAIMS["compute_capability"])},
        # MemTotal excludes firmware/iGPU reservations, so 8 GB installed reads ~7.6-7.9 GiB.
        "system_ram": {"claim": "8 GB",
                       **verdict(round(ram / 2**30, 2) if ram else None,
                                 bool(ram) and 7.0 * 2**30 <= ram <= 8.0 * 2**30)},
        "os": {"claim": SPEC_CLAIMS["os"], **verdict(dist, bool(dist) and "Ubuntu" in dist)},
        "vram": {"claim": "to be discovered", **verdict(p["gpu"]["vram_total_mib"]["value"], True)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="research/hardware_profile.json")
    ap.add_argument("--raw-dir", default=None,
                    help="where to keep raw evidence (default research/audit_raw/<timestamp>)")
    ap.add_argument("--idle-seconds", type=int, default=10,
                    help="passive idle telemetry sample length; 0 disables")
    ap.add_argument("--sudo", action="store_true",
                    help="allow `sudo -n dmidecode -t memory` (non-interactive, read-only)")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw = Path(args.raw_dir or f"research/audit_raw/{stamp}")
    c = Collector(raw, timeout=args.timeout, allow_sudo=args.sudo)
    profile = build_profile(c, args.idle_seconds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, sort_keys=False) + "\n")

    claims = profile["spec_claims_check"]
    print(f"hardware profile written to {out} (raw evidence in {raw})")
    for k, v in claims.items():
        print(f"  {k:20s} {v['result']:13s} observed={v['observed']!r}")
    print(f"  {len(profile['unknowns'])} facts unavailable or not detected (see 'unknowns')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
