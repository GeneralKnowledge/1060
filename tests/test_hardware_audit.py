"""Parser tests for the M0 audit. Fixtures are synthetic, not measurements."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab"))
import hardware_audit as ha  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def test_parse_nvidia_smi_xml():
    r = ha.parse_nvidia_smi_xml((FIX / "nvidia_smi_synthetic.xml").read_text())
    assert r["driver_version"] == "580.95.05"
    g = r["gpus"][0]
    assert g["product_architecture"] == "Pascal"
    assert g["fb_memory_total_mib"] == 6144
    assert g["max_clocks_mhz"] == {"graphics": 1911.0, "sm": 1911.0, "mem": 4004.0}
    assert g["power"]["limit_w"] is None  # N/A on many laptop GPUs; must not become 0
    assert g["power"]["draw_w"] == 4.5
    assert g["temp_slowdown_c"] == 99
    assert g["clock_event_reasons"]["clocks_event_reason_gpu_idle"] == "Active"
    assert g["supported_clocks"][0]["graphics_mhz"] == [1911.0, 139.0]
    assert g["pcie_link"]["max_gen"] == "3"


def test_parse_nvidia_smi_xml_garbage():
    assert ha.parse_nvidia_smi_xml("not xml") == {}
    assert ha.parse_nvidia_smi_xml(None) == {}


def test_parse_vulkaninfo_summary():
    text = """Vulkan Instance Version: 1.3.275

Devices:
========
GPU0:
\tapiVersion         = 1.4.312
\tdriverVersion      = 580.95.5.0
\tvendorID           = 0x10de
\tdeviceID           = 0x1c20
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = NVIDIA GeForce GTX 1060 with Max-Q Design
\tdriverName         = NVIDIA
GPU1:
\tapiVersion         = 1.4.305
\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName         = llvmpipe (LLVM 19.1.1, 256 bits)
"""
    r = ha.parse_vulkaninfo_summary(text)
    assert r["instance_version"] == "1.3.275"
    assert len(r["devices"]) == 2
    assert r["devices"][0]["deviceID"] == "0x1c20"
    assert r["devices"][1]["deviceType"].endswith("CPU")


def test_parse_dmidecode_memory():
    text = """Handle 0x0010, DMI type 17, 40 bytes
Memory Device
\tSize: 8 GB
\tForm Factor: SODIMM
\tLocator: DIMM A
\tType: DDR4
\tSpeed: 2667 MT/s
\tConfigured Memory Speed: 2667 MT/s

Handle 0x0011, DMI type 17, 40 bytes
Memory Device
\tSize: No Module Installed
\tLocator: DIMM B
"""
    d = ha.parse_dmidecode_memory(text)
    assert [x["populated"] for x in d] == [True, False]
    assert d[0]["Type"] == "DDR4"


def test_meminfo_and_flags():
    assert ha.parse_meminfo("MemTotal:  8000000 kB\nFoo: bar")["MemTotal"] == 8000000 * 1024
    assert "avx2" in ha.parse_cpu_flags("flags\t\t: fpu avx2 fma\n")


def test_whitelist_is_read_only():
    """Guard rail: no setter flags may enter the command whitelist."""
    forbidden = {"-pl", "-lgc", "-lmc", "-rgc", "-rmc", "-pm", "-ac", "-rac", "-c", "-e",
                 "--power-limit", "--lock-gpu-clocks", "-r", "--gpu-reset"}
    for argv in ha.READ_ONLY_COMMANDS.values():
        if argv[0] == "nvidia-smi":
            assert not forbidden & set(argv), argv
        if argv[0] == "sudo":
            assert argv[1] == "-n" and argv[2] == "dmidecode"


def test_full_run_without_gpu(tmp_path):
    """The audit must complete and stay honest on a machine with no NVIDIA GPU."""
    out = tmp_path / "profile.json"
    assert ha.main(["--out", str(out), "--raw-dir", str(tmp_path / "raw"), "--idle-seconds", "0"]) == 0
    p = json.loads(out.read_text())
    assert p["audit"]["read_only"] is True
    for path, f in ha.walk_facts(p):
        assert f["status"] in {"verified", "derived", "not_detected", "unavailable"}, path
        if f["status"] in ("not_detected", "unavailable"):
            assert path in p["unknowns"]
