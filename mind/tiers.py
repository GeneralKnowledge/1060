"""Language tiers: local llama-server instances the mind's processes can recruit.

A tier never decides what to think about. A process hands it one question built
from the mind's own memories and gets one structured answer back. Calls run in a
background worker so the mind keeps perceiving while a tier thinks; the answer is
picked up on a later tick (a thought waiting on it is simply not finished yet).

There is no persona or system prompt. Prompts state the task and the mind's own
material (its memories, beliefs and earlier notes) and ask for JSON.

Tiers
  fast        always loaded, CPU            (LFM2.5-350M)
  reflective  loaded on demand, CPU,        (LFM2.5-1.2B-Thinking)
              unloaded when idle or when host RAM runs low
  deep        GPU, A3 (not wired here)
"""
from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def mem_available_mb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024
    except OSError:
        pass
    return None


def parse_json(text: str):
    """The last JSON object in a reply (thinking models may write before it)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    found, i = None, 0
    while (start := text.find("{", i)) != -1:
        depth, end = 0, None
        for j in range(start, len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                end = j
                break
        if end is None:
            break
        try:
            found = json.loads(text[start:end + 1])
            i = end + 1
        except ValueError:
            i = start + 1
    return found


class Tier:
    def __init__(self, name: str, cfg: dict, repo: Path, lab: Path | None = None):
        self.name, self.cfg = name, cfg
        self.url = cfg.get("url") or f"http://127.0.0.1:{cfg.get('port', 8110)}"
        self.model = (repo / cfg["model"]) if cfg.get("model") else None
        self.server_bin = (lab or repo) / cfg.get("llama_server", "backends/builds/d2db58ba90808f32/build/bin/llama-server")
        self.proc = None
        self.last_used = 0.0
        self.calls = 0

    # -- process management (only for tiers we start ourselves)
    def healthy(self):
        try:
            urllib.request.urlopen(self.url + "/health", timeout=2)
            return True
        except Exception:
            return False

    def available(self):
        return self.model is not None and self.model.exists() and self.server_bin.exists() or bool(self.cfg.get("url"))

    def ensure(self, timeout=180) -> bool:
        if self.healthy():
            return True
        if self.cfg.get("url") or not self.available():
            return False
        free = mem_available_mb()
        if free is not None and free < self.cfg.get("min_free_mb", 1500):
            return False
        port = self.url.rsplit(":", 1)[1]
        args = [str(self.server_bin), "-m", str(self.model), "--port", port, "-ngl", str(self.cfg.get("ngl", 0)),
                "-t", str(self.cfg.get("threads", 2)), "-c", str(self.cfg.get("ctx", 4096)), "-np", "1"]
        self.proc = subprocess.Popen(["nice", "-n", "10", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                self.proc = None
                return False
            if self.healthy():
                return True
            time.sleep(0.5)
        self.stop()
        return False

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    # -- one question, one answer
    def ask(self, prompt: str, schema: dict | None = None, max_tokens: int | None = None) -> dict:
        body = {"messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or self.cfg.get("max_tokens", 512),
                "temperature": self.cfg.get("temperature", 0.3)}
        if schema and self.cfg.get("json_schema", True):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.cfg.get("timeout_s", 300)) as r:
            data = json.load(r)
        self.last_used = time.time()
        self.calls += 1
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        return dict(content=content, json=parse_json(content), seconds=round(time.time() - t0, 2),
                    tokens=data.get("usage", {}).get("completion_tokens"),
                    reasoning_chars=len(msg.get("reasoning_content") or ""))


class TierPool:
    """Background worker: jobs in, results out. One job at a time (CPU is shared)."""

    def __init__(self, cfg: dict, repo: Path, lab: Path | None = None):
        self.tiers = {name: Tier(name, c, repo, lab) for name, c in cfg.items() if isinstance(c, dict)}
        self.jobs: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.pending = 0
        self._stop = False
        self.thread = threading.Thread(target=self._work, daemon=True, name="tiers")
        self.thread.start()

    def has(self, name):
        t = self.tiers.get(name)
        return bool(t and t.available())

    def submit(self, tier, kind, prompt, schema=None, context=None, max_tokens=None):
        if not self.has(tier):
            return False
        self.pending += 1
        self.jobs.put((tier, kind, prompt, schema, context or {}, max_tokens))
        return True

    def _work(self):
        while not self._stop:
            try:
                tier, kind, prompt, schema, ctx, mt = self.jobs.get(timeout=0.5)
            except queue.Empty:
                self._idle_unload()
                continue
            t = self.tiers[tier]
            res = dict(tier=tier, kind=kind, context=ctx)
            try:
                if not t.ensure():
                    res["error"] = "tier unavailable (not loaded or low RAM)"
                else:
                    res.update(t.ask(prompt, schema, mt))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
                res["error"] = f"{type(e).__name__}: {e}"[:300]
            self.results.put(res)

    def _idle_unload(self):
        for t in self.tiers.values():
            idle = t.cfg.get("idle_unload_s")
            low = (mem_available_mb() or 1e9) < t.cfg.get("min_free_mb", 0) * 0.66
            if t.proc and ((idle and time.time() - t.last_used > idle) or low):
                t.stop()

    def drain(self, wait_s: float = 0.0):
        """Results ready now (optionally waiting up to wait_s for the first one)."""
        out = []
        try:
            out.append(self.results.get(timeout=wait_s) if wait_s else self.results.get_nowait())
            while True:
                out.append(self.results.get_nowait())
        except queue.Empty:
            pass
        self.pending -= len(out)
        return out

    def close(self):
        self._stop = True
        for t in self.tiers.values():
            t.stop()
