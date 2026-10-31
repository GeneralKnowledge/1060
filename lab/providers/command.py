"""Shell out to any CLI model runner (e.g. ["claude", "-p"]). Prompt on stdin."""
from __future__ import annotations

import subprocess

from lab.providers import Provider, ProviderError


class CommandProvider(Provider):
    name = "command"

    def __init__(self, cfg: dict):
        self.argv = cfg["argv"]
        self.timeout = float(cfg.get("timeout_s", 1800))

    def complete(self, system: str, messages: list[dict]) -> str:
        prompt = system + "\n\n" + "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        try:
            p = subprocess.run(self.argv, input=prompt, capture_output=True, text=True,
                               timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ProviderError(str(e)) from e
        if p.returncode != 0:
            raise ProviderError(f"{self.argv[0]} exit {p.returncode}: {p.stderr[-300:]}")
        return p.stdout
