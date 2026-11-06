"""Shell out to any CLI model runner (e.g. ["claude", "-p"]). Prompt on stdin."""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta, timezone

from lab.providers import Provider, ProviderError, UsageLimitError

# The claude CLI prints these on stdout, e.g. "You've hit your 5-hour limit · resets 1:50am (UTC)".
LIMIT_RE = re.compile(r"hit your .*limit|usage limit|rate limit|rate_limit|too many requests|\b429\b",
                      re.IGNORECASE)
RESET_RE = re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap]m)?\s*\(?UTC\)?",
                      re.IGNORECASE)


def reset_wait_s(text: str, now: datetime | None = None) -> float | None:
    """Seconds until the reset time named in a usage-limit message, if it names one in UTC."""
    m = RESET_RE.search(text)
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    now = now or datetime.now(timezone.utc)
    reset = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return (reset - now).total_seconds()


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
            # Some CLIs (claude -p) print their error on stdout, so keep both streams.
            detail = " | ".join(x.strip()[-300:] for x in (p.stderr, p.stdout) if x and x.strip())
            msg = f"{self.argv[0]} exit {p.returncode}: {detail or '(no output)'}"
            if LIMIT_RE.search(detail):
                raise UsageLimitError(msg, reset_wait_s(detail))
            raise ProviderError(msg)
        return p.stdout
