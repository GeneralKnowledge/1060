"""Any OpenAI-compatible /chat/completions endpoint, stdlib HTTP only."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from lab.providers import Provider, ProviderError


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"

    def __init__(self, cfg: dict):
        self.base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.model = cfg["model"]
        self.key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        self.max_tokens = int(cfg.get("max_tokens", 16000))
        self.timeout = float(cfg.get("timeout_s", 600))

    def complete(self, system: str, messages: list[dict]) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, *messages]}
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ProviderError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ProviderError(f"connection error: {e}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"unexpected response: {str(data)[:300]}") from e
