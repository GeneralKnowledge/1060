"""Claude via the official Anthropic SDK."""
from __future__ import annotations

import os

from lab.providers import Provider, ProviderError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, cfg: dict):
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError("pip install anthropic to use the anthropic provider") from e
        self.anthropic = anthropic
        key_env = cfg.get("api_key_env")
        key = os.environ.get(key_env) if key_env else None
        # With no key in the env the SDK falls back to its own credential chain.
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.model = cfg.get("model", "claude-opus-5-5")
        self.effort = cfg.get("effort", "high")
        self.max_tokens = int(cfg.get("max_tokens", 16000))
        self.fallbacks = cfg.get("fallbacks", True)

    def complete(self, system: str, messages: list[dict]) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=messages, output_config={"effort": self.effort})
        try:
            if self.fallbacks:
                # Server-side fallback re-runs a refused request on another model.
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except self.anthropic.RateLimitError as e:
            raise ProviderError(f"rate limited: {e}") from e
        except self.anthropic.APIStatusError as e:
            raise ProviderError(f"API error {e.status_code}: {e.message}") from e
        except self.anthropic.APIConnectionError as e:
            raise ProviderError(f"connection error: {e}") from e
        if resp.stop_reason == "refusal":
            raise ProviderError("model declined the request")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason == "max_tokens":
            raise ProviderError("reply truncated at max_tokens; raise provider.max_tokens")
        return text
