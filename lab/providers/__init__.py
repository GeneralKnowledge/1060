"""LLM providers behind one interface: complete(system, messages) -> str.

kind = "anthropic"          official Anthropic SDK (pip install anthropic)
kind = "openai_compatible"  any /chat/completions endpoint (OpenAI, OpenRouter, vLLM, ...)
kind = "command"            any CLI that reads a prompt on stdin and prints a reply
kind = "scripted"           canned replies, for tests and dry runs
"""
from __future__ import annotations


class ProviderError(Exception):
    pass


class Provider:
    name = "base"

    def complete(self, system: str, messages: list[dict]) -> str:  # pragma: no cover
        raise NotImplementedError


def make_provider(cfg: dict) -> Provider:
    kind = cfg.get("kind")
    if kind == "anthropic":
        from lab.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(cfg)
    if kind == "openai_compatible":
        from lab.providers.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(cfg)
    if kind == "command":
        from lab.providers.command import CommandProvider
        return CommandProvider(cfg)
    if kind == "scripted":
        from lab.providers.scripted import ScriptedProvider
        return ScriptedProvider(cfg.get("replies", []))
    raise ProviderError(f"unknown provider kind {kind!r}")
