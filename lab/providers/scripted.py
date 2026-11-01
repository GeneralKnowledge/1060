"""Replays canned replies in order. For tests and dry runs."""
from __future__ import annotations

from lab.providers import Provider, ProviderError


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[tuple[str, list]] = []

    def complete(self, system, messages):
        self.calls.append((system, messages))
        if not self.replies:
            raise ProviderError("scripted provider has no replies left")
        return self.replies.pop(0)
