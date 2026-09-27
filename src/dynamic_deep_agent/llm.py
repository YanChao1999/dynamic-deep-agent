"""OpenAI-compatible chat client and token usage tracking.

Uses the Chat Completions HTTP API (``/v1/chat/completions``) so it works
with OpenAI, Azure OpenAI (chat URL), and compatible local gateways.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TokenUsage:
    """Cumulative chat-completion token counts."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, *, prompt: int = 0, completion: int = 0, total: int | None = None) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total if total is not None else prompt + completion
        self.calls += 1

    def merge(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }

    def summary(self) -> str:
        return (
            f"tokens: prompt={self.prompt_tokens} completion={self.completion_tokens} "
            f"total={self.total_tokens} calls={self.calls}"
        )


@dataclass
class ChatResult:
    """One chat completion result."""

    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Minimal OpenAI-style chat interface."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        ...


class OpenAIChatClient:
    """HTTP client for OpenAI-style ``chat/completions``.

    Env defaults: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``OPENAI_MODEL``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            if base_url is not None
            else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model if model is not None else os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; pass api_key= or export the env var"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc

        content = raw["choices"][0]["message"]["content"] or ""
        usage_raw = raw.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
            calls=1,
        )
        return ChatResult(content=content, usage=usage, raw=raw)


@dataclass
class FakeLLMClient:
    """Deterministic client for tests — returns scripted contents."""

    responses: list[str]
    _i: int = 0
    prompt_tokens_per_call: int = 10
    completion_tokens_per_call: int = 5

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        if self._i >= len(self.responses):
            raise RuntimeError("FakeLLMClient: no more scripted responses")
        content = self.responses[self._i]
        self._i += 1
        usage = TokenUsage(
            prompt_tokens=self.prompt_tokens_per_call,
            completion_tokens=self.completion_tokens_per_call,
            total_tokens=self.prompt_tokens_per_call + self.completion_tokens_per_call,
            calls=1,
        )
        return ChatResult(content=content, usage=usage, raw={"fake": True})
