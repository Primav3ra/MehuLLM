"""Minimal Ollama client -- plain httpx, two endpoints, no `ollama` package."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

__all__ = ["OllamaClient", "OllamaError", "strip_thinking"]

DEFAULT_HOST = "http://localhost:11434"

# Qwen3 emits <think>...</think>. Also tolerate an unclosed block, which
# happens whenever generation is cut off by num_predict mid-thought.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    model: str = "qwen3:1.7b"
    host: str = DEFAULT_HOST
    timeout: float = 180.0
    num_ctx: int = 2048  # a draft + exemplars, never a conversation
    keep_alive: str = "30m"

    # One shared connection: a client per call meant an SSL context per call,
    # each reading certifi off disk for a plain-HTTP localhost connection.
    _shared: httpx.Client | None = field(default=None, repr=False, compare=False)

    def _client(self) -> httpx.Client:
        if self._shared is None or self._shared.is_closed:
            try:
                self._shared = httpx.Client(base_url=self.host, timeout=self.timeout)
            except OSError as e:
                raise OllamaError(f"could not open a connection to ollama: {e}") from e
        return self._shared

    def close(self) -> None:
        if self._shared is not None and not self._shared.is_closed:
            self._shared.close()
        self._shared = None

    # No `with` on the shared client -- it would close on exit and rebuild.
    def is_up(self) -> bool:
        try:
            return self._client().get("/api/version").status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self, model: str | None = None) -> bool:
        want = (model or self.model).split(":")[0]
        try:
            tags = self._client().get("/api/tags").json().get("models", [])
        except httpx.HTTPError:
            return False
        return any(m.get("name", "").split(":")[0] == want for m in tags)

    def preflight(self) -> None:
        """Fail loudly and early rather than 12,000 requests into a batch."""
        if not self.is_up():
            raise OllamaError(
                f"Ollama is not responding at {self.host}. Start it with `ollama serve`."
            )
        if not self.has_model():
            raise OllamaError(f"Model {self.model!r} not installed. Run: ollama pull {self.model}")

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        num_predict: int = 96,
        no_think: bool = True,
        client: httpx.Client | None = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "keep_alive": self.keep_alive,
            "stream": False,
            "think": not no_think,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
                "top_p": 0.9,
            },
        }
        if system:
            payload["system"] = system

        # A caller-supplied client stays the caller's to close; the shared one is
        # never closed here -- it is reused for the life of this object.
        c = client or self._client()
        try:
            r = c.post("/api/generate", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"ollama returned {r.status_code}: {r.text[:300]}")
            return strip_thinking(r.json().get("response", ""))
        except httpx.HTTPError as e:
            raise OllamaError(f"ollama request failed: {e}") from e

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        num_predict: int = 96,
        no_think: bool = True,
        fmt: dict | str | None = None,
        client: httpx.Client | None = None,
    ) -> str:
        """Use this, not `generate`, for few-shot: on the completion endpoint a small model continues the examples instea."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "keep_alive": self.keep_alive,
            "stream": False,
            "think": not no_think,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
                "top_p": 0.9,
            },
        }
        if fmt is not None:
            # `think: false` alone does not stop qwen3 narrating its reasoning;
            # a forced schema makes prose impossible, which is what cuts latency.
            payload["format"] = fmt
        c = client or self._client()
        try:
            r = c.post("/api/chat", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"ollama returned {r.status_code}: {r.text[:300]}")
            return strip_thinking(r.json().get("message", {}).get("content", ""))
        except httpx.HTTPError as e:
            raise OllamaError(f"ollama request failed: {e}") from e
