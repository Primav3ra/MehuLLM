"""Minimal Ollama client.

Plain httpx against the REST API rather than the `ollama` package: one less
dependency, and the whole surface we need is two endpoints. This module is
shared by the offline draft generator (`pipeline/neutralize.py`) and, later,
the runtime voice layer.

Qwen3 is a hybrid-thinking model. Left alone it emits `<think>...</think>`
blocks, which would end up baked into training data and then leak into
WhatsApp replies at inference. `strip_thinking()` removes them and
`OllamaClient.generate(no_think=True)` suppresses them at the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    num_ctx: int = 2048  # the voice model sees a draft + exemplars, never a conversation

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.host, timeout=self.timeout)

    # -- health ---------------------------------------------------------

    def is_up(self) -> bool:
        try:
            with self._client() as c:
                return c.get("/api/version").status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self, model: str | None = None) -> bool:
        want = (model or self.model).split(":")[0]
        try:
            with self._client() as c:
                tags = c.get("/api/tags").json().get("models", [])
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

    # -- generation -----------------------------------------------------

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

        owned = client is None
        c = client or self._client()
        try:
            r = c.post("/api/generate", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"ollama returned {r.status_code}: {r.text[:300]}")
            return strip_thinking(r.json().get("response", ""))
        except httpx.HTTPError as e:
            raise OllamaError(f"ollama request failed: {e}") from e
        finally:
            if owned:
                c.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        num_predict: int = 96,
        no_think: bool = True,
        client: httpx.Client | None = None,
    ) -> str:
        """Chat-completions call.

        Use this, NOT `generate`, for few-shot prompting. On the completion
        endpoint a small model treats worked examples as text to CONTINUE --
        observed concretely: given three Hinglish->English examples it emitted
        all three answers concatenated with the real one. Structuring them as
        alternating user/assistant turns fixes it, because the chat template
        marks where each answer ends.
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": not no_think,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
                "top_p": 0.9,
            },
        }
        owned = client is None
        c = client or self._client()
        try:
            r = c.post("/api/chat", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"ollama returned {r.status_code}: {r.text[:300]}")
            return strip_thinking(r.json().get("message", {}).get("content", ""))
        except httpx.HTTPError as e:
            raise OllamaError(f"ollama request failed: {e}") from e
        finally:
            if owned:
                c.close()
