"""Voice layer + fact-invariant firewall."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import regex

from mehullm.voice.client import OllamaClient

_INVARIANTS = [
    # Fact citations, FIRST because they are what makes an answer auditable.
    regex.compile(r"\[F\d+\]"),
    regex.compile(r"https?://\S+"),
    regex.compile(r"\b[\w.\-+]+@[\w\-]+\.\w{2,}\b"),
    regex.compile(r"@[A-Za-z0-9_\-]{2,}"),
    regex.compile(r"⟦PII_[A-Z]+_\d+⟧"),
    # ':' belongs in the separator class. Without it "4:30" decomposed into the.
    regex.compile(r"\d+(?:[.,:]\d+)*"),
    regex.compile(r"\"[^\"]{3,}\""),
]

MAX_EXPANSION = 2.5


def extract_invariants(text: str) -> set[str]:
    out: set[str] = set()
    for pat in _INVARIANTS:
        out |= {m.group(0) for m in pat.finditer(text)}
    return out


def voice_if_available(host: str, model: str) -> VoiceRewriter | None:
    """None when ollama is down or the model is not installed -- both are normal states, neither is fatal."""
    from mehullm.voice.client import OllamaClient

    try:
        if OllamaClient(model=model, host=host).has_model():
            return VoiceRewriter(host=host, model=model)
    except Exception:
        return None
    return None


@dataclass
class VoiceRewriter:
    host: str = "http://localhost:11434"
    model: str = "mehul-voice"
    # LEAVE THIS EMPTY. Measured on v1: populating it inverts person and copies.
    context: str = ""

    # 0.3, not the 0.85 the plan guessed before there was a model to measure.
    temperature: float = 0.3

    async def rewrite(self, draft: str) -> tuple[str, bool]:
        """Returns (text, ok)."""
        client = OllamaClient(model=self.model, host=self.host, num_ctx=2048)
        prompt = f"<context>\n{self.context}\n</context>\n<draft>\n{draft.strip()}\n</draft>"
        voiced = await asyncio.to_thread(
            client.generate, prompt, temperature=self.temperature, num_predict=200
        )
        voiced = (voiced or "").strip()
        if not voiced:
            return draft, False

        facts = extract_invariants(draft)
        if not facts.issubset(extract_invariants(voiced)):
            return draft, False
        if len(voiced) > MAX_EXPANSION * len(draft):
            return draft, False
        return voiced, True
