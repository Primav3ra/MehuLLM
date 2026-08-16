"""Voice layer + fact-invariant firewall.

A 1.7B model rewriting for style WILL occasionally drop a digit or mangle a URL.
So every number, URL, email, @handle, date and quoted string in the draft must
survive into the voiced output, or the draft is used instead.

This is a concrete hallucination firewall at a system boundary, and it is the
reason it is safe to let a small local model touch the user-facing answer at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import regex

from mehullm.voice.client import OllamaClient

_INVARIANTS = [
    regex.compile(r"https?://\S+"),
    regex.compile(r"\b[\w.\-+]+@[\w\-]+\.\w{2,}\b"),
    regex.compile(r"@[A-Za-z0-9_\-]{2,}"),
    regex.compile(r"⟦PII_[A-Z]+_\d+⟧"),
    regex.compile(r"\d+(?:[.,]\d+)*"),
    regex.compile(r"\"[^\"]{3,}\""),
]

MAX_EXPANSION = 2.5


def extract_invariants(text: str) -> set[str]:
    out: set[str] = set()
    for pat in _INVARIANTS:
        out |= {m.group(0) for m in pat.finditer(text)}
    return out


@dataclass
class VoiceRewriter:
    host: str = "http://localhost:11434"
    model: str = "mehul-voice"
    context: str = ""

    async def rewrite(self, draft: str) -> tuple[str, bool]:
        """Returns (text, ok). ok=False means the invariant check failed and the
        caller should fall back to the draft."""
        client = OllamaClient(model=self.model, host=self.host, num_ctx=2048)
        prompt = (
            f"<context>\n{self.context}\n</context>\n<draft>\n{draft.strip()}\n</draft>"
        )
        voiced = await asyncio.to_thread(
            client.generate, prompt, temperature=0.85, num_predict=200
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
