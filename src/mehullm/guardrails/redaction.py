"""Reversible PII vault for the runtime path."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import regex

from mehullm.pipeline.pii import _ORDER, PATTERNS

TOKEN_RE = re.compile(r"⟦PII_[A-Z]+_\d+⟧")

# Hard-redacted, never vaulted: a recoverable placeholder could put a live
# credential back into a later prompt.
SECRET_RES: list[tuple[str, regex.Pattern]] = [
    ("openai_key", regex.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("github_pat", regex.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws_key", regex.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_key", regex.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack_token", regex.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}")),
    ("private_key", regex.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", regex.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
]


@dataclass
class Vault:
    """Per-run placeholder store. Lives and dies with the Run."""

    fwd: dict[str, str] = field(default_factory=dict)  # real -> token
    rev: dict[str, str] = field(default_factory=dict)  # token -> real
    counts: Counter[str] = field(default_factory=Counter)
    secrets_found: Counter[str] = field(default_factory=Counter)

    def redact(self, text: str) -> str:
        if not text:
            return text
        for name in _ORDER:
            kind = "OTP" if name.startswith("OTP") else name

            def _sub(m: regex.Match, k: str = kind) -> str:
                raw = m.group(0)
                if raw not in self.fwd:
                    self.counts[k] += 1
                    tok = f"⟦PII_{k}_{self.counts[k]}⟧"
                    self.fwd[raw] = tok
                    self.rev[tok] = raw
                return self.fwd[raw]

            text = PATTERNS[name].sub(_sub, text)
        return text

    def scrub_secrets(self, text: str) -> tuple[str, list[str]]:
        """Hard-redact credentials. Not reversible, deliberately."""
        found: list[str] = []
        for name, pat in SECRET_RES:
            text, n = pat.subn("<REDACTED_SECRET>", text)
            if n:
                found.append(name)
                self.secrets_found[name] += n
        return text, found

    def rehydrate(self, obj: Any) -> Any:
        """Walk a tool-argument structure and restore real values."""
        if isinstance(obj, str):
            return TOKEN_RE.sub(lambda m: self.rev.get(m.group(0), m.group(0)), obj)
        if isinstance(obj, dict):
            return {k: self.rehydrate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.rehydrate(v) for v in obj]
        return obj

    @property
    def total(self) -> int:
        return sum(self.counts.values())
