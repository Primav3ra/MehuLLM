"""Irreversible PII scrubbing for the training corpus and stored traces."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import regex

__all__ = ["PATTERNS", "NameMap", "scrub", "scrub_with_stats"]

# Digit class covering ASCII and Devanagari numerals. Indian phone numbers do.
_D = r"[0-9०-९]"

PATTERNS: dict[str, regex.Pattern] = {
    "EMAIL": regex.compile(r"\b[\w.\-+]+@[\w\-]+\.[\w.\-]{2,}\b"),
    "URL": regex.compile(r"https?://\S+|\bwww\.[\w\-]+\.\w{2,}\S*"),
    # UPI handles: mehul@okicici, 9876543210@ybl.
    "UPI": regex.compile(
        r"\b[\w.\-]{2,}@(?:ok(?:icici|hdfcbank|axis|sbi)|paytm|ybl|upi|apl|ibl|axl)\b",
        regex.IGNORECASE,
    ),
    # PAN: 5 letters, 4 digits, 1 letter. Very specific, safe.
    "PAN": regex.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    # Aadhaar: exactly 12 digits, optionally spaced 4-4-4. The trailing.
    "AADHAAR": regex.compile(rf"\b{_D}{{4}}\s?{_D}{{4}}\s?{_D}{{4}}\b(?![\s\-]?{_D})"),
    # Indian mobile: optional +91/0 prefix, then 10 digits starting 6-9.
    "PHONE": regex.compile(rf"(?<!{_D})(?:\+?91[\-\s]?|0)?[6-9६-९]{_D}{{9}}(?!{_D})"),
    # Vehicle registration: MH12AB1234.
    "VEHICLE": regex.compile(rf"\b[A-Z]{{2}}{_D}{{2}}[A-Z]{{1,2}}{_D}{{4}}\b"),
    # OTP -- REQUIRES a context word. A bare 6-digit number is left alone.
    "OTP": regex.compile(
        rf"\b(?:OTP|O\.T\.P|one[\s\-]?time(?:\s+password)?|verification\s+code|"
        rf"security\s+code)\b\D{{0,25}}{_D}{{4,8}}\b",
        regex.IGNORECASE,
    ),
    "OTP_TRAILING": regex.compile(
        rf"\b{_D}{{4,8}}\b(?=\D{{0,25}}\b(?:is\s+your\s+)?(?:OTP|one[\s\-]?time|"
        rf"verification\s+code)\b)",
        regex.IGNORECASE,
    ),
    # Card numbers: 13-19 digits in 4-digit groups.
    "CARD": regex.compile(rf"\b{_D}{{4}}[\s\-]?{_D}{{4}}[\s\-]?{_D}{{4}}[\s\-]?{_D}{{1,7}}\b"),
    # PIN code -- REQUIRES a context word, for the same reason as OTP.
    "PINCODE": regex.compile(
        rf"\b(?:pin\s?code|pincode|postal\s+code|zip)\b\D{{0,10}}{_D}{{6}}\b", regex.IGNORECASE
    ),
    # Street addresses -- context-anchored, not a bare number sweep.
    "ADDR": regex.compile(
        r"\b(?:flat|plot|house|h\.?\s?no\.?|door\s?no\.?|sector|block|tower|wing)\s*"
        r"[\w/\-]{1,12}\b",
        regex.IGNORECASE,
    ),
}

# Order is load-bearing, for two independent reasons:
_ORDER = [
    "EMAIL",
    "URL",
    "UPI",
    "PAN",
    "VEHICLE",
    "OTP",
    "OTP_TRAILING",
    "CARD",
    "AADHAAR",
    "PHONE",
    "PINCODE",
    "ADDR",
]


def scrub(text: str) -> str:
    """Replace PII with stable ``<TYPE>`` placeholders. Style is untouched."""
    return scrub_with_stats(text)[0]


# Two-tier OTP handling. The patterns above need the keyword and the digits to.
_OTP_KEYWORD_RE = regex.compile(
    r"\b(?:OTP|O\.T\.P|one[\s\-]?time\s*(?:password|code|pin)|verification\s+code)\b",
    regex.IGNORECASE,
)
_SHORT_DIGIT_RUN = regex.compile(rf"(?<!{_D}){_D}{{4,8}}(?!{_D})")


def scrub_with_stats(text: str) -> tuple[str, Counter[str]]:
    """Scrub and report what was found -- drives the week-3 manual audit gate."""
    found: Counter[str] = Counter()
    for name in _ORDER:
        key = "OTP" if name.startswith("OTP") else name
        placeholder = f"<{key}>"

        def _sub(m: regex.Match, k=key, p=placeholder) -> str:
            found[k] += 1
            return p

        text = PATTERNS[name].sub(_sub, text)

    if _OTP_KEYWORD_RE.search(text):

        def _otp_sub(m: regex.Match) -> str:
            found["OTP"] += 1
            return "<OTP>"

        text = _SHORT_DIGIT_RUN.sub(_otp_sub, text)

    return text, found


# Consistent name pseudonymisation.


@dataclass
class NameMap:
    """Stable ``Rohan -> Person_A`` mapping across the whole corpus."""

    mapping: dict[str, str] = field(default_factory=dict)
    keep: set[str] = field(default_factory=set)  # names left verbatim (the owner)
    _next: int = 0

    _ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def _alias(self) -> str:
        n, letters = self._next, ""
        while True:
            letters = self._ALPHABET[n % 26] + letters
            n = n // 26 - 1
            if n < 0:
                break
        self._next += 1
        return f"Person_{letters}"

    def alias_for(self, name: str) -> str:
        key = name.strip().casefold()
        if key in {k.casefold() for k in self.keep}:
            return name
        if key not in self.mapping:
            self.mapping[key] = self._alias()
        return self.mapping[key]

    def pseudonymize(self, text: str, names: list[str]) -> str:
        """Replace known names in free text, longest-first to avoid partials."""
        for name in sorted(names, key=len, reverse=True):
            if not name.strip() or name in self.keep:
                continue
            pattern = regex.compile(rf"\b{regex.escape(name)}\b", regex.IGNORECASE)
            text = pattern.sub(self.alias_for(name), text)
        return text

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"mapping": self.mapping, "keep": sorted(self.keep), "next": self._next},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NameMap:
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            mapping=d.get("mapping", {}), keep=set(d.get("keep", [])), _next=d.get("next", 0)
        )
