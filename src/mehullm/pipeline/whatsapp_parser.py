"""Parse WhatsApp `Export chat` text files into structured messages.

HEADER_RE decides "new message?" (a line failing it is a continuation of the
previous one); BODY_RE splits sender from text (a line failing it is a system
event).

Normalises as little as possible -- the corpus exists to capture how one person
writes. NFC only, never NFKC/NFKD (which decompose Devanagari matras). ZWJ/ZWNJ
are kept: load-bearing in conjuncts and emoji.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import regex

__all__ = ["Message", "Chat", "parse_export", "parse_text"]

# Character-level normalisation

# Control marks WhatsApp injects into exports. These are exporter artifacts,
# not authored content, and they silently break naive regexes.
_INVISIBLE = dict.fromkeys(
    map(
        ord,
        "‎"  # LEFT-TO-RIGHT MARK      (very common, prefixes iOS media lines)
        "‏"  # RIGHT-TO-LEFT MARK
        "‪"  # LEFT-TO-RIGHT EMBEDDING
        "‫"  # RIGHT-TO-LEFT EMBEDDING
        "‬"  # POP DIRECTIONAL FORMATTING
        "‭"  # LEFT-TO-RIGHT OVERRIDE
        "‮"  # RIGHT-TO-LEFT OVERRIDE
        "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
    ),
    None,
)

# NBSP variants -> plain space. iOS uses U+202F (NARROW NO-BREAK SPACE) between
# the time and AM/PM, which is the single most common cause of "my regex works
# on Android exports but not iOS".
_NBSP = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " "})

# NOTE: we deliberately do NOT strip ZWJ (U+200D) or ZWNJ (U+200C) -- they are
# load-bearing inside Devanagari conjuncts and inside emoji sequences
# (e.g. family emoji, profession emoji).


def normalize_line(line: str) -> str:
    """Strip exporter artifacts. Preserves all authored content verbatim."""
    return unicodedata.normalize("NFC", line.translate(_INVISIBLE).translate(_NBSP))


# Stage 1: message header

HEADER_RE = regex.compile(
    r"""
    ^\s*
    (?:\[\s*)?                                          # iOS opens with '['
    (?P<f1>\d{1,4})[/\-.](?P<f2>\d{1,2})[/\-.](?P<f3>\d{2,4})
    ,?\s+
    (?P<hh>\d{1,2}):(?P<mi>\d{2})(?::(?P<ss>\d{2}))?
    (?:\s*(?P<ampm>[AaPp]\.?\s?[Mm]\.?))?
    \s*
    (?:\]\s*|[-–—]\s+)                        # ']' on iOS, ' - ' on Android
    (?P<rest>.*)$
    """,
    regex.VERBOSE | regex.DOTALL,
)

# Stage 2: sender / text
# '~' prefixes a non-contact participant's push name in group exports.
# Sender is non-greedy and colon-free so "me: check this: link" splits at the
# FIRST colon, giving sender="me", text="check this: link".
BODY_RE = regex.compile(
    r"^~?\s*(?P<sender>[^:\n]{1,80}?):(?:[ ](?P<text>.*))?$",
    regex.DOTALL,
)

# Checked BEFORE BODY_RE: system notices containing a colon would otherwise be
# mis-split into a bogus sender.
#
# Every alternation is anchored and the actor prefix bounded by `[^:\n]{0,60}`.
# Unbounded `.*\ added\b` branches backtracked quadratically -- minutes instead
# of seconds on 25 MB. The colon-free class also stops "Mehul: I left home"
# reaching the `left` verb, so it is faster and more correct at once.
_SYSTEM_RE = regex.compile(
    r"""
    ^(?:
      Messages\ and\ calls\ are\ end-to-end\ encrypted
    | Your\ security\ code\ with\
    | Live\ location\ shared
    | You\ (?:blocked|unblocked)\ this\ contact
    | This\ (?:business|chat)\ (?:uses|is)\
    | Disappearing\ messages\
    | You're\ now\ an\ admin
    | Tap\ to\ (?:learn\ more|change)
    | [^:\n]{0,60}?\ (?:
          changed\ (?:the\ subject|their\ phone\ number|to\ |the\ group)
        | created\ (?:group|this\ group)
        | added\
        | removed\
        | left$
        | joined\ using
        | turned\ (?:on|off)\ disappearing
        | pinned\ a\ message
        | used\ this\ group's\ invite\ link
        | was\ added
        | is\ an\ admin
      )
    )
    """,
    regex.VERBOSE | regex.IGNORECASE,
)

# Media placeholders. Counted for statistics, excluded from training text.
# Bounded quantifiers throughout -- see the performance note on _SYSTEM_RE.
_MEDIA_RE = regex.compile(
    r"""^\s*(?:
      <\ ?Media\ omitted\ ?>
    | (?:image|video|audio|sticker|GIF|document|Contact\ card|photo)\ omitted
    | <attached:[^\n]{0,300}>
    | [^\n]{1,200}\.(?:jpg|jpeg|png|webp|mp4|opus|m4a|pdf|docx?)\s*\(file\ attached\)
    )\s*$""",
    regex.VERBOSE | regex.IGNORECASE,
)

# Tombstones: the message existed but its content is gone. Not style data.
_TOMBSTONE_RE = regex.compile(
    r"^\s*(?:This\ message\ was\ deleted|You\ deleted\ this\ message|null|"
    r"Waiting\ for\ this\ message|This\ message\ (?:is|was)\ (?:not\ )?supported)\s*$",
    regex.VERBOSE | regex.IGNORECASE,
)

# STRIPPED, not dropped -- the content is still authored by the sender.
_EDITED_SUFFIX_RE = regex.compile(r"\s*<\s*This message was edited\s*>\s*$", regex.IGNORECASE)

_POLL_RE = regex.compile(r"^\s*POLL:\s", regex.IGNORECASE)

# Call notices carry a SENDER ("Rohan: Missed voice call"), so they arrive via
# BODY_RE as ordinary messages and must be classified at the message level --
# not in _SYSTEM_RE, which only sees senderless lines.
_CALL_NOTICE_RE = regex.compile(
    r"^\s*(?:Missed\ (?:voice|video|group)\ call|Video\ call|Voice\ call|"
    r"(?:Call|Video)\ (?:ended|declined|unanswered)|No\ answer|Tap\ to\ call\ back)\s*$",
    regex.VERBOSE | regex.IGNORECASE,
)

# Devanagari + other Indic scripts, used for the script-mix statistic that
# drives the embedding-model and tokenizer decisions.
_DEVANAGARI_RE = regex.compile(r"\p{Devanagari}")


# Data model


@dataclass(slots=True)
class Message:
    ts: datetime
    sender: str | None  # None for system events
    text: str
    line_no: int
    is_system: bool = False
    is_media: bool = False
    is_tombstone: bool = False
    was_edited: bool = False

    @property
    def is_content(self) -> bool:
        """True only for real, usable authored text."""
        return not (self.is_system or self.is_media or self.is_tombstone) and bool(self.text.strip())

    @property
    def has_devanagari(self) -> bool:
        return bool(_DEVANAGARI_RE.search(self.text))


@dataclass(slots=True)
class Chat:
    chat_id: str
    path: Path
    messages: list[Message] = field(default_factory=list)
    date_order: str = "unknown"  # 'day_first' | 'month_first' | 'year_first'
    date_order_evidence: str = ""
    unparsed_lines: int = 0

    @property
    def participants(self) -> Counter[str]:
        return Counter(m.sender for m in self.messages if m.sender)

    @property
    def content_messages(self) -> list[Message]:
        return [m for m in self.messages if m.is_content]


# Date-order resolution -- decided ONCE PER FILE, never per line


def _resolve_date_order(pairs: list[tuple[int, int, int]]) -> tuple[str, str]:
    """Decide whether the file is DD/MM, MM/DD, or YYYY-MM-DD.

    Per-line guessing produces silent, non-obvious corruption (03/04 and 04/03
    both parse), so this looks at the whole file and commits to one reading.
    """
    if not pairs:
        return "day_first", "no dates found; defaulted to day-first"

    if any(f1 > 31 for f1, _, _ in pairs):
        return "year_first", "field 1 exceeds 31 -> ISO-style year-first"

    # Count evidence rather than trusting the FIRST hit. A single malformed
    # line -- a pasted date, a "13/14 done" in a message body that happens to
    # satisfy the header shape -- would otherwise flip every timestamp in a
    # 180k-message file. Observed in the real corpus: one such line against
    # 14,185 contrary ones.
    n1 = sum(1 for f1, _, _ in pairs if f1 > 12)
    n2 = sum(1 for _, f2, _ in pairs if f2 > 12)

    if n1 and n2:
        # Contradictory. Trust a decisive majority; otherwise fall through to
        # the chronology check below, which is evidence-based rather than a coin flip.
        if n1 >= 10 * n2:
            return "day_first", f"day-first dominates ({n1:,} vs {n2:,} contradictory)"
        if n2 >= 10 * n1:
            return "month_first", f"month-first dominates ({n2:,} vs {n1:,} contradictory)"
    elif n1:
        return "day_first", f"field 1 exceeds 12 in {n1:,} dates -> must be the day"
    elif n2:
        return "month_first", f"field 2 exceeds 12 in {n2:,} dates -> must be the day"

    # Fully ambiguous (every date is <= 12/12). Fall back to which reading
    # yields a non-decreasing timestamp sequence -- exports are chronological.
    def monotonic(day_first: bool) -> int:
        seq, bad = [], 0
        for f1, f2, f3 in pairs:
            d, m = (f1, f2) if day_first else (f2, f1)
            try:
                seq.append(datetime(_full_year(f3), m, d))
            except ValueError:
                bad += 1
        bad += sum(1 for a, b in zip(seq, seq[1:], strict=False) if b < a)
        return bad

    df, mf = monotonic(True), monotonic(False)
    if df < mf:
        return "day_first", f"day-first gave fewer ordering violations ({df} vs {mf})"
    if mf < df:
        return "month_first", f"month-first gave fewer ordering violations ({mf} vs {df})"
    return "day_first", "fully ambiguous; defaulted to day-first (India)"


def _full_year(y: int) -> int:
    return y if y >= 1000 else 2000 + y


# Parsing


def parse_text(raw: str, chat_id: str = "chat", path: Path | None = None) -> Chat:
    lines = raw.splitlines()

    date_fields: list[tuple[int, int, int]] = []
    for line in lines:
        m = HEADER_RE.match(normalize_line(line))
        if m:
            date_fields.append((int(m["f1"]), int(m["f2"]), int(m["f3"])))
    order, evidence = _resolve_date_order(date_fields)

    chat = Chat(
        chat_id=chat_id,
        path=path or Path(chat_id),
        date_order=order,
        date_order_evidence=evidence,
    )

    current: Message | None = None
    for i, raw_line in enumerate(lines, start=1):
        line = normalize_line(raw_line).rstrip()
        if not line and current is None:
            continue

        m = HEADER_RE.match(line)
        if not m:
            # Continuation of a multi-line message. Preserve the newline: line
            # breaks inside a message are part of how someone writes.
            if current is not None:
                current.text += "\n" + line
            else:
                chat.unparsed_lines += 1
            continue

        if current is not None:
            chat.messages.append(_finalize(current))

        current = _build_message(m, order, i)

    if current is not None:
        chat.messages.append(_finalize(current))

    return chat


def _build_message(m: regex.Match, order: str, line_no: int) -> Message:
    f1, f2, f3 = int(m["f1"]), int(m["f2"]), int(m["f3"])
    if order == "year_first":
        year, month, day = _full_year(f1), f2, f3
    elif order == "month_first":
        year, month, day = _full_year(f3), f1, f2
    else:
        year, month, day = _full_year(f3), f2, f1

    hour, minute = int(m["hh"]), int(m["mi"])
    second = int(m["ss"]) if m["ss"] else 0
    if ampm := m["ampm"]:
        pm = ampm.strip().lower().replace(".", "").replace(" ", "").startswith("p")
        hour = (hour % 12) + (12 if pm else 0)

    try:
        ts = datetime(year, month, day, hour, minute, second)
    except ValueError:
        # Impossible date (e.g. 31 Feb from a wrong order guess). Keep the
        # message rather than silently dropping authored text.
        ts = datetime(year, 1, 1)

    rest = m["rest"]

    # System check must run BEFORE BODY_RE -- several notices contain a colon.
    if _SYSTEM_RE.search(rest):
        return Message(ts=ts, sender=None, text=rest, line_no=line_no, is_system=True)

    body = BODY_RE.match(rest)
    if not body:
        return Message(ts=ts, sender=None, text=rest, line_no=line_no, is_system=True)

    return Message(
        ts=ts,
        sender=body["sender"].strip(),
        text=body["text"] or "",
        line_no=line_no,
    )


def _finalize(msg: Message) -> Message:
    """Classify a completed message once its continuation lines are attached."""
    if msg.is_system:
        return msg

    text, n = _EDITED_SUFFIX_RE.subn("", msg.text)
    if n:
        msg.was_edited = True
    msg.text = text

    if _MEDIA_RE.match(msg.text):
        msg.is_media = True
    elif _TOMBSTONE_RE.match(msg.text):
        msg.is_tombstone = True
    elif _POLL_RE.match(msg.text) or _CALL_NOTICE_RE.match(msg.text):
        msg.is_system = True

    return msg


def parse_export(path: str | Path, chat_id: str | None = None) -> Chat:
    """Parse one exported chat file.

    Reads as UTF-8 with BOM tolerance; WhatsApp writes UTF-8 but Windows tools
    frequently re-save with a BOM.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8-sig", errors="replace")
    return parse_text(raw, chat_id=chat_id or p.stem, path=p)
