"""Turn merging, sessionisation, and (context -> reply) pair construction.

Shared by the census (`stats.py`) and the SFT builder, so the number the
go/no-go gate reports is the *same* number the training set will actually have.
Two separate implementations would drift and the gate would lie.

The single most important step here is BURST MERGING. A WhatsApp reply is
usually 2-4 messages sent in quick succession:

    Mehul: haan yaar
    Mehul: kaam khatam nahi ho raha
    Mehul: bas thoda aur

Training on those as three independent targets teaches the model to emit
one-line fragments. Merged into one turn, it learns the actual rhythm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mehullm.pipeline.whatsapp_parser import Message

__all__ = ["Turn", "Session", "Pair", "merge_bursts", "split_sessions", "build_pairs"]

BURST_WINDOW_S = 180  # consecutive same-sender messages within 3 min = one turn
SESSION_GAP_S = 6 * 3600  # > 6 h apart starts a new conversation
CONTEXT_TURNS = 6  # how much history each training pair carries
CONTEXT_MAX_GAP_S = 1800  # the prompting turn must be within 30 min


@dataclass(slots=True)
class Turn:
    sender: str
    text: str
    ts_start: datetime
    ts_end: datetime
    n_messages: int = 1

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class Session:
    turns: list[Turn] = field(default_factory=list)

    @property
    def span_s(self) -> float:
        if not self.turns:
            return 0.0
        return (self.turns[-1].ts_end - self.turns[0].ts_start).total_seconds()


@dataclass(slots=True)
class Pair:
    context: list[Turn]
    target: Turn
    chat_id: str


def merge_bursts(messages: list[Message], window_s: int = BURST_WINDOW_S) -> list[Turn]:
    """Collapse consecutive same-sender messages into single turns."""
    turns: list[Turn] = []
    for m in messages:
        if not m.is_content or not m.sender:
            continue
        if (
            turns
            and turns[-1].sender == m.sender
            and (m.ts - turns[-1].ts_end).total_seconds() <= window_s
        ):
            t = turns[-1]
            t.text += "\n" + m.text
            t.ts_end = m.ts
            t.n_messages += 1
        else:
            turns.append(Turn(sender=m.sender, text=m.text, ts_start=m.ts, ts_end=m.ts))
    return turns


def split_sessions(turns: list[Turn], gap_s: int = SESSION_GAP_S) -> list[Session]:
    """Split a turn stream into conversations on long silences."""
    sessions: list[Session] = []
    current = Session()
    for t in turns:
        if current.turns and (t.ts_start - current.turns[-1].ts_end).total_seconds() > gap_s:
            sessions.append(current)
            current = Session()
        current.turns.append(t)
    if current.turns:
        sessions.append(current)
    return sessions


def build_pairs(
    sessions: list[Session],
    self_aliases: set[str],
    chat_id: str = "",
    context_turns: int = CONTEXT_TURNS,
    max_gap_s: int = CONTEXT_MAX_GAP_S,
) -> list[Pair]:
    """Emit one (context -> reply) pair per turn authored by ``self_aliases``.

    A target is only usable if somebody *else* spoke recently. Without that
    guard you collect unprompted monologues -- "bhai sun", "hello", "?" sent
    into a dead chat -- which teach the model to open conversations rather
    than to reply to them.
    """
    folded = {a.casefold() for a in self_aliases}
    pairs: list[Pair] = []

    for session in sessions:
        for i, turn in enumerate(session.turns):
            if turn.sender.casefold() not in folded:
                continue
            context = session.turns[max(0, i - context_turns) : i]
            if not context:
                continue
            recent_other = [
                c
                for c in context
                if c.sender.casefold() not in folded
                and (turn.ts_start - c.ts_end).total_seconds() <= max_gap_s
            ]
            if not recent_other:
                continue
            pairs.append(Pair(context=context, target=turn, chat_id=chat_id))
    return pairs
