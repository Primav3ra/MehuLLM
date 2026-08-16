"""Index WhatsApp exports into the memory store.

CPU-ONLY (fastembed is ONNX), so this can run while the GPU is busy with draft
generation. Fact extraction is the part that needs the LLM, and it queues.

Indexes:
  * style exemplars -- Mehul's own turns, for the voice model's few-shot baseline
  * session chunks  -- whole exchanges, for the brain's retrieval AND as the
                       unit of work for fact extraction

Group chats ARE indexed here, unlike in build_sft.py. They were excluded as SFT
targets because group voice differs, but they are perfectly good sources of
facts about his life.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mehullm.memory.embed import embed_passages, transliterate_hinglish
from mehullm.memory.store import MemoryStore
from mehullm.pipeline.pii import scrub
from mehullm.pipeline.sessionize import merge_bursts, split_sessions
from mehullm.pipeline.whatsapp_parser import parse_export

BATCH = 64
MAX_SESSION_CHARS = 2000


@dataclass
class IndexStats:
    chats: int = 0
    style_chunks: int = 0
    session_chunks: int = 0
    skipped_duplicates: int = 0
    jobs_queued: int = 0


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _iter_units(raw_dir: Path, self_aliases: set[str]) -> Iterator[tuple[str, dict]]:
    """Yield (text, metadata) for everything worth indexing."""
    folded = {a.casefold() for a in self_aliases}
    files = sorted({p.resolve() for p in raw_dir.rglob("*") if p.suffix.lower() == ".txt"})

    for f in files:
        chat = parse_export(f)
        turns = merge_bursts(chat.content_messages)
        sessions = split_sessions(turns)

        for si, session in enumerate(sessions):
            sid = f"{chat.chat_id}#{si}"

            # Style exemplars: only Mehul's own turns, and only substantial ones.
            for turn in session.turns:
                if turn.sender.casefold() in folded and 12 <= len(turn.text) <= 400:
                    yield scrub(turn.text), {
                        "kind": "style",
                        "chat_id": chat.chat_id,
                        "session_id": sid,
                        "speaker": "self",
                        "ts": int(turn.ts_start.timestamp()),
                    }

            # Session chunk: the retrievable unit, and the unit of extraction.
            body = "\n".join(f"{t.sender}: {t.text}" for t in session.turns)
            if len(body) < 40:
                continue
            yield scrub(body[:MAX_SESSION_CHARS]), {
                "kind": "doc",
                "chat_id": chat.chat_id,
                "session_id": sid,
                "speaker": "",
                "ts": int(session.turns[0].ts_start.timestamp()),
            }


def index(
    raw_dir: str | Path,
    db_path: str | Path,
    self_aliases: set[str],
    *,
    on_progress=None,
) -> IndexStats:
    store = MemoryStore(db_path)
    stats = IndexStats()
    buf_text: list[str] = []
    buf_meta: list[dict] = []
    session_keys: set[str] = set()

    def flush() -> None:
        if not buf_text:
            return
        vecs = embed_passages(buf_text)
        for text, meta, vec in zip(buf_text, buf_meta, vecs, strict=True):
            rid = store.add_chunk(
                text,
                vec,
                kind=meta["kind"],
                chat_id=meta["chat_id"],
                session_id=meta["session_id"],
                speaker=meta["speaker"],
                ts=meta["ts"],
                translit=transliterate_hinglish(text) if meta["kind"] == "style" else "",
                sha256=_sha(text),
            )
            if rid is None:
                stats.skipped_duplicates += 1
            elif meta["kind"] == "style":
                stats.style_chunks += 1
            else:
                stats.session_chunks += 1
                session_keys.add(meta["session_id"])
        store.commit()
        buf_text.clear()
        buf_meta.clear()
        if on_progress:
            on_progress(stats)

    seen_chats: set[str] = set()
    for text, meta in _iter_units(Path(raw_dir), self_aliases):
        seen_chats.add(meta["chat_id"])
        buf_text.append(text)
        buf_meta.append(meta)
        if len(buf_text) >= BATCH:
            flush()
    flush()

    stats.chats = len(seen_chats)
    stats.jobs_queued = store.queue_jobs(sorted(session_keys))
    return stats
