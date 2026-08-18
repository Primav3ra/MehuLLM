"""Local embeddings via fastembed (ONNX, CPU, NO PyTorch)."""

from __future__ import annotations

from functools import lru_cache

DIM = 384
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# e5 is asymmetric and needs prefixes; MiniLM/BGE-style models do not.
_USES_PREFIXES = "e5" in MODEL.lower()
_QUERY_PREFIX = "query: " if _USES_PREFIXES else ""
_PASSAGE_PREFIX = "passage: " if _USES_PREFIXES else ""


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL)


@lru_cache(maxsize=1)
def _transliterator():
    """Latin -> Devanagari, for the second lexical index."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        def fn(s: str) -> str:
            try:
                return transliterate(s, sanscript.ITRANS, sanscript.DEVANAGARI)
            except Exception:
                return ""

        return fn
    except ImportError:
        return lambda _s: ""


def transliterate_hinglish(text: str) -> str:
    return _transliterator()(text)


def embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [list(v) for v in _model().embed([_PASSAGE_PREFIX + t for t in texts])]


def embed_query(text: str) -> list[float]:
    return list(next(iter(_model().embed([_QUERY_PREFIX + text]))))
