"""Local embeddings via fastembed (ONNX, CPU, NO PyTorch).

The no-torch part is not a style choice. Installing torch on a box with ~0.4 GB
free RAM to compute 384-dim vectors would cost ~2.5 GB of disk and hundreds of
MB resident, for a job that ONNX does fine on the CPU.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384 dims, 0.22 GB, 50+ languages including Hindi).

NOT the originally-planned intfloat/multilingual-e5-small: fastembed does not
ship it. The only e5 multilingual it has is `-large` at 2.24 GB / 1024 dims,
which is far too heavy for a box with ~0.4 GB free RAM. MiniLM keeps the same
384 dims, so the sqlite-vec schema is unchanged.

TWO THINGS THAT SILENTLY COST RECALL IF YOU GET THEM WRONG:

1. PREFIXES ARE MODEL-SPECIFIC. e5 models require "query: " / "passage: ".
   MiniLM does NOT -- feeding it those prefixes embeds the literal word
   "passage" into every vector and makes documents look more like each other
   than like the query. `_PREFIXES` is therefore gated on the model name, so
   swapping back to an e5 model later re-enables them automatically.

2. Neither model was trained on romanized Hindi, which is ~22% of this corpus.
   So each chunk is ALSO stored transliterated into Devanagari in a second FTS5
   column and queries match against both. Lexical carries Hinglish recall that
   the dense side cannot.
"""

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
    """Latin -> Devanagari, for the second lexical index.

    Optional: if indic-transliteration is not installed we degrade to
    lexical-on-Latin only rather than failing the whole pipeline.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        def fn(s: str) -> str:
            try:
                return transliterate(s, sanscript.ITRANS, sanscript.DEVANAGARI)
            except Exception:  # noqa: BLE001
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


def embed_queries(texts: list[str]) -> list[list[float]]:
    return [list(v) for v in _model().embed([_QUERY_PREFIX + t for t in texts])]
