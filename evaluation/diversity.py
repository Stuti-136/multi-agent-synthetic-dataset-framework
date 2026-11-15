"""Duplicate and near-duplicate detection (Member 1).

Contract (shared with the team):
    check_duplicate(example, accepted_list) -> (is_duplicate, similarity)

Two steps:
1. Exact duplicates: text is normalized (lowercase, punctuation removed, spaces collapsed,
   every digit replaced with #) so documents that differ only in numbers count as the same.
2. Near duplicates: embedding similarity (sentence-transformers) against accepted examples.
   similarity >= threshold (default 0.90, experimental) means near-duplicate.

If sentence-transformers is not installed (or its model cannot be downloaded) a simple
hashed bag-of-words embedder is used and a warning is printed. That fallback is for tests
and offline work only; the 0.90 threshold is meant for real sentence embeddings.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
import zlib
from typing import Any, Callable, Sequence

try:  # settings live in Member 3's config.py; fall back to defaults if absent
    import config as _cfg
except ImportError:  # pragma: no cover
    _cfg = None

SIMILARITY_THRESHOLD = getattr(_cfg, "SIMILARITY_THRESHOLD", 0.90)
EMBEDDING_MODEL = getattr(_cfg, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")

Embedder = Callable[[Sequence[str]], "list[list[float]]"]
_WARNED = False
_MODEL = None


def normalize_text(text: str) -> str:
    text = re.sub(r"\d", "#", text.lower())
    text = re.sub(r"[^\w\s#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_key(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def bow_embedder(texts: Sequence[str], dim: int = 512) -> list[list[float]]:
    """Deterministic hashed bag-of-words vectors (fallback only)."""
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        for token in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(token.encode("utf-8")) % dim] += 1.0
        vectors.append(vec)
    return vectors


def default_embedder(texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
    global _MODEL, _WARNED
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as err:  # not installed, or model download failed
            _MODEL = False
            if not _WARNED:
                print(f"[diversity] sentence-transformers unavailable ({err}); "
                      "using bag-of-words fallback", file=sys.stderr)
                _WARNED = True
    if _MODEL is False:
        return bow_embedder(texts)
    return [[float(x) for x in v] for v in _MODEL.encode(list(texts), normalize_embeddings=True)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class DuplicateChecker:
    """Caches embeddings so each document is embedded only once per run."""

    def __init__(self, threshold: float | None = None, embedder: Embedder | None = None):
        self.threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
        self.embedder = embedder or default_embedder
        self._cache: dict[str, list[float]] = {}

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vec in zip(missing, self.embedder(missing)):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]

    def find_closest(self, example: dict[str, Any],
                     accepted: Sequence[dict[str, Any]]) -> tuple[bool, float, str | None]:
        """Return (is_duplicate, highest_similarity, id_of_closest_example)."""
        text = example["source_document"]
        others = [a for a in accepted if a.get("id") != example.get("id")]
        if not others:
            return False, 0.0, None

        key = exact_key(text)
        for other in others:
            if exact_key(other["source_document"]) == key:
                return True, 1.0, other.get("id")

        vectors = self._vectors([text] + [o["source_document"] for o in others])
        sims = [cosine(vectors[0], v) for v in vectors[1:]]
        best = max(range(len(sims)), key=sims.__getitem__)
        return sims[best] >= self.threshold, sims[best], others[best].get("id")

    def check(self, example: dict[str, Any], accepted: Sequence[dict[str, Any]]) -> tuple[bool, float]:
        is_dup, sim, _ = self.find_closest(example, accepted)
        return is_dup, sim

    def duplicate_rate(self, examples: Sequence[dict[str, Any]]) -> float:
        """Percent of examples that are near-duplicates of an earlier example in the list."""
        if not examples:
            return 0.0
        dups = sum(1 for i, ex in enumerate(examples) if self.check(ex, examples[:i])[0])
        return 100.0 * dups / len(examples)


_default_checker: DuplicateChecker | None = None


def _checker() -> DuplicateChecker:
    global _default_checker
    if _default_checker is None:
        _default_checker = DuplicateChecker()
    return _default_checker


def check_duplicate(example: dict[str, Any], accepted_list: Sequence[dict[str, Any]]) -> tuple[bool, float]:
    """Shared contract: (is_duplicate, highest similarity to the accepted examples)."""
    return _checker().check(example, accepted_list)


def find_closest_duplicate(example: dict[str, Any],
                           accepted_list: Sequence[dict[str, Any]]) -> tuple[bool, float, str | None]:
    return _checker().find_closest(example, accepted_list)


def duplicate_rate(examples: Sequence[dict[str, Any]]) -> float:
    return _checker().duplicate_rate(examples)
