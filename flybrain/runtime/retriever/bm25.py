"""Pure-numpy BM25 ranker over a small in-memory corpus.

Tokenisation is the dumbest thing that works — lowercase + split on
non-alphanumeric. This is good enough for routing-level decisions
("does this snippet match the query?"); a real retriever (Yandex
embeddings + hybrid BM25) lands in Phase 4.

Reference: K. Sparck Jones et al., "A probabilistic model of information
retrieval", Information Processing & Management, 2000.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04ff0-9_]+")  # Latin + Cyrillic block + digits


def _tokenise(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


@dataclass(slots=True)
class BM25Retriever:
    """Plain BM25Okapi (k1=1.5, b=0.75)."""

    corpus: list[str] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75

    _tokens: list[list[str]] = field(default_factory=list, init=False, repr=False)
    _doc_lens: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
        init=False,
        repr=False,
    )
    _avg_dl: float = field(default=1.0, init=False, repr=False)
    _doc_freqs: list[Counter] = field(default_factory=list, init=False, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.corpus:
            self._reindex()

    def add(self, doc: str) -> None:
        self.corpus.append(doc)
        self._reindex()

    def _reindex(self) -> None:
        self._tokens = [_tokenise(d) for d in self.corpus]
        self._doc_lens = np.asarray([len(t) for t in self._tokens], dtype=np.float32)
        self._avg_dl = float(self._doc_lens.mean()) if len(self._doc_lens) else 1.0
        self._doc_freqs = [Counter(t) for t in self._tokens]
        df: Counter = Counter()
        for tokens in self._tokens:
            for term in set(tokens):
                df[term] += 1
        n = len(self._tokens)
        # BM25Okapi idf with the customary +1 smoothing
        self._idf = {
            term: float(np.log((n - count + 0.5) / (count + 0.5) + 1.0))
            for term, count in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        if not self.corpus:
            return 0.0
        terms = _tokenise(query)
        freqs = self._doc_freqs[doc_idx]
        dl = float(self._doc_lens[doc_idx])
        denom_dl_factor = self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1e-6))
        s = 0.0
        for t in terms:
            if t not in self._idf or t not in freqs:
                continue
            tf = float(freqs[t])
            idf = self._idf[t]
            s += idf * (tf * (self.k1 + 1)) / (tf + denom_dl_factor + 1e-9)
        return s

    def top_k(self, query: str, k: int = 3) -> list[tuple[float, int, str]]:
        if not self.corpus:
            return []
        scores = [(self.score(query, i), i, self.corpus[i]) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]
