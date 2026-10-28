"""Retriever over the evidence cards: filter by verdict and heatmap regions, rank, then diversify.

1. Filter: FAKE -> artifact cards that share a region with the heatmap's top-3; REAL -> authentic cards that do.
2. Rank with up to three rankers, fused by reciprocal rank fusion (RRF), so no weights need tuning:
     region  overlap between the card's regions and the top-3 regions, weighted by rank (top region counts most)
     bm25    keyword match between the query and the card
     dense   cosine similarity of embeddings (BAAI/bge-small-en-v1.5, run locally with transformers)
3. Diversify: FAKE -> the best card of each of up to 3 artifact types, plus the scope cards of those types;
   REAL -> the best 2 authentic cards. Every result ends with one limitation card.

The demo (prominent=True) uses the prominent regions (Explanation.prominent_regions) instead of the top 3 and
adds, for any prominent region no picked card lists, the best card that does, so every region it explains has
evidence. The query is filled from the evidence by a fixed template (no LLM), so retrieval is deterministic.
"""
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.explain.corpus import load_corpus

RANKERS = ("region", "bm25", "dense")
CENTRAL = {"left eye", "right eye", "nose", "mouth / teeth", "teeth", "mouth"}
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's instruction for short queries
RRF_K = 60


def tokens(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def card_text(d) -> str:
    """What the rankers see for a card: its label, kind, regions, pattern and text."""
    about = d.artifact_type.replace("_", " ") if d.artifact_type else "limitation of the evidence"
    return f"{about}. {d.kind}. Regions: {', '.join(d.regions)}. {d.pattern} pattern. {d.text}"


def build_query(exp, k: int = 3, regions: list = None):
    """(query, regions, pattern) for an Explanation: fixed template, no model involved.
    regions: the regions to search for (default: the top k)."""
    regions = regions or [r for r, _ in exp.top_regions[:k]]
    pattern = "central" if regions and regions[0] in CENTRAL else "outer"
    verdict = "fake, manipulated" if exp.label == "fake" else "real, authentic"
    query = (f"{verdict} face; detector evidence concentrated in {', '.join(regions)}; {pattern} pattern; "
             f"{exp.confidence} confidence")
    return query, regions, pattern


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.tf = [Counter(d) for d in docs]
        self.length = np.array([len(d) for d in docs], dtype=float)
        df = Counter(w for d in docs for w in set(d))
        self.idf = {w: math.log(1 + (len(docs) - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: list) -> np.ndarray:
        norm = self.k1 * (1 - self.b + self.b * self.length / self.length.mean())
        out = np.zeros(len(self.tf))
        for w in set(query):
            if w in self.idf:
                f = np.array([tf.get(w, 0) for tf in self.tf], dtype=float)
                out += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return out


class DenseEncoder:
    """Sentence embeddings with a bge model: CLS pooling, L2-normalised."""

    def __init__(self, model: str = DENSE_MODEL, device: str = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.name = torch, model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()

    def encode(self, texts: list, query: bool = False) -> np.ndarray:
        texts = [QUERY_PREFIX + t for t in texts] if query else list(texts)
        with self.torch.no_grad():
            enc = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            emb = self.model(**enc).last_hidden_state[:, 0]
            emb = self.torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy()


def _ranks(scores: dict) -> dict:
    """Competition ranks (0 = best); tied scores share a rank."""
    values = sorted(scores.values(), reverse=True)
    return {i: values.index(s) for i, s in scores.items()}


class Retriever:
    def __init__(self, rankers=RANKERS, n_types: int = 3, n_authentic: int = 2, filter_regions: bool = True,
                 diversify: bool = True, encoder: DenseEncoder = None, atomic: bool = False, prominent: bool = False):
        """rankers: any of "region", "bm25", "dense". filter_regions/diversify=False give a plain top-n_types
        retriever (still verdict-aware) for comparison. `encoder` lets several retrievers share one model.
        atomic: match cards by atomic regions (the demo; needs Explanations from Explainer(atomic=True)).
        prominent: search for the prominent regions, with a card for each (the demo), instead of the top 3."""
        unknown = set(rankers) - set(RANKERS)
        if unknown or not rankers:
            raise ValueError(f"rankers must be a non-empty subset of {RANKERS}, got {rankers}")
        self.docs = load_corpus(atomic)
        self.rankers = tuple(r for r in RANKERS if r in rankers)
        self.n_types, self.n_authentic = n_types, n_authentic
        self.filter_regions, self.diversify, self.prominent = filter_regions, diversify, prominent
        self.bm25 = BM25([tokens(card_text(d)) for d in self.docs])
        self.encoder = self.card_emb = None
        if "dense" in self.rankers:
            self.encoder = encoder or DenseEncoder()
            self.card_emb = self.encoder.encode([card_text(d) for d in self.docs])

    def describe(self) -> str:
        extras = [] if self.filter_regions and self.diversify else ["plain top-k"]
        return "+".join(self.rankers) + (f" ({', '.join(extras)})" if extras else "")

    def _rank(self, pool: list, query: str, regions: list) -> list:
        """Card indices in `pool`, best first: reciprocal rank fusion of the rankers, ties broken by card id."""
        rankings = []
        if "region" in self.rankers:
            rankings.append(_ranks({i: sum(1 / (r + 1) for r, name in enumerate(regions) if name in self.docs[i].regions)
                                    for i in pool}))
        if "bm25" in self.rankers:
            s = self.bm25.scores(tokens(query))
            rankings.append(_ranks({i: float(s[i]) for i in pool}))
        if "dense" in self.rankers:
            s = self.card_emb @ self.encoder.encode([query], query=True)[0]
            rankings.append(_ranks({i: float(s[i]) for i in pool}))
        fused = {i: sum(1 / (RRF_K + rk[i] + 1) for rk in rankings) for i in pool}
        return sorted(pool, key=lambda i: (-fused[i], self.docs[i].id))

    def _pool(self, kind: str, regions: list) -> list:
        pool = [i for i, d in enumerate(self.docs) if d.kind == kind]
        if self.filter_regions:
            pool = [i for i in pool if set(self.docs[i].regions) & set(regions)] or pool
        return pool

    def _cover(self, picked: list, ranked: list, regions: list) -> list:
        """`picked` plus, for every region no picked card lists, the best-ranked card that lists it (if any)."""
        for region in regions:
            if not any(region in self.docs[i].regions for i in picked):
                best = next((i for i in ranked if region in self.docs[i].regions), None)
                if best is not None:
                    picked.append(best)
        return picked

    def retrieve(self, exp, k_regions: int = 3) -> list:
        """Cards for one Explanation (verdict, top or prominent regions, confidence), in prompt order."""
        query, regions, _ = build_query(exp, k_regions, exp.prominent_regions() if self.prominent else None)
        if exp.label == "fake":
            ranked = self._rank(self._pool("artifact", regions), query, regions)
            if self.diversify:
                picked, types = [], []
                for i in ranked:
                    if self.docs[i].artifact_type not in types and len(types) < self.n_types:
                        types.append(self.docs[i].artifact_type)
                        picked.append(i)
                if self.prominent:
                    picked = self._cover(picked, ranked, regions)
                    types = list(dict.fromkeys(self.docs[i].artifact_type for i in picked))
                picked += sorted((i for i, d in enumerate(self.docs) if d.kind == "scope" and d.artifact_type in types),
                                 key=lambda i: types.index(self.docs[i].artifact_type))
            else:
                picked = ranked[:self.n_types]
        else:
            ranked = self._rank(self._pool("authentic", regions), query, regions)
            picked = ranked[:self.n_authentic]
            if self.prominent:
                picked = self._cover(picked, ranked, regions)
        limits = [i for i, d in enumerate(self.docs) if d.kind == "limitation"
                  and (not d.confidence or exp.confidence in d.confidence)]
        specific = [i for i in limits if exp.confidence in self.docs[i].confidence]
        picked.append(self._rank(specific or limits, query, regions)[0])
        return [self.docs[i] for i in picked]
