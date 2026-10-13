"""TF-IDF retriever over the forensic corpus, re-ranked by overlap with the Grad-CAM++ regions.

A sparse retriever is enough for a ~10-document corpus, is deterministic and needs no model download.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.explain.corpus import load_corpus


class Retriever:
    def __init__(self, region_weight: float = 0.5):
        self.docs = load_corpus()
        self.region_weight = region_weight
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(
            [f"{d.artifact_type.replace('_', ' ')} {' '.join(d.regions)} {d.text}" for d in self.docs])

    def search(self, query: str, regions=(), verdict: str = "fake", k: int = 3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        scored = []
        for doc, sim in zip(self.docs, sims):
            # Region prior: weight overlap by rank of the region in the heatmap (top region counts most)
            overlap = sum(1.0 / (rank + 1) for rank, r in enumerate(regions) if r in doc.regions)
            score = sim + self.region_weight * overlap
            if verdict == "real" and doc.artifact_type == "none":
                score += 1.0
            scored.append((score, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [doc for _, doc in scored[:k]]
