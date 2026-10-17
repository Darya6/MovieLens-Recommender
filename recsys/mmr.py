import numpy as np


def mmr_rerank(items, relevance, k, lambda_, similarity):
    if not len(items): return []
    relevance = np.asarray(relevance, dtype=np.float32)
    normalized = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-9)
    rel = {int(item): float(score) for item, score in zip(items, normalized)}
    selected, remaining = [], list(map(int, items))
    while remaining and len(selected) < k:
        best = max(remaining, key=lambda x: lambda_ * rel[x] - (1 - lambda_) * max((similarity(x, s) for s in selected), default=0.0))
        selected.append(best); remaining.remove(best)
    return selected

class MMRRecommender:
    def __init__(self, base, lambda_, similarity, k=10, top_n=200):
        self.base, self.lambda_, self.similarity, self.k, self.top_n = base, float(lambda_), similarity, k, top_n
        if not 0 <= self.lambda_ <= 1: raise ValueError("lambda должен быть в [0, 1]")
    def score(self, user_id, item_ids):
        raw = np.asarray(self.base.score(user_id, item_ids), dtype=float)
        shortlist_idx = np.argsort(raw)[::-1][:min(self.top_n, len(item_ids))]
        ordered = mmr_rerank(np.asarray(item_ids)[shortlist_idx], raw[shortlist_idx], self.k, self.lambda_, self.similarity)
        ranks = {item: self.k - rank for rank, item in enumerate(ordered)}
        return np.asarray([ranks.get(int(item), -1e9) for item in item_ids], dtype=np.float32)
