import numpy as np
import pandas as pd
from .config import CONFIG
from .metrics import ndcg_at_k, ild_at_k

class Evaluator:
    """Fixed sampled candidates, shared by all models for a given split."""
    def __init__(self, history, target, catalog, similarity, threshold=CONFIG.rating_threshold, n_neg=CONFIG.n_neg, max_users=CONFIG.max_users_eval, seed=CONFIG.seed):
        self.history = history.groupby("user_id").movie_id.apply(lambda s: set(map(int, s))).to_dict()
        self.positive = target[target.rating >= threshold].groupby("user_id").movie_id.apply(lambda s: set(map(int, s))).to_dict()
        users = sorted(set(self.history) & set(self.positive))
        if max_users is not None: users = users[:max_users]
        rng = np.random.default_rng(seed); catalog = set(map(int, catalog)); self.data = {}
        for u in users:
            pos = self.positive[u]; pool = list(catalog - self.history[u] - pos)
            if pos and len(pool) >= n_neg:
                neg = rng.choice(pool, size=n_neg, replace=False).tolist()
                self.data[int(u)] = (np.asarray(sorted(pos) + neg, dtype=np.int64), pos)
        self.similarity = similarity
    def evaluate(self, model, k=CONFIG.k, allowed=None):
        rows = []; total_pos = known_pos = 0
        for user, (candidates, positives0) in self.data.items():
            total_pos += len(positives0)
            positives = positives0 if allowed is None else positives0 & allowed
            known_pos += len(positives)
            if not positives: continue
            mask = np.ones(len(candidates), dtype=bool) if allowed is None else np.asarray([int(x) in allowed for x in candidates])
            items = candidates[mask]
            if not len(items): continue
            scores = np.nan_to_num(np.asarray(model.score(user, items), dtype=float), nan=-1e12, posinf=1e12, neginf=-1e12)
            rec = items[np.argsort(scores, kind="stable")[::-1][:k]].astype(int).tolist()
            rows.append({"user_id": user, "history_len": len(self.history[user]), f"recall@{k}": len(set(rec) & positives) / len(positives), f"ndcg@{k}": ndcg_at_k(rec, positives, k), f"precision@{k}": len(set(rec) & positives) / k, f"ild@{k}": ild_at_k(rec, self.similarity)})
        detail = pd.DataFrame(rows)
        if detail.empty: return {"n_users": 0, "oov_target_share": 1.0}, detail
        metrics = detail.drop(columns=["user_id", "history_len"]).mean().to_dict()
        metrics.update(n_users=len(detail), oov_target_share=1 - known_pos / max(total_pos, 1))
        return metrics, detail
    def evaluate_many(self, models, k=CONFIG.k):
        rows=[]; details={}
        for name, model in models.items():
            m, d = self.evaluate(model, k=k, allowed=getattr(model, "known_items", None)); m["model"] = name
            rows.append(m); details[name] = d
        return pd.DataFrame(rows).set_index("model").sort_values(f"ndcg@{k}", ascending=False), details
