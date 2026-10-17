import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GenreSimilarity:
    def __init__(self, movies):
        self.movie_ids = movies.movie_id.astype(int).to_numpy()
        genres = movies.genres.fillna("(no genres listed)").str.get_dummies(sep="|")
        self.matrix = cosine_similarity(genres.to_numpy(dtype=np.float32)).astype(np.float32)
        self.index = {int(mid): i for i, mid in enumerate(self.movie_ids)}
    def __call__(self, a, b):
        ia, ib = self.index.get(int(a)), self.index.get(int(b))
        return float(self.matrix[ia, ib]) if ia is not None and ib is not None else 0.0

def ndcg_at_k(recommended, relevant, k):
    rel = set(relevant)
    gains = np.array([int(x in rel) for x in recommended[:k]], dtype=float)
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    ideal_n = min(k, len(rel))
    idcg = float(np.sum(1 / np.log2(np.arange(2, ideal_n + 2)))) if ideal_n else 0.0
    return dcg / idcg if idcg else 0.0

def ild_at_k(recommended, similarity):
    pairs = [similarity(recommended[i], recommended[j]) for i in range(len(recommended)) for j in range(i + 1, len(recommended))]
    return 1.0 - float(np.mean(pairs)) if pairs else 0.0
