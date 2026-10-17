import numpy as np
import scipy.sparse as sp
from .config import CONFIG

class RandomRecommender:
    def __init__(self, seed=CONFIG.seed): self.seed = seed
    def fit(self, data): return self
    def score(self, user_id, item_ids): return np.random.default_rng(self.seed + int(user_id)).random(len(item_ids))

class PopularityRecommender:
    def __init__(self, threshold=CONFIG.rating_threshold): self.threshold = threshold
    def fit(self, data):
        self.popularity = data[data.rating >= self.threshold].groupby("movie_id").size().to_dict(); return self
    def score(self, user_id, item_ids): return np.asarray([self.popularity.get(int(i), 0) for i in item_ids], dtype=float)

class SurpriseRecommender:
    def __init__(self, kind, seed=CONFIG.seed): self.kind, self.seed = kind, seed
    def fit(self, data):
        try:
            from surprise import Dataset, Reader, KNNBasic, SVD, SVDpp
        except ImportError as e: raise RuntimeError("Установите scikit-surprise (см. requirements.txt)") from e
        x = data[["user_id", "movie_id", "rating"]].copy()
        x["user_id"] = x.user_id.astype(str); x["movie_id"] = x.movie_id.astype(str)
        train = Dataset.load_from_df(x, Reader(rating_scale=(1, 5))).build_full_trainset()
        if self.kind == "ItemKNN": self.model = KNNBasic(k=40, sim_options={"name":"cosine", "user_based":False}, verbose=False)
        elif self.kind == "SVD": self.model = SVD(n_factors=100, n_epochs=20, random_state=self.seed, verbose=False)
        elif self.kind == "SVD++": self.model = SVDpp(n_factors=50, n_epochs=15, random_state=self.seed, verbose=False)
        else: raise ValueError(self.kind)
        self.model.fit(train); return self
    def score(self, user_id, item_ids): return np.asarray([self.model.predict(str(int(user_id)), str(int(i))).est for i in item_ids])

class ALSRecommender:
    def __init__(self, factors=64, regularization=.05, iterations=CONFIG.als_iterations, threshold=CONFIG.rating_threshold, seed=CONFIG.seed):
        self.factors, self.regularization, self.iterations, self.threshold, self.seed = factors, regularization, iterations, threshold, seed
    def fit(self, data):
        try: from implicit.als import AlternatingLeastSquares
        except ImportError as e: raise RuntimeError("Установите implicit (см. requirements.txt)") from e
        d = data[data.rating >= self.threshold]
        users, items = sorted(d.user_id.unique()), sorted(d.movie_id.unique())
        self.user_index = {int(x): i for i, x in enumerate(users)}; self.item_index = {int(x): i for i, x in enumerate(items)}
        rows = d.user_id.map(self.user_index).to_numpy(); cols = d.movie_id.map(self.item_index).to_numpy()
        matrix = sp.csr_matrix((np.ones(len(d), dtype=np.float32), (rows, cols)), shape=(len(users), len(items)))
        self.model = AlternatingLeastSquares(factors=self.factors, regularization=self.regularization, iterations=self.iterations, random_state=self.seed)
        self.model.fit(matrix, show_progress=False)
        self.user_factors, self.item_factors = self.model.user_factors, self.model.item_factors
        self.known_items = set(self.item_index)
        return self
    def score(self, user_id, item_ids):
        out = np.zeros(len(item_ids), dtype=np.float32); u = self.user_index.get(int(user_id))
        if u is None: return out
        for j, item in enumerate(item_ids):
            i = self.item_index.get(int(item))
            if i is not None: out[j] = self.user_factors[u] @ self.item_factors[i]
        return out
