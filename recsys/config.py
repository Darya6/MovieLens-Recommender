from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass(frozen=True)
class Config:
    seed: int = 42
    rating_threshold: float = 4.0
    min_ratings_per_user: int = 20
    k: int = 10
    n_neg: int = 200
    max_users_eval: Optional[int] = None

    als_factors: tuple = (32, 64, 128)
    als_regularization: tuple = (0.01, 0.05, 0.1)
    als_iterations: int = 30

    mmr_lambdas: tuple = tuple(round(i / 10, 1) for i in range(11))
    mmr_top_n: int = 200

    sasrec_epochs: int = 8
    max_sequence_length: int = 50

    @property
    def data_dir(self) -> Path:
        explicit = os.getenv("ML1M_DIR")
        if explicit:
            return Path(explicit).expanduser().resolve()

        candidates = [
            Path("/content"),
            Path("/content/ml-1m"),
            Path("data"),
            Path("."),
        ]

        for path in candidates:
            if (path / "ratings.dat").exists() or any(path.glob("*.zip")):
                return path.resolve()

        return Path("data").resolve()

    @property
    def output_dir(self) -> Path:
        return Path(
            os.getenv("RECSYS_OUTPUT_DIR", "results")
        ).resolve()

    @property
    def artifact_dir(self) -> Path:
        return Path(
            os.getenv("RECSYS_ARTIFACT_DIR", "artifacts")
        ).resolve()


CONFIG = Config()