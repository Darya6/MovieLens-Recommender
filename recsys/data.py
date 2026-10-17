from pathlib import Path
import zipfile
import pandas as pd
from .config import CONFIG, Config

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]

def _read_dat(source, names):
    return pd.read_csv(source, sep="::", engine="python", header=None, names=names, encoding="latin-1")

def load_movielens(data_dir: Path | None = None):
    """Load MovieLens-1M from an extracted directory or a ZIP containing *.dat files."""
    root = Path(data_dir or CONFIG.data_dir)
    if (root / "ratings.dat").exists() and (root / "movies.dat").exists():
        ratings = _read_dat(root / "ratings.dat", RATING_COLUMNS)
        movies = _read_dat(root / "movies.dat", MOVIE_COLUMNS)
    else:
        zips = [root] if root.is_file() and root.suffix.lower() == ".zip" else sorted(root.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"Не найдены ratings.dat/movies.dat или ZIP в {root}. Укажите ML1M_DIR.")
        with zipfile.ZipFile(zips[0]) as z:
            names = {Path(n).name: n for n in z.namelist()}
            if "ratings.dat" not in names or "movies.dat" not in names:
                raise FileNotFoundError(f"В архиве {zips[0]} нет ratings.dat и movies.dat")
            ratings = _read_dat(z.open(names["ratings.dat"]), RATING_COLUMNS)
            movies = _read_dat(z.open(names["movies.dat"]), MOVIE_COLUMNS)
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
    ratings = ratings.dropna(subset=["timestamp"]).copy()
    ratings[["user_id", "movie_id"]] = ratings[["user_id", "movie_id"]].astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings[ratings["rating"].between(1, 5)]
    ratings = ratings.sort_values(["timestamp", "user_id", "movie_id"], kind="mergesort")
    duplicates = int(ratings.duplicated(["user_id", "movie_id"]).sum())
    if duplicates:
        ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="last")
    ratings = ratings.reset_index(drop=True)
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return ratings, movies

def chronological_split(ratings: pd.DataFrame, min_ratings_per_user: int = 20):
    """Per-user chronological 80/10/10 split; train_final = train_fit + validation."""
    counts = ratings.groupby("user_id").size()
    active = counts[counts > min_ratings_per_user].index
    data = ratings[ratings.user_id.isin(active)].copy()
    fit_parts, val_parts, test_parts = [], [], []
    for user, group in data.groupby("user_id", sort=False):
        group = group.sort_values(["timestamp", "movie_id"], kind="mergesort").reset_index(drop=True)
        n = len(group)
        n_fit, n_val = int(n * .8), int(n * .1)
        if n_fit < 1 or n_val < 1 or n_fit + n_val >= n:
            raise ValueError(f"Недостаточно оценок для временного split у пользователя {user}: {n}")
        fit_parts.append(group.iloc[:n_fit])
        val_parts.append(group.iloc[n_fit:n_fit+n_val])
        test_parts.append(group.iloc[n_fit+n_val:])
    train_fit = pd.concat(fit_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    train_final = pd.concat([train_fit, val], ignore_index=True)
    return train_fit, val, train_final, test
