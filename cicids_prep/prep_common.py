"""
Общие функции предобработки (используются 04_make_arrays.py, 06_check_*.py, 08_rf_seeds.py).
Логика та же, что была внутри 04_make_arrays.py: статистики считаются ТОЛЬКО по train.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

META = ["flow_id", "day", "row_in_file", "ts", "src_ip", "dst_ip", "src_port", "protocol", "label", "row_hash", "feat_hash"]
CHUNK = 200_000


def feature_columns(clean_path: str, keep_dst_port: bool = False):
    feats = [c for c in pq.read_schema(clean_path).names if c not in META]
    if not keep_dst_port:
        feats = [c for c in feats if c != "Destination Port"]
    return feats


def encode_labels(lab: pd.Series, mode: str):
    lab = lab.astype(str)
    if mode == "binary":
        return (lab != "BENIGN").astype("int64").to_numpy(), ["BENIGN", "ATTACK"]
    if mode == "merge_web":
        lab = lab.where(~lab.str.startswith("Web Attack"), "Web Attack")
    names = ["BENIGN"] + sorted(l for l in lab.unique() if l != "BENIGN")
    m = {n: i for i, n in enumerate(names)}
    return lab.map(m).to_numpy().astype("int64"), names


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _prep(block, feats, med, logcols):
    X = block[feats].fillna(med).to_numpy(dtype="float64")
    for c in logcols:
        j = feats.index(c)
        X[:, j] = slog(X[:, j])
    return X


def fit_preprocessor(df_train: pd.DataFrame, feats, log_thr=1000.0):
    """df_train: только train-строки с колонками feats. Возвращает словарь параметров."""
    nun = df_train[feats].nunique(dropna=True)
    const = nun[nun <= 1].index.tolist()
    feats = [c for c in feats if c not in const]
    med = df_train[feats].median()
    mx = df_train[feats].abs().max()
    logcols = mx[mx > log_thr].index.tolist()
    lo = np.full(len(feats), np.inf)
    hi = np.full(len(feats), -np.inf)
    for i in range(0, len(df_train), CHUNK):
        X = _prep(df_train.iloc[i:i + CHUNK], feats, med, logcols)
        lo = np.minimum(lo, X.min(axis=0))
        hi = np.maximum(hi, X.max(axis=0))
    return {"feats": feats, "const": const, "med": med, "logcols": logcols, "lo": lo, "hi": hi, "log_thr": log_thr}


def load_preprocessor(json_params: dict):
    """Восстановить параметры из arrays_*.json (то, что сохраняет 04_make_arrays.py)."""
    feats = json_params["features"]
    return {"feats": feats, "const": json_params["dropped_constant"],
            "med": pd.Series(json_params["median"])[feats],
            "logcols": json_params["log_cols"],
            "lo": np.array([json_params["min"][c] for c in feats]),
            "hi": np.array([json_params["max"][c] for c in feats]),
            "log_thr": json_params["log_thr"]}


def apply_preprocessor(df_block: pd.DataFrame, P) -> np.ndarray:
    """-> float32 в [0,1], форма (n, len(feats)). Обрабатывает блоками, чтобы не раздувать память."""
    feats = P["feats"]
    rng = np.where(P["hi"] - P["lo"] > 0, P["hi"] - P["lo"], 1.0)
    out = np.empty((len(df_block), len(feats)), dtype="float32")
    for i in range(0, len(df_block), CHUNK):
        X = _prep(df_block.iloc[i:i + CHUNK], feats, P["med"], P["logcols"])
        out[i:i + CHUNK] = np.clip((X - P["lo"]) / rng, 0.0, 1.0).astype("float32")
    return out


def cap_indices(y: np.ndarray, cap: int, seed: int):
    """Индексы подвыборки: не более cap на класс (0 = без ограничения)."""
    rng = np.random.default_rng(seed)
    keep = []
    for k in np.unique(y):
        ii = np.flatnonzero(y == k)
        if cap and len(ii) > cap:
            ii = rng.choice(ii, cap, replace=False)
        keep.append(ii)
    return np.sort(np.concatenate(keep))


def hash_rows(X: np.ndarray) -> np.ndarray:
    """64-битный хэш каждой строки float32-матрицы (для поиска совпадающих векторов)."""
    U = np.ascontiguousarray(X, dtype="float32").view(np.uint32).astype(np.uint64)
    h = np.zeros(len(U), dtype=np.uint64)
    P = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for j in range(U.shape[1]):
            h = (h ^ U[:, j]) * P
    return h
