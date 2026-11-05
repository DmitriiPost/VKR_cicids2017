"""
Этап 4. Подготовка массивов для SCN / Random Forest по готовому разбиению.
ВСЕ статистики (константные колонки, медианы, масштаб) считаются ТОЛЬКО по train (до подвыборки).

Шаги:
  1. берём признаки (без ID/IP/Source Port/Timestamp/Protocol; Destination Port по умолчанию тоже выкидываем);
  2. выкидываем колонки, константные на train;
  3. NaN (бывшие inf и невозможные отрицательные значения) -> медиана train;
  4. тяжёлые хвосты: для колонок с max|x| > --log-thr на train применяем sign(x)*log1p(|x|);
  5. MinMax в [0,1] по train; val/test обрезаем (clip) в [0,1]   <- SCN-статья нормирует вход в [0,1];
  6. метки: full (15 классов) | merge_web (Web Attack x3 -> один класс) | binary (BENIGN=0, атака=1);
  7. train-подвыборка: не более --train-cap потоков на класс (случайно, seed); val/test в естественном распределении.

Выход: <out>.npz (X_train,y_train,X_val,y_val,X_test,y_test, flow_id_*, feature_names, class_names)
       <out>.json (параметры предобработки, чтобы повторить на новых данных)

Запуск:
  python 04_make_arrays.py --clean data/processed/clean.parquet --split data/processed/split_time.parquet \
      --out data/processed/arrays_time_full --labels full --train-cap 20000
"""
import argparse, json
import numpy as np
import pandas as pd

META = ["flow_id", "day", "row_in_file", "ts", "src_ip", "dst_ip", "src_port", "protocol", "label", "row_hash", "feat_hash"]


def encode_labels(lab: pd.Series, mode: str):
    lab = lab.astype(str)
    if mode == "binary":
        names = ["BENIGN", "ATTACK"]
        y = (lab != "BENIGN").astype("int64").to_numpy()
    else:
        if mode == "merge_web":
            lab = lab.where(~lab.str.startswith("Web Attack"), "Web Attack")
        names = ["BENIGN"] + sorted(l for l in lab.unique() if l != "BENIGN")
        m = {n: i for i, n in enumerate(names)}
        y = lab.map(m).to_numpy().astype("int64")
    return y, names


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", choices=["full", "merge_web", "binary"], default="full")
    ap.add_argument("--train-cap", type=int, default=20000, help="макс. потоков на класс в train; 0 = без ограничения")
    ap.add_argument("--keep-dst-port", action="store_true", help="оставить Destination Port как признак (по умолчанию нет)")
    ap.add_argument("--log-thr", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    allcols = pd.read_parquet(a.clean, columns=["flow_id"]).columns  # проверка доступа
    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(a.clean).names
    feats = [c for c in schema_cols if c not in META]
    if not a.keep_dst_port:
        feats = [c for c in feats if c != "Destination Port"]

    df = pd.read_parquet(a.clean, columns=["flow_id", "label"] + feats)
    sp = pd.read_parquet(a.split)
    df = df.merge(sp, on="flow_id", how="inner")
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    y_all, class_names = encode_labels(df["label"], a.labels)
    df["y"] = y_all

    tr = df["split"] == "train"
    # 2. константные на train
    nun = df.loc[tr, feats].nunique(dropna=True)
    const = nun[nun <= 1].index.tolist()
    feats = [c for c in feats if c not in const]
    # 3. медианы train
    med = df.loc[tr, feats].median()
    # 4. колонки для signed-log
    mx = df.loc[tr, feats].abs().max()
    logcols = mx[mx > a.log_thr].index.tolist()

    def prep(block: pd.DataFrame) -> np.ndarray:
        X = block[feats].fillna(med).to_numpy(dtype="float64")
        for c in logcols:
            j = feats.index(c)
            X[:, j] = slog(X[:, j])
        return X

    Xtr_full = prep(df.loc[tr])
    lo = Xtr_full.min(axis=0)
    hi = Xtr_full.max(axis=0)
    rngv = np.where(hi - lo > 0, hi - lo, 1.0)

    def scale(X):
        return np.clip((X - lo) / rngv, 0.0, 1.0).astype("float32")

    out = {}
    # train: подвыборка по классам
    idx_tr = np.flatnonzero(tr.to_numpy())
    ytr = df.loc[tr, "y"].to_numpy()
    keep = []
    for k in np.unique(ytr):
        ii = np.flatnonzero(ytr == k)
        if a.train_cap and len(ii) > a.train_cap:
            ii = rng.choice(ii, a.train_cap, replace=False)
        keep.append(ii)
    keep = np.sort(np.concatenate(keep))
    out["X_train"] = scale(Xtr_full[keep]); out["y_train"] = ytr[keep]
    out["flow_id_train"] = df.loc[tr, "flow_id"].to_numpy()[keep]
    del Xtr_full
    for s in ["val", "test"]:
        m = df["split"] == s
        out[f"X_{s}"] = scale(prep(df.loc[m]))
        out[f"y_{s}"] = df.loc[m, "y"].to_numpy()
        out[f"flow_id_{s}"] = df.loc[m, "flow_id"].to_numpy()

    np.savez_compressed(a.out + ".npz", **out, feature_names=np.array(feats), class_names=np.array(class_names))
    params = {"labels": a.labels, "train_cap": a.train_cap, "seed": a.seed, "dropped_constant": const,
              "dst_port_used": bool(a.keep_dst_port), "log_cols": logcols, "log_thr": a.log_thr,
              "features": feats, "median": med.to_dict(), "min": dict(zip(feats, lo.tolist())),
              "max": dict(zip(feats, hi.tolist())), "class_names": class_names}
    json.dump(params, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)

    print(f"признаков: {len(feats)} (константных убрано {len(const)}; signed-log у {len(logcols)})")
    rows = []
    for k, n in enumerate(class_names):
        rows.append([n] + [int((out[f'y_{s}'] == k).sum()) for s in ["train", "val", "test"]]
                    + [int(((df['y'] == k) & tr).sum())])
    print(pd.DataFrame(rows, columns=["класс", "train(подвыборка)", "val", "test", "train(до подвыборки)"]).to_string(index=False))
    for s in ["train", "val", "test"]:
        X = out[f"X_{s}"]
        print(f"{s}: X{X.shape}, min {X.min():.3f}, max {X.max():.3f}, доля нулей {np.mean(X == 0):.2f}, доля единиц {np.mean(X == 1):.4f}")


if __name__ == "__main__":
    main()
