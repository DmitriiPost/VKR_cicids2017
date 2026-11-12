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

Сама логика предобработки лежит в prep_common.py.

Выход: <out>.npz (X_train,y_train,X_val,y_val,X_test,y_test, flow_id_*, feature_names, class_names)
       <out>.json (параметры предобработки, чтобы повторить на новых данных)

Запуск:
  python 04_make_arrays.py --clean data/processed/clean.parquet --split data/processed/split_time.parquet \
      --out data/processed/arrays_time_full --labels full --train-cap 20000
"""
import argparse, json
import numpy as np
import pandas as pd
from prep_common import (feature_columns, encode_labels, fit_preprocessor, apply_preprocessor, cap_indices)


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

    feats = feature_columns(a.clean, a.keep_dst_port)
    df = pd.read_parquet(a.clean, columns=["flow_id", "label"] + feats)
    sp = pd.read_parquet(a.split)
    df = df.merge(sp, on="flow_id", how="inner")
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    y_all, class_names = encode_labels(df["label"], a.labels)
    df["y"] = y_all

    tr = (df["split"] == "train").to_numpy()
    P = fit_preprocessor(df.loc[tr], feats, a.log_thr)          # только train, до подвыборки
    ytr_full = df.loc[tr, "y"].to_numpy()
    keep = cap_indices(ytr_full, a.train_cap, a.seed)
    dtr = df.loc[tr].iloc[keep]

    out = {"X_train": apply_preprocessor(dtr, P), "y_train": dtr["y"].to_numpy(), "flow_id_train": dtr["flow_id"].to_numpy()}
    for s in ["val", "test"]:
        b = df[df["split"] == s]
        out[f"X_{s}"] = apply_preprocessor(b, P)
        out[f"y_{s}"] = b["y"].to_numpy()
        out[f"flow_id_{s}"] = b["flow_id"].to_numpy()

    feats2 = P["feats"]
    np.savez_compressed(a.out + ".npz", **out, feature_names=np.array(feats2), class_names=np.array(class_names))
    params = {"labels": a.labels, "train_cap": a.train_cap, "seed": a.seed, "dropped_constant": P["const"],
              "dst_port_used": bool(a.keep_dst_port), "log_cols": P["logcols"], "log_thr": a.log_thr,
              "features": feats2, "median": P["med"].to_dict(), "min": dict(zip(feats2, P["lo"].tolist())),
              "max": dict(zip(feats2, P["hi"].tolist())), "class_names": class_names}
    json.dump(params, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)

    print(f"признаков: {len(feats2)} (константных убрано {len(P['const'])}; signed-log у {len(P['logcols'])})")
    rows = []
    for k, n in enumerate(class_names):
        rows.append([n] + [int((out[f'y_{s}'] == k).sum()) for s in ["train", "val", "test"]] + [int((ytr_full == k).sum())])
    print(pd.DataFrame(rows, columns=["класс", "train(подвыборка)", "val", "test", "train(до подвыборки)"]).to_string(index=False))
    for s in ["train", "val", "test"]:
        X = out[f"X_{s}"]
        print(f"{s}: X{X.shape}, min {X.min():.3f}, max {X.max():.3f}, доля нулей {np.mean(X == 0):.2f}, доля единиц {np.mean(X == 1):.4f}")


if __name__ == "__main__":
    main()
