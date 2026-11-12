"""
Этап 8. Random Forest на нескольких сидах (по умолчанию 0..4).

Что меняется от сида:
  * подвыборка train (не более --cap потоков на класс);
  * random_state леса;
  * для протокола random ещё и само разбиение (03_make_splits.split_random с этим сидом).
Протокол time детерминирован: разбиение одно и то же, сид влияет только на подвыборку и лес.
Статистики предобработки считаются по ПОЛНОМУ train (до подвыборки), как в 04_make_arrays.py.

Метрики на test (естественное распределение): accuracy, macro-F1 по классам, которые есть в test,
macro-F1 по классам с >= --min-support потоками в test, P/R/F1 по каждому классу.
Результат дописывается в csv после каждого сида (прогресс не теряется).

Запуск (фиксированное разбиение):
  python 08_rf_seeds.py --clean processed/clean.parquet --protocol time --split processed/split_time.parquet --tag time_base
Запуск (случайное, разбиение строится по сиду):
  python 08_rf_seeds.py --clean processed/clean.parquet --protocol random --tag random_base [--dedup-on model]
"""
import argparse, importlib.util, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_common import feature_columns, encode_labels, fit_preprocessor, apply_preprocessor, cap_indices


def load_splits_module():
    spec = importlib.util.spec_from_file_location("make_splits", os.path.join(HERE, "03_make_splits.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def evaluate(yt, yp, names, min_support):
    labs = [k for k in range(len(names)) if (yt == k).any()]
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labs, zero_division=0)
    rows = [{"class": names[k], "n_test": int(si), "P": pi, "R": ri, "F1": fi} for k, pi, ri, fi, si in zip(labs, p, r, f, s)]
    big = [x["F1"] for x in rows if x["n_test"] >= min_support]
    rows.append({"class": "__ACCURACY__", "n_test": len(yt), "P": np.nan, "R": np.nan, "F1": accuracy_score(yt, yp)})
    rows.append({"class": "__MACRO_F1_all_classes__", "n_test": len(yt), "P": np.nan, "R": np.nan, "F1": float(np.mean(f))})
    rows.append({"class": f"__MACRO_F1_support>={min_support}__", "n_test": len(yt), "P": np.nan, "R": np.nan, "F1": float(np.mean(big)) if big else np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--protocol", choices=["time", "random"], required=True)
    ap.add_argument("--split", default=None, help="для time: parquet с разбиением")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="checks")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--trees", type=int, default=60)
    ap.add_argument("--min-support", type=int, default=50)
    ap.add_argument("--dedup-on", choices=["all", "model"], default="all", help="только для random: как в 03_make_splits.py")
    ap.add_argument("--exclude-flow-ids", default=None, help="только для random: parquet со столбцом flow_id")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, f"rf_{a.tag}.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)

    feats = feature_columns(a.clean, False)
    if a.protocol == "random":  # фильтр (может читать все признаки) делаем ДО загрузки основной матрицы: экономим память
        ms = load_splits_module()
        meta = pd.read_parquet(a.clean, columns=ms.COLS)
        excl = set(pd.read_parquet(a.exclude_flow_ids)["flow_id"].tolist()) if a.exclude_flow_ids else None
        meta = ms.prefilter(meta, True, True, a.dedup_on, a.clean, excl)
        import gc; gc.collect()
    base = pd.read_parquet(a.clean, columns=["flow_id", "label"] + feats)
    _, names = encode_labels(base["label"], "full")
    name2k = {n: i for i, n in enumerate(names)}
    base["y"] = base["label"].astype(str).map(name2k).to_numpy()
    base = base.drop(columns=["label"])

    fixed = None
    if a.protocol == "time":
        sp = pd.read_parquet(a.split)
        fixed = base.merge(sp, on="flow_id", how="inner")
        fixed = fixed[fixed["split"].isin(["train", "val", "test"])].reset_index(drop=True)
        tr = fixed[fixed.split == "train"]
        P = fit_preprocessor(tr, feats)
        Xte = apply_preprocessor(fixed[fixed.split == "test"], P); yte = fixed.loc[fixed.split == "test", "y"].to_numpy()
        print(f"[{a.tag}] fixed split: train {len(tr)}, test {len(yte)}, признаков {len(P['feats'])}", flush=True)

    for seed in a.seeds:
        t0 = time.time()
        if a.protocol == "random":
            spl = ms.split_random(meta, 0.70, 0.15, seed)
            sp = pd.DataFrame({"flow_id": meta["flow_id"].to_numpy(), "split": spl.to_numpy()})
            d = base.merge(sp, on="flow_id", how="inner")
            tr = d[d.split == "train"]
            P = fit_preprocessor(tr, feats)
            te = d[d.split == "test"]
            Xte = apply_preprocessor(te, P); yte = te["y"].to_numpy()
        ytr_full = tr["y"].to_numpy()
        keep = cap_indices(ytr_full, a.cap, seed)
        Xtr = apply_preprocessor(tr.iloc[keep], P); ytr = ytr_full[keep]
        rf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)
        rows = evaluate(yte, yp, names, a.min_support)
        res = pd.DataFrame(rows).assign(seed=seed, tag=a.tag, cap=a.cap, n_train=len(ytr), n_test_total=len(yte))
        res.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        acc = [r for r in rows if r["class"] == "__ACCURACY__"][0]["F1"]
        mf = [r for r in rows if r["class"] == "__MACRO_F1_all_classes__"][0]["F1"]
        print(f"[{a.tag}] seed {seed}: train {len(ytr)}, test {len(yte)}, acc {acc:.4f}, macroF1 {mf:.4f}, {time.time()-t0:.0f}s", flush=True)
        del Xtr, rf
        if a.protocol == "random":
            del d, tr, te, Xte

    R = pd.read_csv(out_csv)
    g = R.groupby("class")
    S = pd.DataFrame({"n_seeds": g["F1"].count(), "F1_mean": g["F1"].mean(), "F1_std": g["F1"].std(ddof=1),
                      "F1_min": g["F1"].min(), "F1_max": g["F1"].max(), "n_test_mean": g["n_test"].mean().round(0)})
    S.to_csv(os.path.join(a.outdir, f"rf_{a.tag}_summary.csv"))
    pd.set_option("display.width", 200)
    print(f"\n=== {a.tag}: среднее по {len(a.seeds)} сидам ===\n", S.round(4).to_string())


if __name__ == "__main__":
    main()
