"""
Этап 9. Эксперименты с Destination Port (Random Forest).

Базовое разбиение (--split) — stratified temporal split (протокол C, `03_make_splits.py --protocol time`).

--exp port        Эксперимент 1: разбиение без изменений. Сравнивается модель С портом (69 признаков)
                  и БЕЗ порта (68 признаков): флаг --use-port 1 / 0.
--exp portsplit   Эксперимент 2: PortScan-потоки переразбиваются по ПОРТАМ: непересекающиеся наборы
                  целевых портов train / val / test (порты случайно перемешиваются по сиду и раскладываются
                  в пропорции ~70/15/15 по числу потоков). Остальные классы остаются как в протоколе C.
                  Сид меняет разбиение портов, подвыборку train и лес.

Для каждого запуска (сид) записывается: accuracy, macro-F1 (все классы и классы с >= --min-support потоков в test),
F1/P/R класса PortScan и число совпадающих ВЕКТОРОВ между выборками (точные копии финальных векторов, то есть тех
признаков, которые видит модель: с портом или без него):
  val->train, test->train (весь train и подвыборка, на которой обучается модель), test->val;
  отдельно тестовые PortScan-потоки с копией в train; число различных общих векторов train & test.

Результат: <outdir>/ports_<tag>.csv (по строке на сид) и <outdir>/ports_<tag>_perclass.csv (F1 по классам).

Запуск:
  python 09_port_experiments.py --clean processed/clean.parquet --split processed/split_time.parquet \
      --exp port --use-port 1 --tag base_port_with_cap20000 --outdir processed/checks
"""
import argparse, importlib.util, os, sys, time, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_common import (feature_columns, encode_labels, fit_preprocessor, apply_preprocessor, cap_indices,
                         hash_rows, CHUNK)


def load_rf_module():
    spec = importlib.util.spec_from_file_location("rf_seeds", os.path.join(HERE, "08_rf_seeds.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def port_disjoint_split(sp: pd.DataFrame, ports: pd.DataFrame, seed: int, f_train=0.70, f_val=0.15):
    """sp: flow_id, split (протокол C). ports: flow_id, label, dport. Возвращает новый sp и сведения о портах."""
    ps = ports[ports["label"] == "PortScan"][["flow_id", "dport"]].merge(sp[["flow_id"]], on="flow_id")
    cnt = ps.groupby("dport").size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(cnt.index.to_numpy())
    c = cnt.reindex(order).to_numpy()
    mid = (np.cumsum(c) - c / 2) / c.sum()
    part = np.where(mid < f_train, "train", np.where(mid < f_train + f_val, "val", "test"))
    port2split = dict(zip(order, part))
    new = sp.copy()
    is_ps = new["flow_id"].isin(ps["flow_id"])
    fid2port = dict(zip(ps["flow_id"], ps["dport"]))
    new.loc[is_ps, "split"] = new.loc[is_ps, "flow_id"].map(lambda f: port2split[fid2port[f]]).to_numpy()
    info = {f"ports_{s}": int((part == s).sum()) for s in ["train", "val", "test"]}
    info.update({f"portscan_flows_{s}": int(new.loc[is_ps, "split"].eq(s).sum()) for s in ["train", "val", "test"]})
    return new, info


def block_hash(df_block, P):
    out = []
    for i in range(0, len(df_block), CHUNK):
        out.append(hash_rows(apply_preprocessor(df_block.iloc[i:i + CHUNK], P)))
    return np.concatenate(out) if out else np.empty(0, dtype=np.uint64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--split", required=True, help="split_time.parquet (stratified temporal split) или его вариант")
    ap.add_argument("--exp", choices=["port", "portsplit"], required=True)
    ap.add_argument("--use-port", type=int, choices=[0, 1], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="checks")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cap", type=int, default=20000, help="0 = без ограничения на класс")
    ap.add_argument("--trees", type=int, default=60)
    ap.add_argument("--min-support", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out_runs = os.path.join(a.outdir, f"ports_{a.tag}.csv")
    out_pc = os.path.join(a.outdir, f"ports_{a.tag}_perclass.csv")
    for f in (out_runs, out_pc):
        if os.path.exists(f):
            os.remove(f)
    rfm = load_rf_module()

    feats_all = feature_columns(a.clean, keep_dst_port=True)
    feats = feats_all if a.use_port else [c for c in feats_all if c != "Destination Port"]
    base = pd.read_parquet(a.clean, columns=["flow_id", "label"] + feats_all)
    _, names = encode_labels(base["label"], "full")
    name2k = {n: i for i, n in enumerate(names)}
    k_ps = name2k["PortScan"]
    base["y"] = base["label"].astype(str).map(name2k).to_numpy()
    ports = pd.DataFrame({"flow_id": base["flow_id"].to_numpy(), "label": base["label"].astype(str).to_numpy(),
                          "dport": base["Destination Port"].to_numpy()})
    base = base.drop(columns=["label"])
    sp0 = pd.read_parquet(a.split)
    print(f"[{a.tag}] признаков {len(feats)}, exp={a.exp}, cap={a.cap}, trees={a.trees}", flush=True)

    fixed = None
    for seed in a.seeds:
        t0 = time.time()
        info = {}
        if a.exp == "port":
            sp = sp0
        else:
            sp, info = port_disjoint_split(sp0, ports, seed)
        smap = pd.Series(sp["split"].to_numpy(), index=sp["flow_id"].to_numpy())
        split_col = base["flow_id"].map(smap)
        m_tr = (split_col == "train").to_numpy(); m_va = (split_col == "val").to_numpy(); m_te = (split_col == "test").to_numpy()
        tr = base.loc[m_tr]
        if a.exp == "port" and fixed is not None:
            P, h_tr_full = fixed
        else:
            P = fit_preprocessor(tr, feats)
            h_tr_full = None
        ytr_full = tr["y"].to_numpy()
        keep = cap_indices(ytr_full, a.cap, seed)
        if h_tr_full is None:
            h_tr_full = block_hash(tr, P)
            if a.exp == "port":
                fixed = (P, h_tr_full)
        Xtr = apply_preprocessor(tr.iloc[keep], P); ytr = ytr_full[keep]
        h_tr_cap = h_tr_full[keep]
        del tr
        va = base.loc[m_va]; te = base.loc[m_te]
        Xva = apply_preprocessor(va, P); Xte = apply_preprocessor(te, P)
        yva = va["y"].to_numpy(); yte = te["y"].to_numpy()
        h_va = hash_rows(Xva); h_te = hash_rows(Xte)
        del va, te, Xva
        gc.collect()

        tw_va_tr = np.isin(h_va, h_tr_full)
        tw_te_trf = np.isin(h_te, h_tr_full)
        tw_te_trc = np.isin(h_te, h_tr_cap)
        tw_te_va = np.isin(h_te, h_va)
        ps_te = yte == k_ps
        row = {"tag": a.tag, "exp": a.exp, "use_port": a.use_port, "cap": a.cap, "trees": a.trees, "seed": seed,
               "n_features": len(feats), "n_train_model": len(ytr), "n_train_full": len(ytr_full), "n_val": len(yva), "n_test": len(yte),
               "twins_val_in_train_full": int(tw_va_tr.sum()), "twins_test_in_train_full": int(tw_te_trf.sum()),
               "twins_test_in_train_model": int(tw_te_trc.sum()), "twins_test_in_val": int(tw_te_va.sum()),
               "shared_distinct_vectors_train_test": int(len(np.intersect1d(np.unique(h_tr_full), np.unique(h_te)))),
               "portscan_test": int(ps_te.sum()), "portscan_test_twin_in_train_full": int(tw_te_trf[ps_te].sum()),
               "portscan_train_model": int((ytr == k_ps).sum())}
        row.update(info)
        rf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)
        rows = rfm.evaluate(yte, yp, names, a.min_support)
        by = {r["class"]: r for r in rows}
        row["accuracy"] = by["__ACCURACY__"]["F1"]
        row["macro_f1_all"] = by["__MACRO_F1_all_classes__"]["F1"]
        row["macro_f1_support_ge_min"] = by[f"__MACRO_F1_support>={a.min_support}__"]["F1"]
        for k in ["P", "R", "F1"]:
            row[f"portscan_{k}"] = by["PortScan"][k] if "PortScan" in by else np.nan
        pd.DataFrame([row]).to_csv(out_runs, mode="a", header=not os.path.exists(out_runs), index=False)
        pd.DataFrame(rows).assign(tag=a.tag, seed=seed).to_csv(out_pc, mode="a", header=not os.path.exists(out_pc), index=False)
        print(f"[{a.tag}] seed {seed}: train {len(ytr)}/{len(ytr_full)}, test {len(yte)}, acc {row['accuracy']:.4f}, "
              f"macroF1 {row['macro_f1_all']:.4f}, PortScan F1 {row['portscan_F1']:.3f} (n={row['portscan_test']}), "
              f"копий test->train {row['twins_test_in_train_full']}, {time.time()-t0:.0f}s", flush=True)
        del Xtr, Xte, rf
        gc.collect()

    R = pd.read_csv(out_runs)
    cols = ["accuracy", "macro_f1_all", "macro_f1_support_ge_min", "portscan_F1", "twins_test_in_train_full", "twins_test_in_train_model", "portscan_test_twin_in_train_full"]
    print(f"\n=== {a.tag}: среднее ± std по {len(R)} сидам ===")
    for c in cols:
        print(f"{c:36s} {R[c].mean():.4f} ± {R[c].std(ddof=1) if len(R) > 1 else float('nan'):.4f}")


if __name__ == "__main__":
    main()
