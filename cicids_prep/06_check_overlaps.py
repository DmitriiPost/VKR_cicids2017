"""
Этап 6. Проверки перед обучением.

(1) Совпадающие векторы между выборками (после разбиения):
    - на уровне сырых признаков (feat_hash из clean.parquet): ожидаем 0 из-за фильтра дубликатов;
    - на уровне ФИНАЛЬНЫХ векторов (то, что видит модель: 68 признаков после удаления Destination Port и
      констант, signed-log, MinMax, clip, float32). Здесь возможны новые совпадения: потоки, различавшиеся только
      удалёнными колонками или обрезанные в 0/1. Считаем точные совпадения и «почти совпадения» (округление до
      --decimals знаков). Сравнения: val->train, test->train, test->val; train берётся и ПОЛНЫЙ (до подвыборки по
      классам), и тот, что реально видит модель (подвыборка из arrays_*.npz).
(2) Для протокола time: пересечения временных интервалов между train / val / test БЕЗ учёта классов
    (минутные корзины), число пересечений (непрерывных интервалов) и потоков в них; отдельно только атаки;
    плюс попарные пересечения блоков (день, класс) у разных классов.

Запуск:
  python 06_check_overlaps.py --clean processed/clean.parquet --processed processed --outdir processed/checks
"""
import argparse, json, os
import numpy as np
import pandas as pd
from prep_common import load_preprocessor, apply_preprocessor, hash_rows, encode_labels, CHUNK


# ---------------------------------------------------------------- (1) векторы
def block_hashes(block: pd.DataFrame, P, decimals: int):
    he, hr = [], []
    for i in range(0, len(block), CHUNK):
        X = apply_preprocessor(block.iloc[i:i + CHUNK], P)
        he.append(hash_rows(X))
        hr.append(hash_rows(np.round(X, decimals)))
    return np.concatenate(he), np.concatenate(hr)


def match(hA, yA, hB, yB):
    """Для каждой строки A: есть ли в B такой же хэш (любая метка) и такой же хэш с ТОЙ ЖЕ меткой."""
    B = pd.DataFrame({"h": hB, "y": yB}).drop_duplicates()
    any_ = np.isin(hA, B["h"].to_numpy())
    A = pd.DataFrame({"h": hA, "y": yA})
    same = A.merge(B.assign(f=1), on=["h", "y"], how="left")["f"].notna().to_numpy()
    return any_, same


def vectors(clean, processed, protocol, outdir, decimals):
    pj = os.path.join(processed, f"arrays_{protocol}_full.json")
    P = load_preprocessor(json.load(open(pj)))
    feats = P["feats"]
    npz = np.load(os.path.join(processed, f"arrays_{protocol}_full.npz"), allow_pickle=True)
    capped_ids = set(npz["flow_id_train"].tolist())
    df = pd.read_parquet(clean, columns=["flow_id", "label", "feat_hash"] + feats)
    sp = pd.read_parquet(os.path.join(processed, f"split_{protocol}.parquet"))
    df = df.merge(sp, on="flow_id")
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    y, names = encode_labels(df["label"], "full")
    df["y"] = y

    # 1a. сырые признаки
    raw = {s: df.loc[df.split == s, "feat_hash"].to_numpy() for s in ["train", "val", "test"]}
    raw_rows = []
    for a, b in [("val", "train"), ("test", "train"), ("test", "val")]:
        raw_rows.append({"protocol": protocol, "pair": f"{a}->{b}", "n": len(raw[a]), "raw_identical": int(np.isin(raw[a], raw[b]).sum())})

    # 1b. финальные векторы
    H = {}
    for s in ["train", "val", "test"]:
        blk = df[df.split == s]
        he, hr = block_hashes(blk, P, decimals)
        H[s] = {"exact": he, "round": hr, "y": blk["y"].to_numpy(), "fid": blk["flow_id"].to_numpy()}
    cap_mask = np.array([f in capped_ids for f in H["train"]["fid"]])
    refs = {"train_full": (H["train"]["exact"], H["train"]["round"], H["train"]["y"]),
            "train_capped": (H["train"]["exact"][cap_mask], H["train"]["round"][cap_mask], H["train"]["y"][cap_mask])}

    rows, per_class = [], []
    for a, bname in [("val", "train_full"), ("val", "train_capped"), ("test", "train_full"), ("test", "train_capped"),
                     ("test", "val")]:
        if bname == "val":
            bE, bR, bY = H["val"]["exact"], H["val"]["round"], H["val"]["y"]
        else:
            bE, bR, bY = refs[bname]
        for kind in ["exact", "round"]:
            hA = H[a][kind]; yA = H[a]["y"]
            any_, same = match(hA, yA, bE if kind == "exact" else bR, bY)
            rows.append({"protocol": protocol, "pair": f"{a}->{bname}", "kind": kind if kind == "exact" else f"round{decimals}",
                         "n": len(hA), "with_twin": int(any_.sum()), "share_%": round(100 * any_.mean(), 3),
                         "twin_same_label": int(same.sum()), "twin_only_other_label": int((any_ & ~same).sum())})
            if a == "test" and bname == "train_capped":
                for k, n in enumerate(names):
                    m = yA == k
                    if m.any():
                        per_class.append({"protocol": protocol, "kind": rows[-1]["kind"], "class": n, "n_test": int(m.sum()),
                                          "with_twin_in_train_capped": int(any_[m].sum()),
                                          "share_%": round(100 * any_[m].mean(), 2), "same_label": int(same[m].sum())})
    return pd.DataFrame(raw_rows), pd.DataFrame(rows), pd.DataFrame(per_class)


# ---------------------------------------------------------------- (2) время
def runs(minutes: np.ndarray) -> int:
    """число непрерывных (подряд идущих минут) интервалов в отсортированном массиве минут (в минутах от эпохи)."""
    if len(minutes) == 0:
        return 0
    return int(1 + (np.diff(minutes) > 1).sum())


def time_overlaps(clean, processed, protocol, outdir):
    d = pd.read_parquet(clean, columns=["flow_id", "day", "ts", "label"])
    sp = pd.read_parquet(os.path.join(processed, f"split_{protocol}.parquet"))
    d = d.merge(sp, on="flow_id")
    d = d[d["split"].isin(["train", "val", "test"])].copy()
    d["label"] = d["label"].astype(str)
    # минуты от начала недели (не зависит от разрешения datetime64: в pandas 3 это us, в pandas 2 ns)
    d["m"] = ((d["ts"].dt.floor("min") - pd.Timestamp("2017-07-03")) // pd.Timedelta(minutes=1)).astype("int64")
    d["att"] = d["label"] != "BENIGN"

    rows = []
    for scope, sub in [("все потоки (включая BENIGN)", d), ("только атаки", d[d.att])]:
        cnt = {s: sub[sub.split == s].groupby("m").size() for s in ["train", "val", "test"]}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            common = np.intersect1d(cnt[a].index.to_numpy(), cnt[b].index.to_numpy())
            fa = int(cnt[a].reindex(common).sum()); fb = int(cnt[b].reindex(common).sum())
            na, nb = int(cnt[a].sum()), int(cnt[b].sum())
            rows.append({"protocol": protocol, "scope": scope, "pair": f"{a} & {b}", "overlap_minutes": len(common),
                         "overlap_intervals": runs(common), f"flows_{a}_in_overlap": fa, f"share_{a}_%": round(100 * fa / max(na, 1), 2),
                         f"flows_{b}_in_overlap": fb, f"share_{b}_%": round(100 * fb / max(nb, 1), 2)})
    # доля АТАКУЮЩИХ потоков val/test, у которых в ту же минуту есть поток train (любого класса / BENIGN / другой атаки)
    rows2 = []
    tr_all = set(d.loc[d.split == "train", "m"]); tr_ben = set(d.loc[(d.split == "train") & ~d.att, "m"])
    for s in ["val", "test"]:
        x = d[(d.split == s) & d.att]
        row = {"protocol": protocol, "split": s, "attack_flows": len(x)}
        row["share_minute_has_any_train_flow_%"] = round(100 * x["m"].isin(tr_all).mean(), 2)
        row["share_minute_has_train_BENIGN_%"] = round(100 * x["m"].isin(tr_ben).mean(), 2)
        oth = []
        for lab, g in x.groupby("label"):
            t_other = set(d.loc[(d.split == "train") & d.att & (d.label != lab), "m"])
            oth.append(g["m"].isin(t_other).sum())
        row["share_minute_has_train_OTHER_attack_%"] = round(100 * sum(oth) / max(len(x), 1), 2)
        rows2.append(row)

    # блоки (день, класс, часть) и попарные пересечения у РАЗНЫХ классов
    blk = d.groupby(["day", "label", "split"], observed=True).agg(start=("ts", "min"), end=("ts", "max"), flows=("flow_id", "size")).reset_index()
    blk = blk.sort_values(["day", "start"])
    pairs = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        A = blk[blk.split == a]; B = blk[blk.split == b]
        n_pairs, flows_a, flows_b, ids_a, ids_b = 0, 0, 0, set(), set()
        for ia, ra in A.iterrows():
            for ib, rb in B.iterrows():
                if ra.label != rb.label and ra.start <= rb.end and rb.start <= ra.end:
                    n_pairs += 1; ids_a.add(ia); ids_b.add(ib)
        pairs.append({"protocol": protocol, "pair": f"{a} & {b}", "block_pairs_overlapping(different classes)": n_pairs,
                      f"blocks_{a}_involved": len(ids_a), f"flows_in_those_{a}_blocks": int(A.loc[list(ids_a), "flows"].sum()) if ids_a else 0,
                      f"blocks_{b}_involved": len(ids_b), f"flows_in_those_{b}_blocks": int(B.loc[list(ids_b), "flows"].sum()) if ids_b else 0})
    return pd.DataFrame(rows), pd.DataFrame(rows2), blk, pd.DataFrame(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--only", choices=["vectors", "time"], default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    if a.only in (None, "time"):
        T = [time_overlaps(a.clean, a.processed, p, a.outdir) for p in ["time", "random"]]
        pd.concat([t[0] for t in T]).to_csv(f"{a.outdir}/time_overlap_minutes.csv", index=False)
        pd.concat([t[1] for t in T]).to_csv(f"{a.outdir}/time_overlap_attack_flows.csv", index=False)
        T[0][2].to_csv(f"{a.outdir}/time_blocks_time_protocol.csv", index=False)
        pd.concat([t[3] for t in T]).to_csv(f"{a.outdir}/time_overlap_block_pairs.csv", index=False)
        for name, i in [("минутные пересечения (по сплитам, без учёта классов)", 0), ("атакующие потоки val/test: что было в train в ту же минуту", 1), ("пересечения блоков (день, класс) у разных классов", 3)]:
            print(f"\n=== {name} ===\n", pd.concat([t[i] for t in T]).to_string(index=False))
        print("\n=== блоки (день, класс, часть) для протокола time ===\n", T[0][2].to_string(index=False))

    if a.only in (None, "vectors"):
        R = [vectors(a.clean, a.processed, p, a.outdir, a.decimals) for p in ["time", "random"]]
        pd.concat([r[0] for r in R]).to_csv(f"{a.outdir}/vectors_raw_identical.csv", index=False)
        pd.concat([r[1] for r in R]).to_csv(f"{a.outdir}/vectors_final_overlap.csv", index=False)
        pd.concat([r[2] for r in R]).to_csv(f"{a.outdir}/vectors_final_overlap_by_class.csv", index=False)
        print("\n=== сырые признаки: идентичные потоки между частями ===\n", pd.concat([r[0] for r in R]).to_string(index=False))
        print("\n=== финальные векторы (то, что видит модель) ===\n", pd.concat([r[1] for r in R]).to_string(index=False))
        print("\n=== test -> train(подвыборка), по классам ===\n", pd.concat([r[2] for r in R]).to_string(index=False))


if __name__ == "__main__":
    main()
