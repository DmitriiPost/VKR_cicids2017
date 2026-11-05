"""
Этап 3. Разбиение train / val / test. Результат: parquet (flow_id, split) + таблица «класс x split».

Протоколы:
  random  (A)  стратифицированное случайное по меткам. Сравнимо с литературой, но завышает оценки:
               почти одинаковые потоки одной атаки попадают и в train, и в test.
  time    (C)  ОСНОВНОЙ. Для каждой пары (день, метка) потоки упорядочены по времени (минутные корзины)
               и режутся на три смежных блока train | val | test; на стыках блоков опциональный зазор
               (gap) из пропущенных корзин. Все классы присутствуют во всех частях (если хватает данных),
               близнецы-потоки одной серии не расходятся по частям.
  lofo    (F)  leave-one-family-out: как time, но потоки заданного семейства (--holdout) целиком
               уходят в test (в train/val их нет). Для оценки обобщения на невиданную атаку.

Перед разбиением (флаги): удаление точных дубликатов (признаки+метка) и «конфликтных» потоков
(одинаковые признаки, разные метки, их нельзя выучить).

Запуск:
  python 03_make_splits.py --clean data/processed/clean.parquet --protocol time --out data/processed/split_time.parquet
  python 03_make_splits.py ... --protocol random --out .../split_random.parquet
  python 03_make_splits.py ... --protocol lofo --holdout "DoS Hulk" "DoS GoldenEye" --out .../split_lofo_dos.parquet
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["flow_id", "day", "ts", "row_in_file", "label", "row_hash", "feat_hash"]


def prefilter(df, dedup, drop_conflicts):
    n0 = len(df)
    info = {}
    if drop_conflicts:
        nl = df.groupby("feat_hash")["label"].transform("nunique")
        mask = nl > 1
        info["conflict_rows_dropped"] = int(mask.sum())
        df = df.loc[~mask]
    if dedup:
        df = df.sort_values(["ts", "row_in_file"])
        dup = df.duplicated("row_hash", keep="first")
        info["duplicate_rows_dropped"] = int(dup.sum())
        df = df.loc[~dup]
    print(f"prefilter: {n0} -> {len(df)}  {info}")
    return df


def split_random(df, f_train, f_val, seed):
    y = df["label"].astype(str).to_numpy()
    idx = np.arange(len(df))
    # классы с <3 объектов стратифицировать нельзя: в train
    cnt = pd.Series(y).value_counts()
    tiny = np.isin(y, cnt[cnt < 4].index)
    rest = idx[~tiny]
    tr, tmp = train_test_split(rest, train_size=f_train, stratify=y[rest], random_state=seed)
    va, te = train_test_split(tmp, train_size=f_val / (1 - f_train), stratify=y[tmp], random_state=seed)
    out = np.full(len(df), "gap", dtype=object)
    out[tr] = "train"; out[va] = "val"; out[te] = "test"; out[tiny] = "train"
    return pd.Series(out, index=df.index)


def split_time(df, f_train, f_val, gap_bins, min_bins_for_gap):
    """Смежные блоки по минутным корзинам внутри каждой пары (day, label)."""
    out = pd.Series("gap", index=df.index, dtype=object)
    d = df.assign(bin=df["ts"].dt.floor("min"))
    for (day, lab), g in d.groupby(["day", "label"], observed=True):
        cnt = g.groupby("bin").size().sort_index()
        n = cnt.sum()
        cum_end = cnt.cumsum() / n
        mid = cum_end - cnt / n / 2
        sp = np.where(mid < f_train, "train", np.where(mid < f_train + f_val, "val", "test"))
        sp = pd.Series(sp, index=cnt.index, dtype=object)
        if len(cnt) >= min_bins_for_gap and gap_bins > 0:
            orig = sp.to_numpy()
            arr = orig.copy()
            for i in np.flatnonzero(orig[1:] != orig[:-1]) + 1:  # границы ищем по ИСХОДНОЙ разметке
                arr[i:i + gap_bins] = "gap"  # только первые корзины позднего блока
            # gap не должен «съесть» целый блок: если съел, оставляем разметку без зазора
            if {"train", "val", "test"} <= set(arr) or not {"train", "val", "test"} <= set(orig):
                sp = pd.Series(arr, index=cnt.index, dtype=object)
        out.loc[g.index] = g["bin"].map(sp).to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=["random", "time", "lofo"], default="time")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-bins", type=int, default=1, help="сколько минутных корзин пропускать в начале каждого блока после границы")
    ap.add_argument("--min-bins-for-gap", type=int, default=10, help="зазор ставится, только если у пары (день,класс) >= стольких корзин")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--keep-conflicts", action="store_true")
    ap.add_argument("--holdout", nargs="*", default=[], help="для lofo: метки скрытого семейства")
    a = ap.parse_args()

    df = pd.read_parquet(a.clean, columns=COLS)
    df = prefilter(df, not a.no_dedup, not a.keep_conflicts)

    if a.protocol == "random":
        sp = split_random(df, a.train, a.val, a.seed)
    elif a.protocol == "time":
        sp = split_time(df, a.train, a.val, a.gap_bins, a.min_bins_for_gap)
    else:
        assert a.holdout, "для lofo нужен --holdout"
        hold = df["label"].astype(str).isin(a.holdout)
        sp = pd.Series("gap", index=df.index, dtype=object)
        sp.loc[~hold] = split_time(df.loc[~hold], a.train, a.val, a.gap_bins, a.min_bins_for_gap)
        sp.loc[hold] = "test"

    res = pd.DataFrame({"flow_id": df["flow_id"].to_numpy(), "split": sp.to_numpy(), "label": df["label"].astype(str).to_numpy()})
    res[["flow_id", "split"]].to_parquet(a.out, index=False)

    tab = pd.crosstab(res["label"], res["split"]).reindex(columns=["train", "val", "test", "gap"], fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total", ascending=False)
    print(f"\nПротокол: {a.protocol}\n", tab.to_string())
    tab.to_csv(a.out.replace(".parquet", "_counts.csv"))
    miss = [(l, s) for l in tab.index for s in ["train", "val", "test"] if tab.loc[l, s] == 0 and not (a.protocol == "lofo" and l in a.holdout and s != "test")]
    if miss:
        print("\nВНИМАНИЕ: пустые части:", miss)

    # диагностика утечки по времени: доля тестовых минут, которые встречаются и в train (для того же класса)
    d = df.assign(bin=df["ts"].dt.floor("min"), split=sp.to_numpy(), lab=df["label"].astype(str))
    d = d[d["lab"] != "BENIGN"]
    trk = set(zip(d.loc[d["split"] == "train", "lab"], d.loc[d["split"] == "train", "bin"]))
    te = d[d["split"] == "test"]
    if len(te):
        shared = np.mean([(l, b) in trk for l, b in zip(te["lab"], te["bin"])])
        print(f"\nдоля тестовых АТАКУЮЩИХ потоков, чья минутная корзина (того же класса) есть и в train: {shared:.3f}")


if __name__ == "__main__":
    main()
