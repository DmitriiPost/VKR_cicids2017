"""
Этап 2. Профиль очищенного набора: то, что нужно знать до выбора протокола разделения.
Пишет таблицы в --outdir (csv) и печатает краткую сводку.

Запуск: python 02_profile.py --clean data/processed/clean.parquet --outdir data/processed/profile
"""
import argparse, os
import numpy as np
import pandas as pd

META = ["flow_id", "day", "row_in_file", "ts", "src_ip", "dst_ip", "src_port", "protocol", "label", "row_hash", "feat_hash"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_parquet(a.clean)
    feats = [c for c in df.columns if c not in META]
    print(f"строк {len(df)}, признаков (вкл. Destination Port) {len(feats)}")

    # --- 1. классы x день
    ct = pd.crosstab(df["label"], df["day"])
    ct["TOTAL"] = ct.sum(axis=1)
    ct.loc[:, :].sort_values("TOTAL", ascending=False).to_csv(f"{a.outdir}/class_by_day.csv")
    print("\n== классы x день ==\n", ct.sort_values("TOTAL", ascending=False).to_string())

    # --- 2. окна времени, источники и цели по классам
    g = df.groupby("label", observed=True)
    win = pd.DataFrame({
        "n": g.size(), "ts_min": g["ts"].min(), "ts_max": g["ts"].max(),
        "n_src_ip": g["src_ip"].nunique(), "n_dst_ip": g["dst_ip"].nunique(),
        "n_dst_port": g["Destination Port"].nunique(),
        "n_minutes_active": g["ts"].apply(lambda s: s.dt.floor("min").nunique()),
    })
    win["top_src_ip"] = g["src_ip"].agg(lambda s: s.value_counts().index[0])
    win["top_src_share"] = g["src_ip"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).round(3)
    win["top_dst_ip"] = g["dst_ip"].agg(lambda s: s.value_counts().index[0])
    win.sort_values("n", ascending=False).to_csv(f"{a.outdir}/class_windows.csv")
    print("\n== окна/источники/цели ==\n", win.sort_values("n", ascending=False).to_string())

    # --- 3. NaN после очистки (inf и невозможные отрицательные уже превращены в NaN)
    nan_rows = df[feats].isna().any(axis=1)
    nan_by_class = pd.DataFrame({"n": df.groupby("label", observed=True).size(),
                                 "rows_with_nan": nan_rows.groupby(df["label"], observed=True).sum()})
    nan_by_class["share_%"] = (100 * nan_by_class["rows_with_nan"] / nan_by_class["n"]).round(3)
    nan_by_class.to_csv(f"{a.outdir}/nan_by_class.csv")
    nan_cols = df[feats].isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
    nan_cols.to_csv(f"{a.outdir}/nan_by_column.csv", header=["n_nan"])
    print("\n== строк с NaN по классам ==\n", nan_by_class.to_string())
    print("\n== колонки с NaN ==\n", nan_cols.to_string())

    # --- 4. константные колонки
    nun = df[feats].nunique(dropna=True)
    const = nun[nun <= 1].index.tolist()
    nun.to_csv(f"{a.outdir}/n_unique_by_column.csv", header=["n_unique"])
    print("\n== константные колонки (удалить) ==\n", const)

    # --- 5. дубликаты (признаки + метка, хэш считался в float64 на этапе 1)
    dup = df.duplicated("row_hash", keep=False)
    dup_extra = df.duplicated("row_hash", keep="first")
    d = pd.DataFrame({"n": df.groupby("label", observed=True).size(),
                      "extra_dups": dup_extra.groupby(df["label"], observed=True).sum()})
    d["unique_after_dedup"] = d["n"] - d["extra_dups"]
    d["dup_share_%"] = (100 * d["extra_dups"] / d["n"]).round(2)
    d.sort_values("n", ascending=False).to_csv(f"{a.outdir}/duplicates_by_class.csv")
    print("\n== дубликаты (признаки+метка) ==\n", d.sort_values("n", ascending=False).to_string())
    print(f"всего лишних дубликатов: {int(dup_extra.sum())} ({100*dup_extra.mean():.1f}%)")

    # --- 6. конфликтные строки: одинаковые признаки, но разные метки (хэш без метки)
    key = df["feat_hash"]
    nl = df.groupby(key)["label"].transform("nunique")
    conf = df.loc[nl > 1]
    print(f"\n== конфликтные строки (одни признаки, разные метки): {len(conf)} ==")
    if len(conf):
        print(conf["label"].value_counts().to_string())
        conf_pairs = (conf.assign(k=key[conf.index]).groupby("k")["label"]
                      .agg(lambda s: " | ".join(sorted(set(map(str, s))))).value_counts().head(15))
        print("\nнаиболее частые пары меток в конфликтах:\n", conf_pairs.to_string())
        conf_pairs.to_csv(f"{a.outdir}/conflict_label_pairs.csv", header=["n_groups"])

    # --- 7. общий публичный IP (важно для группировки по IP, п. «Замечания к статье»)
    victims = df["dst_ip"].astype(str).value_counts().head(8)
    print("\n== топ целей (dst_ip) ==\n", victims.to_string())
    atk_src = df.loc[df["label"] != "BENIGN"].groupby("src_ip", observed=True).size().loc[lambda s: s > 0] \
        .sort_values(ascending=False).head(10)
    print("\n== топ источников среди атакующих потоков ==\n", atk_src.to_string())
    # доля атак среди всех потоков от этих IP: насколько IP сам по себе выдаёт класс
    ipc = df.groupby("src_ip", observed=True)["label"].agg(lambda s: (s != "BENIGN").mean())
    ipn = df.groupby("src_ip", observed=True).size()
    leak = pd.DataFrame({"n": ipn, "attack_share": ipc}).loc[atk_src.index]
    print("\n доля атак среди потоков этих src_ip:\n", leak.round(3).to_string())
    leak.to_csv(f"{a.outdir}/attacker_ips.csv")

    # --- 8. порядок времени внутри файла (можно ли полагаться на row_in_file)
    mono = df.groupby("day", observed=True).apply(lambda x: (x.sort_values("row_in_file")["ts"].diff().dropna() >= pd.Timedelta(0)).mean())
    print("\n== доля неубывающих шагов времени по порядку строк в файле ==\n", mono.round(3).to_string())

    # --- 9. benign по дням и часам (для блочного разбиения)
    b = df.loc[df["label"] == "BENIGN"]
    print("\n== benign по дням ==\n", b["day"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
