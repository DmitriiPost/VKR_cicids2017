"""
Этап 7. Потоки Bot и SSH-Patator вне официальных временных окон атак.

Окна взяты со страницы датасета (unb.ca/cic/datasets/ids-2017.html), время локальное, как в CSV после
поправки часов (см. 01_clean.py):
    SSH-Patator  вт 2017-07-04  14:00 - 15:00
    Bot (Ares)   пт 2017-07-07  10:02 - 11:02

Что делает:
  1. находит потоки этих классов вне окна (с допуском --tol-min минут), считает расстояние до ближайшей границы
     (минуты, знак «-» = до начала окна, «+» = после конца);
  2. печатает распределение по расстоянию, источникам/целям/портам, сравнение «внутри окна / вне окна»
     по ключевым признакам и то, как такие потоки распределены по train/val/test (протокол time);
  3. сохраняет список потоков (csv) и файл flow_id для исключения (--exclude-out), который понимает
     03_make_splits.py --exclude-flow-ids.

Запуск:
  python 07_bot_ssh_windows.py --clean processed/clean.parquet --split processed/split_time.parquet \
      --outdir processed/checks --tol-min 0 --exclude-out processed/checks/exclude_bot_ssh_tol0.parquet
"""
import argparse, os
import numpy as np
import pandas as pd

WINDOWS = {
    "SSH-Patator": ("2017-07-04 14:00", "2017-07-04 15:00"),
    "Bot": ("2017-07-07 10:02", "2017-07-07 11:02"),
}
KEY = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Flow IAT Mean"]
BINS = [-1e9, -60, -30, -15, -5, 0, 5, 15, 30, 60, 120, 1e9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--split", required=True, help="split_time.parquet: чтобы показать, куда попали такие потоки")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tol-min", type=float, default=0.0, help="допуск в минутах вокруг официального окна")
    ap.add_argument("--exclude-out", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

    cols = ["flow_id", "label", "ts", "src_ip", "dst_ip", "Destination Port"] + KEY
    d = pd.read_parquet(a.clean, columns=cols)
    d["label"] = d["label"].astype(str)
    sp = pd.read_parquet(a.split)
    d = d.merge(sp, on="flow_id", how="left")
    d["split"] = d["split"].fillna("removed")  # удалён фильтром дубликатов/конфликтов

    out_rows, summary = [], []
    for lab, (s, e) in WINDOWS.items():
        x = d[d.label == lab].copy()
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        tol = pd.Timedelta(minutes=a.tol_min)
        before = (s - x.ts).dt.total_seconds() / 60          # >0: поток раньше начала
        after = (x.ts - e).dt.total_seconds() / 60           # >0: поток позже конца
        x["rel_min"] = np.where(before > 0, -before, np.where(after > 0, after, 0.0))
        x["outside"] = (x.ts < s - tol) | (x.ts > e + tol)
        o = x[x.outside]
        print(f"\n================ {lab}: окно {s:%H:%M}-{e:%H:%M}, допуск ±{a.tol_min:g} мин ================")
        print(f"всего {len(x)}, вне окна {len(o)} ({100*len(o)/len(x):.1f}%): до начала {int((o.rel_min<0).sum())}, после конца {int((o.rel_min>0).sum())}")
        print("\nрасстояние до ближайшей границы окна (мин; 0 = внутри):")
        print(pd.cut(x.rel_min, BINS).value_counts().sort_index().to_string())
        print("\nвремя вне окна (по 10 минут):\n", o.ts.dt.floor("10min").value_counts().sort_index().to_string())
        print(f"\nsrc_ip внутри: {x[~x.outside].src_ip.astype(str).value_counts().head(5).to_dict()}")
        print(f"src_ip вне:    {o.src_ip.astype(str).value_counts().head(6).to_dict()}")
        print(f"dst_ip внутри: {x[~x.outside].dst_ip.astype(str).value_counts().head(4).to_dict()}")
        print(f"dst_ip вне:    {o.dst_ip.astype(str).value_counts().head(5).to_dict()}")
        print(f"dst_port внутри: {x[~x.outside]['Destination Port'].value_counts().head(4).to_dict()}")
        print(f"dst_port вне:    {o['Destination Port'].value_counts().head(4).to_dict()}")
        cmp = pd.DataFrame({"медиана внутри": x[~x.outside][KEY].median(), "медиана вне": o[KEY].median()})
        print("\nключевые признаки:\n", cmp.to_string())
        ct = pd.crosstab(x.outside.map({False: "внутри окна", True: "вне окна"}), x.split).reindex(columns=["train", "val", "test", "gap", "removed"], fill_value=0)
        print("\nгде оказались потоки при разбиении C (по частям):\n", ct.to_string())
        ct["share_outside_%"] = np.nan
        tot = ct.sum()
        summary.append({"label": lab, "n": len(x), "outside": len(o),
                        **{f"outside_in_{k}": int(ct.loc["вне окна", k]) if "вне окна" in ct.index else 0 for k in ["train", "val", "test"]},
                        **{f"all_in_{k}": int(ct[k].sum()) for k in ["train", "val", "test"]}})
        out_rows.append(o.assign(window_start=s, window_end=e)[["flow_id", "label", "ts", "rel_min", "src_ip", "dst_ip", "Destination Port", "split"] + KEY])

    res = pd.concat(out_rows)
    res.to_csv(f"{a.outdir}/bot_ssh_outside_flows_tol{a.tol_min:g}.csv", index=False)
    S = pd.DataFrame(summary)
    for k in ["train", "val", "test"]:
        S[f"share_outside_of_{k}_%"] = (100 * S[f"outside_in_{k}"] / S[f"all_in_{k}"].replace(0, np.nan)).round(1)
    S.to_csv(f"{a.outdir}/bot_ssh_outside_summary_tol{a.tol_min:g}.csv", index=False)
    print("\n=== итог: доля потоков вне окна в каждой части (протокол C) ===\n", S.to_string(index=False))
    if a.exclude_out:
        res[["flow_id"]].drop_duplicates().to_parquet(a.exclude_out, index=False)
        print(f"\nсписок на исключение: {len(res)} потоков -> {a.exclude_out}")


if __name__ == "__main__":
    main()
