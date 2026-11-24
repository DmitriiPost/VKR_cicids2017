"""
Этап 10. Сводные таблицы по экспериментам с Destination Port (результаты 09_port_experiments.py).

Читает <checks>/ports_<tag>.csv, пишет <checks>/ports_summary.csv (по строке на тег: среднее и std по сидам)
и печатает таблицы в markdown.

Имя тега: <набор>_<эксперимент>_<with|no>port_cap<cap>_t<деревья>, где
  набор:      base (дубликаты/конфликты по всем колонкам) | dedup (по признакам модели, --dedup-on model)
  эксперимент: port (разбиение C как есть) | portsplit (PortScan разбит по непересекающимся наборам портов)

Запуск: python 10_summarize_ports.py --checks processed/checks
"""
import argparse, glob, os, re
import pandas as pd


def ms(x, d=3):
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def fmt_int(x):
    return f"{int(round(x.mean())):,}".replace(",", " ")


def load(checks):
    rows = []
    for f in sorted(glob.glob(os.path.join(checks, "ports_*.csv"))):
        if f.endswith("_perclass.csv"):
            continue
        R = pd.read_csv(f)
        rows.append(R)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parse(tag):
    m = re.match(r"(base|dedup)_(port|portsplit)_(with|no)port_cap(\d+)_t(\d+)", tag)
    return dict(zip(["dataset", "exp", "port", "cap", "trees"], m.groups())) if m else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", required=True)
    a = ap.parse_args()
    R = load(a.checks)
    if R.empty:
        raise SystemExit("нет ports_*.csv")
    out = []
    for tag, g in R.groupby("tag"):
        d = parse(tag)
        out.append({**d, "tag": tag, "n_seeds": len(g),
                    "macro_f1_all_mean": g.macro_f1_all.mean(), "macro_f1_all_std": g.macro_f1_all.std(ddof=1),
                    "macro_f1_ge50_mean": g.macro_f1_support_ge_min.mean(), "macro_f1_ge50_std": g.macro_f1_support_ge_min.std(ddof=1),
                    "accuracy_mean": g.accuracy.mean(),
                    "portscan_F1_mean": g.portscan_F1.mean(), "portscan_F1_std": g.portscan_F1.std(ddof=1),
                    "portscan_test_mean": g.portscan_test.mean(),
                    "twins_val_in_train_full": g.twins_val_in_train_full.mean(),
                    "twins_test_in_train_full": g.twins_test_in_train_full.mean(),
                    "twins_test_in_train_model": g.twins_test_in_train_model.mean(),
                    "twins_test_in_val": g.twins_test_in_val.mean(),
                    "portscan_test_twin_in_train_full": g.portscan_test_twin_in_train_full.mean(),
                    "shared_distinct_train_test": g.shared_distinct_vectors_train_test.mean(),
                    "n_train_model": g.n_train_model.mean(), "n_test_mean": g.n_test.mean()})
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(a.checks, "ports_summary.csv"), index=False)

    def table(sel, title):
        print(f"\n### {title}\n")
        print("| Набор | Порт | macro-F1 (все) | macro-F1 (≥50) | F1 PortScan | PortScan в test | копии test→train | из них PortScan | копии val→train | копии test→val | общих различных векторов train∩test | seeds |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sel.iterrows():
            g = R[R.tag == r.tag]
            print(f"| {'базовый' if r.dataset == 'base' else 'dedup-on'} | {'с портом' if r.port == 'with' else 'без порта'} | {ms(g.macro_f1_all)} | "
                  f"{ms(g.macro_f1_support_ge_min)} | {ms(g.portscan_F1)} | {fmt_int(g.portscan_test)} | {fmt_int(g.twins_test_in_train_full)} | "
                  f"{fmt_int(g.portscan_test_twin_in_train_full)} | {fmt_int(g.twins_val_in_train_full)} | {fmt_int(g.twins_test_in_val)} | "
                  f"{fmt_int(g.shared_distinct_vectors_train_test)} | {len(g)} |")

    main_ = S[(S.cap == "20000") & (S.trees == "60")]
    for exp, title in [("port", "Эксперимент 1: протокол C, с портом и без порта (ограничение 20 000, 60 деревьев)"),
                       ("portsplit", "Эксперимент 2: PortScan по непересекающимся наборам портов (ограничение 20 000, 60 деревьев)")]:
        sel = main_[main_.exp == exp].sort_values(["dataset", "port"], ascending=[True, False])
        table(sel, title)

    cmp_ = S[(S.trees == "30") & (S.dataset == "base")]
    if len(cmp_):
        print("\n### Ограничение на класс: 20 000 и без ограничения (базовый набор, 30 деревьев, 3 сида)\n")
        print("| Эксперимент | Порт | Ограничение | строк в train | macro-F1 (все) | macro-F1 (≥50) | accuracy | F1 PortScan |")
        print("|---|---|---|---:|---:|---:|---:|---:|")
        for _, r in cmp_.sort_values(["exp", "port", "cap"], ascending=[True, False, False]).iterrows():
            g = R[R.tag == r.tag]
            cap = "20 000" if r.cap == "20000" else "нет"
            print(f"| {'1 (порт)' if r.exp == 'port' else '2 (порты PortScan)'} | {'с портом' if r.port == 'with' else 'без порта'} | {cap} | {fmt_int(g.n_train_model)} | "
                  f"{ms(g.macro_f1_all)} | {ms(g.macro_f1_support_ge_min)} | {g.accuracy.mean():.4f} | {ms(g.portscan_F1)} |")


if __name__ == "__main__":
    main()
