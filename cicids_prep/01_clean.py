"""
Этап 1. Очистка CICIDS2017 (GeneratedLabelledFlows / TrafficLabelling).

Что делает (только то, что НЕ зависит от разбиения train/test, то есть утечки нет):
  1. читает 8 CSV (кодировка latin-1), чистит имена колонок (в них пробелы);
  2. выбрасывает полностью пустые строки (в файле Thursday-Morning-WebAttacks их 288 602);
  3. удаляет дублирующую колонку 'Fwd Header Length.1' (идентична 'Fwd Header Length');
  4. нормализует метки ('Web Attack \x96 XSS' -> 'Web Attack - XSS');
  5. восстанавливает время: формат d/m/Y H:M, 12-часовые часы без AM/PM (часы 1..7 -> PM);
  6. inf -> NaN; заведомо некорректные отрицательные значения -> NaN
     (Init_Win_bytes_* = -1 это легальный маркер «окна нет», его НЕ трогаем);
  7. считает хэши: row_hash (признаки + метка, для дубликатов) и feat_hash (только признаки, для конфликтов меток);
  8. сохраняет один parquet: признаки float32 + метаданные (day, ts, src/dst ip, ports, label, ...).

НЕ делает: импутацию, масштабирование, удаление дубликатов. Это делается после разбиения (этап 3).

Запуск:  python 01_clean.py --src data/TL --out data/processed/clean.parquet
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd

ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Protocol", "Timestamp"]
# Destination Port оставляем среди признаков-кандидатов, но помечаем как «спорный» (см. этап 3)
NEG_BAD = [  # отрицательные значения здесь физически невозможны (баг CICFlowMeter)
    "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Min", "Fwd Header Length", "Bwd Header Length", "min_seg_size_forward",
]
DAY_RE = re.compile(r"(Monday|Tuesday|Wednesday|Thursday|Friday)", re.I)
DAY_DATE = {"monday": "2017-07-03", "tuesday": "2017-07-04", "wednesday": "2017-07-05",
            "thursday": "2017-07-06", "friday": "2017-07-07"}


def norm_label(s: pd.Series) -> pd.Series:
    """'Web Attack \x96 XSS' / 'Web Attack ï¿½ XSS' -> 'Web Attack - XSS'; остальные метки не трогаем."""
    def f(x: str) -> str:
        x = x.strip()
        if x.startswith("Web Attack"):
            x = "Web Attack - " + re.sub(r"^Web Attack\W+", "", x)
            x = x.replace("ï¿½", "").replace("Web Attack -  ", "Web Attack - ").strip()
        return x
    u = s.astype(str).unique()
    return s.astype(str).map({k: f(k) for k in u})


def parse_ts(ts: pd.Series) -> pd.Series:
    """Понедельник записан как 'dd/mm/YYYY HH:MM:SS', остальные дни как 'd/m/YYYY H:MM' (без секунд)."""
    t = pd.to_datetime(ts, format="%d/%m/%Y %H:%M", errors="coerce")
    if t.isna().any():
        t2 = pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        t = t.fillna(t2)
    h = t.dt.hour
    # захват шёл ~08:50-17:00, часы 1..7 это на самом деле 13..19 (12-часовой формат без AM/PM)
    return t + pd.to_timedelta(np.where(h.between(1, 7), 12, 0), unit="h")


def load_one(path: str) -> pd.DataFrame:
    name = os.path.basename(path)
    day = DAY_RE.search(name).group(1).lower()
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    n0 = len(df)
    df = df.dropna(how="all")  # пустые хвостовые строки
    df = df.dropna(subset=["Label"]).reset_index(drop=True)
    print(f"{name}: {n0} -> {len(df)} строк (убрано пустых: {n0 - len(df)})")

    if "Fwd Header Length.1" in df:
        assert (df["Fwd Header Length"] == df["Fwd Header Length.1"]).all()
        df = df.drop(columns="Fwd Header Length.1")

    label = norm_label(df["Label"])
    ts = parse_ts(df["Timestamp"])
    assert ts.notna().all(), "не разобралось время"

    feat_cols = [c for c in df.columns if c not in ID_COLS + ["Label"]]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in NEG_BAD:
        X.loc[X[c] < 0, c] = np.nan

    # хэш по значениям в float64 до приведения к float32 (чтобы не плодить ложные дубликаты)
    h = pd.util.hash_pandas_object(pd.concat([X.fillna(-123456.789), label.rename("Label")], axis=1),
                                   index=False).astype("uint64")

    hf = pd.util.hash_pandas_object(X.fillna(-123456.789), index=False).astype("uint64")  # без метки: для поиска конфликтов

    meta = pd.DataFrame({
        "day": day, "row_in_file": np.arange(len(df), dtype="int32"), "ts": ts,
        "src_ip": df["Source IP"].astype("category"), "dst_ip": df["Destination IP"].astype("category"),
        "src_port": pd.to_numeric(df["Source Port"], errors="coerce").astype("int32"),
        "protocol": pd.to_numeric(df["Protocol"], errors="coerce").astype("int16"),
        "label": label.astype("category"), "row_hash": h, "feat_hash": hf,
    })
    out = pd.concat([meta, X.astype("float32")], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="папка с 8 CSV из GeneratedLabelledFlows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "*.csv")))
    assert len(files) == 8, f"ожидалось 8 файлов, найдено {len(files)}"
    parts = [load_one(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    for c in ["day", "label", "src_ip", "dst_ip"]:
        df[c] = df[c].astype("category")
    df["flow_id"] = np.arange(len(df), dtype="int32")  # сквозной id
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print("\nИтого:", df.shape, "-> ", a.out)
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
