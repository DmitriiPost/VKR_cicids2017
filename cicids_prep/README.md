# CICIDS2017: подготовка данных для SCN / Random Forest

Код очищает CICIDS2017, строит три варианта разбиения train/val/test и готовит массивы для обучения.
Подробное описание обработки, статистика классов и обоснование разбиения: **`processed/DATA_PREPARATION.md`**.

## Установка

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Данные

Скачать со страницы https://www.unb.ca/cic/datasets/ids-2017.html архив `GeneratedLabelledFlows.zip`, распаковать 8 CSV в одну папку (далее `data/TL`).
`MachineLearningCSV.zip` не нужен: в нём нет IP и времени.

## Запуск

```
bash run_all.sh data/TL processed
```

или по шагам:

```
python 01_clean.py        --src data/TL --out processed/clean.parquet
python 02_profile.py      --clean processed/clean.parquet --outdir processed/profile
python 03_make_splits.py  --clean processed/clean.parquet --protocol time   --out processed/split_time.parquet
python 03_make_splits.py  --clean processed/clean.parquet --protocol random --out processed/split_random.parquet
python 04_make_arrays.py  --clean processed/clean.parquet --split processed/split_time.parquet   --out processed/arrays_time_full   --labels full --train-cap 20000
python 04_make_arrays.py  --clean processed/clean.parquet --split processed/split_random.parquet --out processed/arrays_random_full --labels full --train-cap 20000
python 05_sanity_rf.py    processed/arrays_random_full.npz processed/arrays_time_full.npz
```

Протокол leave-one-family-out (скрыть выбранные атаки в train): `03_make_splits.py --protocol lofo --holdout "DoS Hulk" "DoS GoldenEye" --out ...`.

## Файлы

| Файл | Назначение |
|---|---|
| `01_clean.py` | чтение 8 CSV, очистка, один `clean.parquet` |
| `02_profile.py` | статистика: классы×день, окна времени, NaN, константы, дубликаты, конфликты, IP (`processed/profile/*.csv`) |
| `03_make_splits.py` | разбиения `random` / `time` / `lofo`, таблицы «класс × часть» (`split_*_counts.csv`) |
| `04_make_arrays.py` | предобработка по train, `X∈[0,1]`, метки, подвыборка train → `arrays_*.npz` + `.json` |
| `05_sanity_rf.py` | контрольный Random Forest по классам на выбранных массивах |
| `run_all.sh` | запуск всех шагов подряд |
| `requirements.txt` | зависимости |
| `processed/` | результаты прогона (большие `.parquet` и `.npz` в git не попадают) и документ `DATA_PREPARATION.md` |
