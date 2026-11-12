# CICIDS2017: подготовка данных для SCN / Random Forest

Код очищает CICIDS2017, строит варианты разбиения train/val/test, готовит массивы для обучения и содержит проверки перед обучением.

Описания (в `processed/`):
- `DATA_PREPARATION.md` — как обработаны данные, статистика классов, обоснование разбиения;
- `PRE_TRAINING_CHECKS.md` — проверки перед обучением: совпадающие векторы, пересечения по времени, ограничение на класс, Bot и SSH вне окон, Random Forest на 5 сидах.

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

Проверки перед обучением (команды целиком — в `processed/PRE_TRAINING_CHECKS.md` и `run_all.sh`):

```
python 06_check_overlaps.py   --clean processed/clean.parquet --processed processed --outdir processed/checks
python 07_bot_ssh_windows.py  --clean processed/clean.parquet --split processed/split_time.parquet --outdir processed/checks --tol-min 0 --exclude-out processed/checks/exclude_bot_ssh_tol0.parquet
python 08_rf_seeds.py         --clean processed/clean.parquet --protocol time --split processed/split_time.parquet --tag time_base --outdir processed/checks
```

Вариант с дедупликацией по признакам модели (результаты в отдельную папку): те же команды, но в `03_make_splits.py` (для `time` и `random`) добавить `--dedup-on model`, в `08_rf_seeds.py --protocol random` тоже `--dedup-on model`, а `--clean`, `--split`, `--out`, `--outdir` указывать на новую папку (например `data/processed-dedup-on`). Сравнение двух вариантов — в `processed/PRE_TRAINING_CHECKS.md` (§6).

Полезные флаги `03_make_splits.py`: `--protocol lofo --holdout "DoS Hulk" ...` (скрыть атаки в train), `--dedup-on model` (дубликаты и конфликты по признакам без `Destination Port`), `--exclude-flow-ids файл.parquet` (удалить потоки до разбиения).

## Файлы

| Файл | Назначение |
|---|---|
| `01_clean.py` | чтение 8 CSV, очистка, один `clean.parquet` |
| `02_profile.py` | статистика: классы×день, окна времени, NaN, константы, дубликаты, конфликты, IP (`processed/profile/*.csv`) |
| `03_make_splits.py` | разбиения `random` / `time` / `lofo`, фильтр дубликатов и конфликтов, таблицы «класс × часть» |
| `04_make_arrays.py` | предобработка по train, `X∈[0,1]`, метки, подвыборка train → `arrays_*.npz` + `.json` |
| `prep_common.py` | общие функции предобработки (используются 04, 06, 08) |
| `05_sanity_rf.py` | быстрый контрольный Random Forest по классам |
| `06_check_overlaps.py` | совпадающие векторы между выборками; пересечения временных интервалов в протоколе time |
| `07_bot_ssh_windows.py` | потоки Bot и SSH-Patator вне официальных окон атак; список для исключения |
| `08_rf_seeds.py` | Random Forest на нескольких сидах, метрики по классам, `--cap` для ограничения на класс |
| `run_all.sh` | запуск всех шагов подряд |
| `requirements.txt` | зависимости |
| `processed/` | результаты прогона (большие `.parquet` и `.npz` в git не попадают), документы, таблицы `profile/` и `checks/` |
