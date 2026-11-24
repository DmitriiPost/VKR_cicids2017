# CICIDS2017: подготовка данных для SCN / Random Forest

Код очищает CICIDS2017, строит варианты разбиения train/val/test (A — случайное, C — stratified temporal split, F — leave-one-family-out), готовит массивы для обучения и содержит проверки перед обучением.

Описания (в `data/processed/`):
- `DATA_PREPARATION.md` — как обработаны данные, статистика классов, обоснование разбиения;
- `PRE_TRAINING_CHECKS.md` — проверки перед обучением: совпадающие векторы, пересечения по времени, ограничение на класс, Bot и SSH вне окон, Random Forest на 5 сидах.
- `PORT_EXPERIMENTS.md` — эксперименты с `Destination Port`: модель с портом и без порта; разбиение PortScan по непересекающимся наборам портов; сравнение ограничения 20 000 на класс и варианта без ограничения.

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
bash run_all.sh data/TL data/processed
```

или по шагам:

```
python 01_clean.py        --src data/TL --out data/processed/clean.parquet
python 02_profile.py      --clean data/processed/clean.parquet --outdir data/processed/profile
python 03_make_splits.py  --clean data/processed/clean.parquet --protocol time   --out data/processed/split_time.parquet
python 03_make_splits.py  --clean data/processed/clean.parquet --protocol random --out data/processed/split_random.parquet
python 04_make_arrays.py  --clean data/processed/clean.parquet --split data/processed/split_time.parquet   --out data/processed/arrays_time_full   --labels full --train-cap 20000
python 04_make_arrays.py  --clean data/processed/clean.parquet --split data/processed/split_random.parquet --out data/processed/arrays_random_full --labels full --train-cap 20000
python 05_sanity_rf.py    data/processed/arrays_random_full.npz data/processed/arrays_time_full.npz
```

Проверки перед обучением (команды целиком — в `data/processed/PRE_TRAINING_CHECKS.md` и `run_all.sh`):

```
python 06_check_overlaps.py   --clean data/processed/clean.parquet --processed data/processed --outdir data/processed/checks
python 07_bot_ssh_windows.py  --clean data/processed/clean.parquet --split data/processed/split_time.parquet --outdir data/processed/checks --tol-min 0 --exclude-out data/processed/checks/exclude_bot_ssh_tol0.parquet
python 08_rf_seeds.py         --clean data/processed/clean.parquet --protocol time --split data/processed/split_time.parquet --tag time_base --outdir data/processed/checks
```

Вариант с дедупликацией по признакам модели (результаты в отдельную папку): те же команды, но в `03_make_splits.py` (для `time` и `random`) добавить `--dedup-on model`, в `08_rf_seeds.py --protocol random` тоже `--dedup-on model`, а `--clean`, `--split`, `--out`, `--outdir` указывать на новую папку (например `data/processed-dedup-on`). Сравнение двух вариантов — в `data/processed/PRE_TRAINING_CHECKS.md` (§6).

Эксперименты с `Destination Port` (нужны результаты базового прогона и прогона с `--dedup-on model`; полное описание в `data/processed/PORT_EXPERIMENTS.md`):

```
bash run_port_experiments.sh data/processed data/processed-dedup-on
```

или по одной конфигурации (`--exp port` — протокол C с портом/без порта, `--exp portsplit` — PortScan по наборам портов; `--use-port 1/0`; `--cap 0` — без ограничения на класс):

```
python 09_port_experiments.py --clean data/processed/clean.parquet --split data/processed/split_time.parquet --exp portsplit --use-port 0 --cap 20000 --trees 60 --tag base_portsplit_noport_cap20000_t60 --outdir data/processed/checks
python 10_summarize_ports.py --checks data/processed/checks
```

Полезные флаги `03_make_splits.py`: `--protocol lofo --holdout "DoS Hulk" ...` (скрыть атаки в train), `--dedup-on model` (дубликаты и конфликты по признакам без `Destination Port`), `--exclude-flow-ids файл.parquet` (удалить потоки до разбиения).

## Файлы

| Файл | Назначение |
|---|---|
| `01_clean.py` | чтение 8 CSV, очистка, один `clean.parquet` |
| `02_profile.py` | статистика: классы×день, окна времени, NaN, константы, дубликаты, конфликты, IP (`data/processed/profile/*.csv`) |
| `03_make_splits.py` | разбиения `random` / `time` / `lofo`, фильтр дубликатов и конфликтов, таблицы «класс × часть» |
| `04_make_arrays.py` | предобработка по train, `X∈[0,1]`, метки, подвыборка train → `arrays_*.npz` + `.json` |
| `prep_common.py` | общие функции предобработки (используются 04, 06, 08) |
| `05_sanity_rf.py` | быстрый контрольный Random Forest по классам |
| `06_check_overlaps.py` | совпадающие векторы между выборками; пересечения временных интервалов в протоколе time |
| `07_bot_ssh_windows.py` | потоки Bot и SSH-Patator вне официальных окон атак; список для исключения |
| `08_rf_seeds.py` | Random Forest на нескольких сидах, метрики по классам, `--cap` для ограничения на класс |
| `09_port_experiments.py` | Random Forest с `Destination Port` и без него; разбиение PortScan по непересекающимся наборам портов; число совпадающих векторов между выборками |
| `10_summarize_ports.py` | сводные таблицы по результатам `09_port_experiments.py` (`ports_summary.csv`) |
| `run_port_experiments.sh` | запуск всех экспериментов с портом |
| `run_all.sh` | запуск всех шагов подряд |
| `requirements.txt` | зависимости |
| `data/processed/` | результаты прогона (большие `.parquet` и `.npz` в git не попадают), документы, таблицы `profile/` и `checks/` |
