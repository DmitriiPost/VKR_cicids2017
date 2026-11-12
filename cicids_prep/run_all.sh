#!/usr/bin/env bash
# Пример: bash run_all.sh data/TL processed      (TL = папка с 8 CSV из GeneratedLabelledFlows.zip)
set -e
SRC=${1:-data/TL}; OUT=${2:-data/processed}; mkdir -p $OUT
python 01_clean.py   --src $SRC --out $OUT/clean.parquet
python 02_profile.py --clean $OUT/clean.parquet --outdir $OUT/profile
for P in time random; do
  python 03_make_splits.py --clean $OUT/clean.parquet --protocol $P --out $OUT/split_$P.parquet
  python 04_make_arrays.py --clean $OUT/clean.parquet --split $OUT/split_$P.parquet --out $OUT/arrays_${P}_full --labels full --train-cap 20000
done
python 05_sanity_rf.py $OUT/arrays_random_full.npz $OUT/arrays_time_full.npz

# --- проверки перед обучением (см. processed/PRE_TRAINING_CHECKS.md) ---
mkdir -p $OUT/checks
python 06_check_overlaps.py --clean $OUT/clean.parquet --processed $OUT --outdir $OUT/checks
python 07_bot_ssh_windows.py --clean $OUT/clean.parquet --split $OUT/split_time.parquet --outdir $OUT/checks \
       --tol-min 0 --exclude-out $OUT/checks/exclude_bot_ssh_tol0.parquet
python 08_rf_seeds.py --clean $OUT/clean.parquet --protocol time   --split $OUT/split_time.parquet --tag time_base   --outdir $OUT/checks
python 08_rf_seeds.py --clean $OUT/clean.parquet --protocol random --tag random_base --outdir $OUT/checks
# варианты: без подозрительных Bot/SSH и дедупликация по признакам модели (без Destination Port)
# python 03_make_splits.py --clean $OUT/clean.parquet --protocol time --exclude-flow-ids $OUT/checks/exclude_bot_ssh_tol0.parquet --out $OUT/split_time_excl_botssh.parquet
# python 03_make_splits.py --clean $OUT/clean.parquet --protocol time --dedup-on model --out $OUT/split_time_dedupmodel.parquet
# python 08_rf_seeds.py --clean $OUT/clean.parquet --protocol time --split $OUT/split_time_dedupmodel.parquet --tag time_dedupmodel --outdir $OUT/checks
# leave-one-family-out (пример: скрыть DoS Hulk и GoldenEye)
# python 03_make_splits.py --clean $OUT/clean.parquet --protocol lofo --holdout "DoS Hulk" "DoS GoldenEye" --out $OUT/split_lofo_dos.parquet
