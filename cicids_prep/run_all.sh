#!/usr/bin/env bash
# Пример: bash run_all.sh data/TL data/processed      (TL = папка с 8 CSV из GeneratedLabelledFlows.zip)
set -e
SRC=${1:-data/TL}; OUT=${2:-data/processed}; mkdir -p $OUT
python 01_clean.py   --src $SRC --out $OUT/clean.parquet
python 02_profile.py --clean $OUT/clean.parquet --outdir $OUT/profile
for P in time random; do
  python 03_make_splits.py --clean $OUT/clean.parquet --protocol $P --out $OUT/split_$P.parquet
  python 04_make_arrays.py --clean $OUT/clean.parquet --split $OUT/split_$P.parquet --out $OUT/arrays_${P}_full --labels full --train-cap 20000
done
python 05_sanity_rf.py $OUT/arrays_random_full.npz $OUT/arrays_time_full.npz
# leave-one-family-out (пример: скрыть DoS Hulk и GoldenEye)
# python 03_make_splits.py --clean $OUT/clean.parquet --protocol lofo --holdout "DoS Hulk" "DoS GoldenEye" --out $OUT/split_lofo_dos.parquet
