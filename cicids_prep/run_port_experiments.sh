#!/usr/bin/env bash
# Эксперименты с Destination Port (см. processed/PORT_EXPERIMENTS.md).
# Пример: bash run_port_experiments.sh processed processed-dedup-on
#   $1 - папка с базовым прогоном (clean.parquet, split_time.parquet), $2 - папка с прогоном --dedup-on model
# Результаты: <папка>/checks/ports_*.csv, затем сводка 10_summarize_ports.py.
set -e
BASE=${1:-processed}; DEDUP=${2:-processed-dedup-on}
run() { # папка набор exp use_port cap trees seeds...
  DIR=$1; DS=$2; EXP=$3; UP=$4; CAP=$5; TR=$6; shift 6
  W=$([ "$UP" = "1" ] && echo with || echo no)
  mkdir -p $DIR/checks
  python 09_port_experiments.py --clean $DIR/clean.parquet --split $DIR/split_time.parquet --exp $EXP --use-port $UP \
      --cap $CAP --trees $TR --seeds "$@" --tag ${DS}_${EXP}_${W}port_cap${CAP}_t${TR} --outdir $DIR/checks
}
# 1) основные таблицы: ограничение 20 000, 60 деревьев, 5 сидов, оба набора
for EXP in port portsplit; do for UP in 1 0; do
  run $BASE  base  $EXP $UP 20000 60 0 1 2 3 4
  run $DEDUP dedup $EXP $UP 20000 60 0 1 2 3 4
done; done
# 2) сравнение «с ограничением 20 000» и «без ограничения» (базовый набор, 30 деревьев, 3 сида; без ограничения долго и тяжело по памяти)
for EXP in port portsplit; do for UP in 1 0; do
  run $BASE base $EXP $UP 20000 30 0 1 2
  run $BASE base $EXP $UP 0 30 0 1 2
done; done
python 10_summarize_ports.py --checks $BASE/checks
python 10_summarize_ports.py --checks $DEDUP/checks
