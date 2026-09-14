#!/usr/bin/env bash
# Data-scaling ablation: trains BERT-base at 10% / 25% / 50% / 100% of the training set.
set -e
DATA_DIR=${1:-data/fakeddit}

for FRAC in 0.10 0.25 0.50 1.00; do
  echo "== bert-base-uncased  train_fraction=$FRAC =="
  python -m src.training.train_transformer --data_dir "$DATA_DIR" --model_name bert-base-uncased \
    --out_dir "results/ablation/bert_base_frac${FRAC}" --train_fraction $FRAC
done

python -m src.evaluation.aggregate_results --results_dir results/ablation --out results/ablation/comparison_table.csv
