#!/usr/bin/env bash
# Runs BERT-base and DistilBERT across 3 seeds each, for mean +/- std reporting.
set -e
DATA_DIR=${1:-data/fakeddit}

for SEED in 42 1 7; do
  echo "== bert-base-uncased  seed=$SEED =="
  python -m src.training.train_transformer --data_dir "$DATA_DIR" --model_name bert-base-uncased \
    --out_dir "results/multiseed/bert_base_seed${SEED}" --seed $SEED

  echo "== distilbert-base-uncased  seed=$SEED =="
  python -m src.training.train_transformer --data_dir "$DATA_DIR" --model_name distilbert-base-uncased \
    --out_dir "results/multiseed/distilbert_seed${SEED}" --seed $SEED
done

python -m src.evaluation.aggregate_results --results_dir results/multiseed --out results/multiseed/comparison_table.csv
