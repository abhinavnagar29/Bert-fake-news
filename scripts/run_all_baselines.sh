#!/usr/bin/env bash
# Trains every model in the comparison table with the default (seed=42, full data) setting.
# Run from the repo root, on a GPU runtime (Colab/Kaggle) for the two Transformer models.
set -e
DATA_DIR=${1:-data/fakeddit}

echo "== 1/4  TF-IDF + Logistic Regression baseline =="
python -m src.training.train_baseline --data_dir "$DATA_DIR" --out_dir results/baseline

echo "== 2/4  Scratch BERT (Task 1 architecture) =="
python -m src.training.train_scratch --data_dir "$DATA_DIR" --out_dir results/scratch_bert

echo "== 3/4  DistilBERT fine-tune =="
python -m src.training.train_transformer --data_dir "$DATA_DIR" --model_name distilbert-base-uncased --out_dir results/distilbert --save_model

echo "== 4/4  BERT-base fine-tune =="
python -m src.training.train_transformer --data_dir "$DATA_DIR" --model_name bert-base-uncased --out_dir results/bert_base --save_model

echo "== Aggregating comparison table =="
python -m src.evaluation.aggregate_results --results_dir results --out results/comparison_table.csv
