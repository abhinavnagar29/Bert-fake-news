"""Convenience entrypoint -- prints available commands.

This project is driven via `python -m src.training.<script>` (see README.md
and NEXT_STEPS.md), not a single train.py, because there are five distinct
things you can train (baseline, scratch-BERT, BERT-base, DistilBERT, plus
ablation/multiseed sweeps of the last two). Running `python train.py` just
prints a summary so you don't have to go hunting through the repo.
"""

COMMANDS = """
Available commands (run from the repo root):

  Setup
  -----
  pip install -r requirements.txt
  python -m pytest tests/ -v                     # run the full test suite (no data/GPU needed)

  Training
  --------
  python -m src.training.train_baseline       --data_dir data/fakeddit --out_dir results/baseline
  python -m src.training.train_scratch        --data_dir data/fakeddit --out_dir results/scratch_bert
  python -m src.training.train_transformer    --data_dir data/fakeddit --model_name distilbert-base-uncased --out_dir results/distilbert --save_model
  python -m src.training.train_transformer    --data_dir data/fakeddit --model_name bert-base-uncased       --out_dir results/bert_base   --save_model

  Or run everything in one go:
  bash scripts/run_all_baselines.sh data/fakeddit

  Experiments
  -----------
  bash scripts/run_multiseed.sh data/fakeddit      # 3 seeds x {bert-base, distilbert}
  bash scripts/run_data_ablation.sh data/fakeddit  # 10% / 25% / 50% / 100% of training data

  Aggregate results into one table
  ---------------------------------
  python -m src.evaluation.aggregate_results --results_dir results --out results/comparison_table.csv

  Demo
  ----
  python app/demo_app.py --model_dir results/bert_base/checkpoint

See NEXT_STEPS.md for the full walkthrough, including where to get the data.
"""

if __name__ == "__main__":
    print(COMMANDS)
