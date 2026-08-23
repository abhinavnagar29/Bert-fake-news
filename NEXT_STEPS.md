# NEXT_STEPS.md — How to reproduce these results

**All experiments described in the README have already been run** — see
[`BERT_Fake_News_Full_Pipeline.ipynb`](BERT_Fake_News_Full_Pipeline.ipynb)
for the complete executed notebook with every cell's real output. This
document is for reproducing that run (e.g. re-verifying numbers, re-running
after a code change, or running with different hyperparameters), not a
list of pending work.

Total GPU time for the full run (baselines + multi-seed + ablation) on a
free Colab T4 was a few hours.

---

## Step 0 — Environment

Use **Google Colab** (free T4 GPU) or **Kaggle Notebooks** (free P100/T4).
Locally, any machine with a CUDA GPU and ~8GB VRAM works fine too.

```bash
git clone <your-repo-url> bert-fake-news
cd bert-fake-news
pip install -r requirements.txt
python -m pytest tests/ -v          # confirm 42 passed before touching data/GPU
```

## Step 1 — Get the Fakeddit data

You need `all_train.tsv`, `all_validate.tsv`, `all_test_public.tsv` (the
`all_samples`, non-multimodal folder). These files are **not** hosted
directly in the Fakeddit GitHub repo — the repo only holds code; the
actual data is distributed via Google Drive.

**Option A — Upload directly (what the notebook does):**
```python
import os
os.makedirs("data/fakeddit", exist_ok=True)
from google.colab import files
uploaded = files.upload()   # select all_train.tsv, all_validate.tsv, all_test_public.tsv
import shutil
for fname in uploaded:
    shutil.move(fname, os.path.join("data/fakeddit", fname))
```

**Option B — Official Fakeddit Google Drive (v2.0)**, linked from the
[Fakeddit GitHub README](https://github.com/entitize/Fakeddit):
[drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm).
Open it, navigate into the `all_samples` (non-multimodal) folder, download
the three TSVs, and upload them via Option A's cell.

Verify:
```bash
python -c "from src.data.preprocess import prepare_dataset; \
  train_df, test_df, stats = prepare_dataset('data/fakeddit', n_per_class=2500); \
  print(stats)"
```
Expected (matches the notebook run): `original_total=968099,
original_fake=484632, original_real=483467, balanced_total=5000,
train_size=4000, test_size=1000`.

## Step 2 — Baseline comparison

Commands exactly as run in the notebook (Step 4):
```bash
python -m src.training.train_baseline --data_dir data/fakeddit --out_dir results/baseline
python -m src.training.train_scratch  --data_dir data/fakeddit --out_dir results/scratch_bert
python -m src.training.train_transformer --data_dir data/fakeddit \
  --model_name distilbert-base-uncased --out_dir results/distilbert --save_model
python -m src.training.train_transformer --data_dir data/fakeddit \
  --model_name bert-base-uncased --out_dir results/bert_base --save_model

python -m src.evaluation.aggregate_results --results_dir results --out results/comparison_table.csv
```
(Or `bash scripts/run_all_baselines.sh data/fakeddit`.) This reproduces the
headline table in the README.

## Step 3 — Multi-seed reporting

```bash
bash scripts/run_multiseed.sh data/fakeddit
```
Runs BERT-base and DistilBERT across seeds 42/1/7. Reproduces the
mean ± std numbers in the README's multi-seed table.

## Step 4 — Data-scaling ablation

```bash
bash scripts/run_data_ablation.sh data/fakeddit
```
Trains BERT-base at 10%/25%/50%/100% of the training set (seed 42).
Reproduces the data-scaling curve in the README.

## Step 5 — Error analysis

`train_transformer.py` writes `error_examples.csv` in each fine-tuned
model's output directory automatically (10 misclassified examples with a
heuristic category). See `results/bert_base/error_examples.csv` after
Step 2, or the notebook's Step 7 output for the exact examples discussed
in the README.

## Step 6 — Integrated Gradients

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.interpretability.integrated_gradients import compute_ig, top_k_tokens

model = AutoModelForSequenceClassification.from_pretrained('results/bert_base/checkpoint')
tokenizer = AutoTokenizer.from_pretrained('results/bert_base/checkpoint')
model.eval()

text = "shocking discovery leaves scientists speechless"
result = compute_ig(text, model, tokenizer, target_class=1)
print(top_k_tokens(result, k=5))
print("convergence delta:", result["convergence_delta"])
```
Note: `compute_ig` is BERT-architecture-specific (see README's
Explainability section) — this snippet only works against the fine-tuned
`bert-base-uncased` checkpoint, not the DistilBERT one.

If `convergence_delta` is above ~0.05, raise `n_steps` (e.g. to 500) and
re-run.

## Step 7 — Demo

```bash
python app/demo_app.py --model_dir results/bert_base/checkpoint
```
Opens a local Gradio UI. Add `--share` for a temporary public link, or
deploy on [Hugging Face Spaces](https://huggingface.co/spaces) for a
permanent one.

> **Fixed:** running this as `python app/demo_app.py` (rather than as a
> module) previously failed with `ModuleNotFoundError: No module named
> 'src'` — this was caught in the actual Colab run captured in the
> notebook. Python puts the *script's own directory* (`app/`) on
> `sys.path`, not the repo root, when run this way. `demo_app.py` now
> inserts the repo root into `sys.path` at the top of the file, so the
> command above works as documented.

---

### If something breaks

- **`FileNotFoundError` in `prepare_dataset`** → you're missing one of the three TSVs in `data_dir`, or the path is wrong. Re-check Step 1.
- **CUDA out of memory** → lower `--batch_size` (e.g. to 16), or reduce `--max_len`.
- **IG `convergence_delta` stays high even at n_steps=500** → try a shorter headline, or use `internal_batch_size=1`.
- **`ModuleNotFoundError: No module named 'src'` when running `app/demo_app.py`** → this is fixed in the current version of the file (see Step 7 note); if you still see it, check you're running the version of `demo_app.py` in this repo, not an older copy.
- **Tests fail after you edit code** → run `python -m pytest tests/ -v` and read the specific assertion; the tests were passing on the exact code in this repo, so a failure means your edit changed behavior — check it's intentional.
