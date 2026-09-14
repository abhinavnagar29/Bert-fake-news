# Next Steps — Environment Setup & Exact Reproduction Commands (NewsGauge)

This file was referenced twice by the README ("see NEXT_STEPS.md for where to get the data" / "exact commands
to reproduce every result") but did not exist in the repository. It's restored here.

## 1. Environment setup

```bash
git clone <this-repo-url>
cd newsgauge
python -m venv .venv && source .venv/bin/activate   # or conda equivalent
pip install -r requirements.txt
```

A GPU is strongly recommended for Sections 4+ (BERT/DistilBERT fine-tuning). A free-tier T4 on Google Colab
or Kaggle is sufficient — the from-scratch model and baseline experiments run fine on CPU.

## 2. Getting the Fakeddit data

Fakeddit is not redistributed in this repository (it's a large, separately-licensed dataset). Download the
three official 2-way-label TSVs from the dataset authors' repository:

1. Go to https://github.com/entitize/Fakeddit and follow the "Download" link (the dataset is hosted via a
   shared Google Drive / direct-download link maintained by the authors; the exact link occasionally moves,
   so use whatever the repo's current README points to).
2. You need: `all_train.tsv`, `all_validate.tsv`, `all_test_public.tsv`.
3. Place all three in a local folder, e.g. `data/fakeddit/`.
4. Point `--data_dir data/fakeddit` (scripts) or `CONFIG["data_dir"]` (notebook) at that folder.

`src/data/preprocess.py::load_and_combine` raises a clear `FileNotFoundError` naming exactly which file is
missing if this step isn't done correctly — you don't need to guess.

## 3. Restoring `src/data/` and `src/calibration/` (one-time, if not already present)

**Audit note**: in the version of this repository audited on 2026-09-11, `src/data/preprocess.py` and
`src/data/dataset.py` existed only inside `BERT_Fake_News_Full_Pipeline.ipynb` (written via a `%%writefile`
cell when the notebook ran in Colab) and were never committed — which meant every command below failed with
`ModuleNotFoundError` as checked into git. This has been fixed: `src/data/`, `src/calibration/`, and the
updated `src/training/engine.py` / `src/training/train_transformer.py` / `src/evaluation/error_analysis.py`
are included directly in this delivery. If you're reading this after re-running
`notebooks/train_final_model.ipynb` (which also writes these files via the same `%%writefile` convention,
Section 4), just make sure to:

```bash
git add src/data src/calibration src/training/engine.py src/training/train_transformer.py src/evaluation/error_analysis.py
git commit -m "Ensure src/data and src/calibration are committed"
```

before pushing, so this doesn't regress.

## 4. Run the test suite

```bash
python -m pytest tests/ -v
```

Two test files need network access to Hugging Face Hub for a ~1MB tiny test checkpoint
(`tests/test_model_loading.py`, using `hf-internal-testing/tiny-random-bert`) — they auto-skip with a clear
message if that's unavailable, rather than failing the whole run.

## 5. Reproduce the original baseline / scratch-model / multi-seed / ablation experiments

These CLI scripts are unchanged from the original methodology (only `train_transformer.py` itself was
updated — see Section 6 below):

```bash
bash scripts/run_all_baselines.sh
bash scripts/run_multiseed.sh
bash scripts/run_data_ablation.sh
```

## 6. Train the final production model — ✅ already done (2026-09-11)

`notebooks/train_final_model.ipynb` was run end-to-end. Real results: `distilbert-base-uncased` (seed 1)
selected, test accuracy 0.7980, test F1 0.7975, ECE 0.0813→0.0656 after calibration. Full per-seed table is in
`MODEL_CARD.md` and the README's "Final Production Model" section. To retrain from scratch (e.g. with
different seeds/data), the same two options still work:

**Option A — the complete notebook:**

```bash
jupyter notebook notebooks/train_final_model.ipynb
# or upload to Google Colab / Kaggle and run top-to-bottom
```

**Option B — CLI, for a single (model, seed) run:**

```bash
python -m src.training.train_transformer \
    --data_dir data/fakeddit \
    --model_name distilbert-base-uncased \
    --out_dir results/final_model \
    --seed 42 \
    --save_model
```

## 7. Upload the model to Hugging Face Hub — ✅ already done

Live at **https://huggingface.co/abhinav-29/fakeddit-bert-fake-news**. This was done via
`notebooks/train_final_model.ipynb` Section 14, which also generates the model card from the run's actual
measured numbers (no placeholders) and uploads it as the repo's `README.md`. To repeat this for a different
run:

```bash
pip install huggingface_hub
huggingface-cli login   # paste a token with "write" scope from https://huggingface.co/settings/tokens

python - <<'PY'
from huggingface_hub import HfApi, create_repo

repo_id = "your-username/newsgauge"   # choose your own
create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path="results/final_model/final_checkpoint",
    repo_id=repo_id,
    repo_type="model",
)
PY
```

## 8. Deploy the Streamlit frontend

`app/streamlit_app.py` already has `MODEL_ID = "abhinav-29/fakeddit-bert-fake-news"` set — no edits needed.

**Important finding (2026-09-11)**: Hugging Face now requires a **paid plan** to create a new Gradio or
Docker Space, even on free `cpu-basic` hardware — only Static Spaces (no Python backend) are free to create.
So the automated path in the notebook's Section 15 (which tries to create a Docker-SDK Space) will hit a
`402 Payment Required` error on a free HF account. Two real options:

**Option A — Streamlit Community Cloud (free, ~2 minutes of manual setup — this is what was actually used):**

1. Push this repository to GitHub (public, or a private repo Streamlit Cloud has access to).
2. Go to https://share.streamlit.io → "New app".
3. Repository: this repo. Branch: `main`. Main file path: `app/streamlit_app.py`.
4. Under "Advanced settings", set **Requirements file path** to `app/requirements.txt` (not the root
   `requirements.txt`, which pulls in the full training stack — captum, matplotlib, datasets — that the
   deployed app doesn't need).
5. No secrets are required (the model loads from a public Hugging Face repo via `from_pretrained`).
6. Deploy. First load is slower (downloading model weights from the Hub); subsequent interactions are fast
   because `st.cache_resource` keeps the model in memory.

**Option B — Hugging Face Space with the Docker SDK (needs HF PRO, ~$9/month):**

Once subscribed, `notebooks/train_final_model.ipynb` Section 15 already does this automatically (creates a
`Dockerfile` that runs `streamlit run ... --server.port=8501`, an `app_port: 8501` / `sdk: docker` Space
`README.md`, and pushes both plus `requirements.txt` and `streamlit_app.py`) — no changes needed, just re-run
that cell once the account has PRO.

Test locally either way before deploying:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```
