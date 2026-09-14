# NewsGauge — Fake News Classification with BERT, From Scratch to Deployed Model

A research-to-deployment project for fake-news headline classification on the
[Fakeddit](https://github.com/entitize/Fakeddit) dataset: a classical TF-IDF baseline, a BERT-style
Transformer encoder implemented from scratch in PyTorch, and fine-tuned pretrained Transformers (BERT-base,
DistilBERT) — with multi-seed evaluation, a data-scaling ablation, error analysis, Integrated Gradients
token-attribution, calibrated confidence, and a public Hugging Face + Streamlit demo of the final model.

> Originally built for CS60075 (NLP), IIT Kharagpur, Spring 2026, and extended into NewsGauge, a full
> experimental + engineering project. The four-model comparison, multi-seed results, and data-scaling ablation
> below come directly from a completed, executed run —
> [`BERT_Fake_News_Full_Pipeline.ipynb`](BERT_Fake_News_Full_Pipeline.ipynb) (kept under its original filename
> as the historical source notebook). The **final production model** (Section "Final Production Model" below)
> was trained by [`notebooks/train_final_model.ipynb`](notebooks/train_final_model.ipynb) — see that section
> for the real, executed results.

## Overview

Fake-news detection is a text classification problem, but *how* you solve it says a lot about what you
understand: a bag-of-words baseline tells you whether lexical cues alone are enough; building a Transformer
encoder from scratch tells you whether you understand *why* attention-based models work, not just how to call
`.fit()`; and fine-tuning pretrained Transformers tells you how much large-scale pretraining actually buys you
over both. This project runs all three in the same experimental harness — same data split, same metrics, same
seeds — so the comparison is apples-to-apples, then takes the winner through calibration, packaging, and a
public demo:

**TF-IDF + Logistic Regression → BERT implemented from scratch → fine-tuned DistilBERT / BERT-base →
validation-based checkpoint selection + temperature-scaled calibration → Hugging Face Hub → Streamlit demo.**

## Audit & improvements (2026-09-11)

Before extending this project, it was audited end-to-end (full source, not just the README's claims). Two
findings changed what shipped:

1. **`src/data/preprocess.py` and `src/data/dataset.py` were missing from the committed repository.** They
   existed only inside `BERT_Fake_News_Full_Pipeline.ipynb`'s `%%writefile` cells and were never `git add`ed,
   which meant every documented CLI command (`train.py`, `scripts/run_*.sh`) and two of the 42 test files
   failed with `ModuleNotFoundError` as checked in. Restored — see `NEXT_STEPS.md` Section 3.
2. **No validation split existed anywhere** — only train/test. The test set was evaluated every epoch inside
   the training loop, and final numbers came from whichever epoch ran last, not a principled best-checkpoint
   choice. `src/training/train_transformer.py` now does a genuine train/validation/test split, selects the
   checkpoint by validation F1, and touches the test set exactly once. `src/calibration/temperature_scaling.py`
   (new) fits post-hoc temperature scaling on validation logits, since the demo shows users a confidence
   number and raw softmax confidence from fine-tuned BERT-style models is a well-documented overconfidence
   risk (Guo et al., 2017).

Everything below the "Key Results" section that predates this audit — the 4-model comparison, multi-seed
table, data-scaling ablation, architecture, and error analysis — was left as-is: it was already measured
correctly and its conclusions (e.g. "BERT-base and DistilBERT are statistically indistinguishable here") were
already appropriately cautious. See `NEXT_STEPS.md` for the full list.

## Key Results (original 4-model comparison)

Dataset: [Fakeddit](https://github.com/entitize/Fakeddit) (968,099 combined posts across its train/validate/test
splits). All four models were trained and evaluated on the **same balanced subsample**: 2,500 real + 2,500 fake
posts (5,000 total), split 80/20 → **4,000 train / 1,000 test**, stratified. This is a deliberate choice for
fast, controlled iteration across four models — not a data availability limit.

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| TF-IDF + Logistic Regression | 0.704 | 0.703 |
| BERT (from scratch, 2-layer/512-dim) | 0.702 | 0.702 |
| BERT-base (fine-tuned) | 0.799 | 0.799 |
| **DistilBERT (fine-tuned)** | **0.811** | **0.811** |

**Multi-seed results** (BERT-base and DistilBERT, 3 seeds each):

| Model | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---|
| BERT-base | 0.8023 ± 0.0104 | 0.8022 ± 0.0104 |
| DistilBERT | 0.8043 ± 0.0124 | 0.8042 ± 0.0124 |

## Experimental Findings

**Fine-tuned Transformers clearly beat both the classical baseline and the from-scratch model** (~+10 points
F1) — pretraining on large text corpora gives BERT-base/DistilBERT a big head start that neither TF-IDF nor a
randomly-initialized 2-layer Transformer can match with only 4,000 training examples.

**The from-scratch BERT did not beat the TF-IDF baseline on a single run** (0.702 vs 0.703 F1) — essentially a
tie. This is not an implementation bug (42 unit tests confirm the attention/embedding/masking logic is
correct), it's the expected outcome of training a randomly-initialized Transformer from scratch on only 4,000
examples: it overfits (train acc 0.80 by epoch 6) while test accuracy peaks earlier and drifts down slightly.

**BERT-base and DistilBERT are statistically indistinguishable on this task at this scale.** The 3-seed
results (0.8022 ± 0.0104 vs 0.8042 ± 0.0124) show the single-run gap (~0.002) is well inside one standard
deviation for both models — run-to-run noise, not a real difference. The honest conclusion is **not**
"DistilBERT is better," it's "the two are comparable here, and DistilBERT gets there with 40% fewer
parameters."

**Performance scales with training data with diminishing returns.** Fixing BERT-base and varying training set
size (seed 42):

| Training data used | Train size | Accuracy | F1 |
|---|---|---|---|
| 10% | 400 | 0.685 | 0.682 |
| 25% | 1,000 | 0.745 | 0.743 |
| 50% | 2,000 | 0.761 | 0.760 |
| 100% | 4,000 | 0.799 | 0.799 |

The biggest jump is 10%→25%; each subsequent doubling buys a smaller gain, consistent with the standard
fine-tuning data-efficiency curve for pretrained Transformers.

## Final Production Model

Trained and deployed 2026-09-11 (`notebooks/train_final_model.ipynb`, executed end to end). Full dataset:
968,099 pooled posts (484,632 fake / 483,467 real); balanced sample: 2,500/class (5,000 total); split 3,500
train / 500 validation / 1,000 test.

- **Selected model**: `distilbert-base-uncased`, seed 1 — bert-base and DistilBERT were statistically
  indistinguishable on this run's 3-seed comparison (Welch's t-test, t=0.886, p=0.4365), so DistilBERT's
  ~40% smaller size broke the tie.
- **All 6 runs** (not just the selected one):

  | Model | Seed | Test Acc | Test F1 | ECE raw→calibrated | T |
  |---|---|---|---|---|---|
  | bert-base-uncased | 42 | 0.8000 | 0.7998 | 0.0753→0.0588 | 1.108 |
  | bert-base-uncased | 1 | 0.8070 | 0.8070 | 0.0820→0.0662 | 1.120 |
  | bert-base-uncased | 7 | 0.7880 | 0.7879 | 0.0569→0.0473 | 1.045 |
  | distilbert-base-uncased | 42 | 0.7720 | 0.7695 | 0.0407→0.0423 | 0.957 |
  | **distilbert-base-uncased (selected)** | **1** | **0.7980** | **0.7975** | **0.0813→0.0656** | **1.110** |
  | distilbert-base-uncased | 7 | 0.7990 | 0.7985 | 0.0769→0.0648 | 1.098 |

- **Multi-seed summary**: bert-base 0.7983±0.0078 acc / 0.7982±0.0079 F1; distilbert 0.7897±0.0125 acc /
  0.7885±0.0134 F1.
- **Model on Hugging Face**: https://huggingface.co/abhinav-29/fakeddit-bert-fake-news (live — kept under its
  existing repo name; Hugging Face repo IDs aren't renamed automatically by a project rebrand. To rename it to
  match "NewsGauge," either use the "Rename" option on the model's Settings page on huggingface.co, or run
  `HfApi().move_repo(from_id="abhinav-29/fakeddit-bert-fake-news", to_id="your-username/newsgauge", repo_type="model")`)
- **Live demo**: deployed via Streamlit Community Cloud — see `NEXT_STEPS.md` Section 9. (Hugging Face Spaces'
  Gradio/Docker SDKs now require a paid plan even on free CPU hardware; Static Spaces are free but can't run
  a Python backend, so Streamlit Community Cloud is the free deployment path here.)
- **A real finding worth flagging, not hiding**: the mandatory save→reload→predict verification step found the
  deployed model predicts "real" with >95% confidence for an empty string and a garbage repeated-character
  string, and "fake" for a plain, unremarkable headline. See `MODEL_CARD.md` Limitations for the full note —
  the 79.8% test accuracy is in-distribution held-out-set performance, not a guarantee about arbitrary input.

## Scratch Transformer Architecture

Implemented in `src/models/` with PyTorch, matching the standard encoder-only BERT design
(`embed_dim=512, num_heads=8, num_layers=2, ff_dim=2048` by default, all configurable via CLI flags):

- **Token, positional, and segment embeddings** (`embeddings.py`) — three learned embedding tables summed and
  passed through LayerNorm + dropout, as in the original BERT paper (not sinusoidal positions).
- **Multi-head self-attention** (`attention.py`) — scaled dot-product attention computed per-head from
  learned Q/K/V projections, with an additive attention mask for padding.
- **Position-wise feed-forward layers** (`feedforward.py`) — two-layer MLP with GELU activation.
- **Transformer encoder blocks** (`encoder.py`) — post-LayerNorm residual connections, stacked `num_layers`
  times.
- **Pooler** (`encoder.py`) — extracts the `[CLS]` hidden state through a `Dense + Tanh` layer.
- **Classification head** (`bert.py`) — dropout + linear layer on the pooled output.

## Explainability

Token-level attribution via [Captum](https://captum.ai/)'s `LayerIntegratedGradients`
(`src/interpretability/integrated_gradients.py`), computing attribution scores at the word-embedding layer
against an all-zero baseline, with convergence delta tracked per example.

**This implementation is specific to the fine-tuned `bert-base-uncased`** checkpoint (accesses
`model.bert.embeddings.word_embeddings` directly) — it does not run against DistilBERT or the from-scratch
model without a small architecture-aware change.

| Text | Top attributed tokens |
|---|---|
| *"katy perry shares nude snap as she plans to vote naked"* (Fake) | `nude`, `she`, `shares`, `vote`, `naked` |
| *"black with noose around neck shortly before lynching mobile alabama"* (Real) | `neck`, `with`, `lynch` |

## Error Analysis

Misclassified test examples are pulled and heuristically bucketed by surface features
(`src/evaluation/error_analysis.py`). For the fine-tuned BERT-base model, of 10 sampled misclassified
examples: 7 fell into no obvious surface-level pattern ("other"), 2 were long titles, 1 was a very short
title — several genuinely look ambiguous even to a human reader without more context, suggesting a meaningful
chunk of the error rate comes from posts where the headline alone doesn't carry enough signal.

## Testing

```bash
python -m pytest tests/ -v
```

54 tests across 13 files (the original 42, now runnable now that `src/data/` is restored, plus 12 new ones
covering the validation split, temperature scaling/ECE, and model save/reload/inference — see
`tests/test_preprocess_val_split.py`, `tests/test_calibration.py`, `tests/test_model_loading.py`):

- **Attention / Embeddings / Encoder / scratch BertModel** (14 tests) — shapes, masking correctness
  (including that padding doesn't change the pooled `[CLS]` representation), determinism, finite parameters.
- **Training engine** (3 tests) — including a real gradient-descent check that loss decreases on a synthetic
  task.
- **Data pipeline** (14 tests) — cleaning, balancing, stratified two-way and three-way splitting, the
  train-fraction ablation, and explicit train/val/test **ID-leakage checks**.
- **Dataset / Metrics / Error analysis / Interpretability / Results aggregation** (17 tests) — tokenization,
  metric computation vs. hand-computed values, categorization logic, top-k attribution selection, multi-run
  summaries.
- **Calibration** (5 tests, new) — temperature scaling doesn't change argmax predictions, fits a positive
  temperature, and ECE is low for well-calibrated / high for badly-miscalibrated synthetic distributions.
- **Model loading** (9 tests, new) — `save_pretrained`/`from_pretrained` round-trip produces identical
  predictions, calibration.json round-trips, inference handles empty/whitespace/very-long/non-ASCII input
  without crashing, and predictions are deterministic in eval mode.

## Repo structure

```
newsgauge/
├── BERT_Fake_News_Full_Pipeline.ipynb   # original executed notebook — source of the 4-model comparison above
├── notebooks/
│   └── train_final_model.ipynb          # end-to-end final production training pipeline
├── src/
│   ├── models/            # scratch BERT: attention.py, feedforward.py, embeddings.py, encoder.py, bert.py
│   ├── data/               # preprocess.py (load/clean/balance/split, incl. 3-way split), dataset.py
│   ├── calibration/         # temperature_scaling.py (Guo et al. 2017 + ECE + reliability diagrams)
│   ├── training/           # train_baseline.py, train_scratch.py, train_transformer.py, engine.py
│   ├── evaluation/         # metrics.py, error_analysis.py, aggregate_results.py
│   └── interpretability/   # integrated_gradients.py
├── tests/                  # 54 unit tests, run offline with pytest, no GPU/data needed
├── configs/                 # YAML hyperparameter configs per model
├── scripts/                 # shell drivers: run_all_baselines.sh, run_multiseed.sh, run_data_ablation.sh
├── app/
│   ├── streamlit_app.py     # NewsGauge's Streamlit demo, replaces app/demo_app.py (Gradio)
│   ├── demo_app.py          # original Gradio demo (kept for reference, unchanged)
│   └── requirements.txt     # lightweight pinned deps for Streamlit Community Cloud
├── results/                 # figures + metrics, from both the original notebook and train_final_model.ipynb
├── MODEL_CARD.md            # Hugging Face model card (filled in from the 2026-09-11 training run)
├── requirements.txt          # research/training environment
└── NEXT_STEPS.md             # environment setup, data download, and exact reproduction/deploy commands
```

## Reproducibility

```bash
git clone <your-repo-url> && cd newsgauge
pip install -r requirements.txt
python -m pytest tests/ -v
```

Reproducing the trained-model results needs a GPU (Colab/Kaggle T4 is enough) and the Fakeddit dataset. All
seeds are fixed and passed explicitly (`--seed`, default 42; multi-seed run uses 42/1/7) via
`src/utils/seed.py`. Full step-by-step commands — including how to get the data, train the final model,
upload to Hugging Face, and deploy the Streamlit app — are in **[NEXT_STEPS.md](NEXT_STEPS.md)**.

## Limitations

- **Not a general-purpose fact-checker.** See `MODEL_CARD.md` for the full statement — this model learns
  headline-style correlates of Fakeddit's distant-supervision labels, not verified claim truth.
- **Non-standard data split.** The official Fakeddit train/validate/test files are pooled and re-split, so
  results here aren't directly comparable to papers using Fakeddit's own official test split.
- **Small balanced subsample.** All results use ~5,000 of Fakeddit's 1M+ posts, chosen for fast controlled
  comparison across models — not necessarily representative of results at full scale.
- **Integrated Gradients is BERT-base-only** and not wired into the live demo (too slow for a shared-CPU
  Streamlit deployment).
- **Calibration reduces average overconfidence, not per-prediction correctness** — see `MODEL_CARD.md`.

## License / academic integrity note

This repository builds on graded coursework for CS60075 at IIT Kharagpur. If you are a current student in
this or a similar course, do not submit any part of this repository as your own assignment work — that would
violate your course's academic integrity policy. Use it as a reference for how to extend a finished assignment
into a portfolio project.
