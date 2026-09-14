"""NewsGauge — Streamlit demo for the Fakeddit fake-news BERT/DistilBERT classifier.

Replaces the previous Gradio demo (app/demo_app.py). Loads the final model
directly from the Hugging Face Hub (set MODEL_ID below after Section 8 of
notebooks/train_final_model.ipynb / the model-upload steps in NEXT_STEPS.md),
caches it once per process with st.cache_resource (not st.cache_data --
a loaded model is a resource, not serializable data; see Streamlit's own
caching docs: https://docs.streamlit.io/develop/concepts/architecture/caching),
and applies the temperature-scaled calibration saved alongside the model so
the confidence shown to users isn't raw, likely-overconfident softmax output.
"""
import json

import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration -----------------------------------------------------------

MODEL_ID = "abhinav-29/fakeddit-bert-fake-news"
MAX_LEN = 64
EXAMPLES = [
    "Local city council approves new budget for public libraries",
    "You won't believe what this celebrity did - doctors are FURIOUS!!!",
    "Study finds moderate coffee consumption linked to lower mortality risk",
    "Scientists confirm the moon is actually a hologram, government admits",
]

# --- Model loading (cached once per process, not per interaction) ------------

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_tokenizer(model_id: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    try:
        cal_path = hf_hub_download(repo_id=model_id, filename="calibration.json")
        with open(cal_path) as f:
            calibration = json.load(f)
    except Exception:
        # Falls back to an uncalibrated temperature of 1.0 (= raw softmax) if
        # calibration.json isn't present in the Hub repo for some reason,
        # rather than crashing the whole app.
        calibration = {"temperature": 1.0, "id2label": {"0": "real", "1": "fake"}}
    return model, tokenizer, calibration


def predict(text: str, model, tokenizer, calibration: dict) -> dict:
    enc = tokenizer(text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    temperature = float(calibration.get("temperature", 1.0))
    probs = F.softmax(logits / temperature, dim=-1)[0]
    pred_idx = int(probs.argmax())
    id2label = calibration.get("id2label", {"0": "real", "1": "fake"})
    return {
        "label": id2label[str(pred_idx)],
        "confidence": float(probs[pred_idx]),
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
    }


# --- Page ----------------------------------------------------------------

st.set_page_config(page_title="NewsGauge", page_icon="📰", layout="centered")
st.title("📰 NewsGauge")
st.caption("BERT/DistilBERT fine-tuned on a balanced Fakeddit subset — 2-way (real/fake) headline classification.")

with st.expander("⚠️ Important limitations — read before trusting a prediction", expanded=False):
    st.markdown(
        """
- **This is not a general-purpose fact-checker.** It was trained only on Reddit post *titles* from the
  [Fakeddit](https://fakeddit.netlify.app/) dataset, using distant-supervision labels from the subreddit a
  post came from — not a human fact-check of each claim.
- It judges the **style and phrasing of a headline**, not the truth of the underlying claim. A real,
  accurately-reported headline that happens to sound sensational may still be flagged as "fake," and a
  fabricated claim written in a sober, neutral style may be flagged as "real."
- The reported confidence has been calibrated with temperature scaling (see the model card), but calibration
  reduces *average* overconfidence — it does not guarantee any single prediction is correct.
- It has not been evaluated on news from outside Reddit, on claims post-dating its training data, or on
  languages other than English.
- **Observed in testing**: this specific model predicted "real" with >95% confidence for an empty input and
  for a nonsense repeated-character string, and "fake" for a plain, unremarkable headline. Confidence on
  inputs unlike anything in training/validation data should not be trusted at face value.
        """
    )

model, tokenizer, calibration = load_model_and_tokenizer(MODEL_ID)

st.subheader("Try it")
example_choice = st.selectbox("Example headlines (or type your own below)", ["(custom)"] + EXAMPLES)
default_text = "" if example_choice == "(custom)" else example_choice
text = st.text_area("Headline text", value=default_text, height=100, max_chars=2000)

col1, col2 = st.columns([1, 3])
run = col1.button("Classify", type="primary", use_container_width=True)

if run:
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        result = predict(text, model, tokenizer, calibration)
        label = result["label"]
        conf = result["confidence"]

        if label == "fake":
            st.error(f"**Predicted: FAKE** — {conf:.1%} confidence")
        else:
            st.success(f"**Predicted: REAL** — {conf:.1%} confidence")

        st.progress(result["prob_fake"], text=f"P(fake) = {result['prob_fake']:.1%}")

        st.markdown(
            f"This means the model's calibrated estimate is that a headline phrased this way resembles "
            f"**{label}** Reddit posts {conf:.1%} of the time, based on patterns in its training data — "
            f"not a verified fact-check of this specific claim."
        )

        with st.expander("Token-level explanation (Integrated Gradients)"):
            st.info(
                "Integrated Gradients attribution (see src/interpretability/integrated_gradients.py) is "
                "currently implemented for bert-base-uncased only, run offline as part of the research "
                "notebook — it is not wired into this live app because running it per-request would make "
                "the app too slow for Streamlit Community Cloud's shared CPU tier. See the model card's "
                "'Explainability' section for saved examples."
            )

st.divider()
with st.expander("Model & evaluation details"):
    st.markdown(
        f"""
- **Model**: `{MODEL_ID}`
- **Task**: 2-way classification (real vs. fake), Fakeddit dataset
- **Calibration**: temperature scaling, T = {calibration.get('temperature', 1.0):.3f}
  (fit on a held-out validation split; see the model card for before/after ECE)
- Full training methodology, dataset details, and measured results: see the model card on the Hugging Face
  Hub page for `{MODEL_ID}`, and `notebooks/train_final_model.ipynb` in the source repository.
        """
    )
