"""Gradio demo: paste a headline, get a prediction + confidence + IG token
attributions rendered as a highlighted-text output.

Run (after Task 3 fine-tuning has produced a checkpoint):
    python app/demo_app.py --model_dir results/bert_base/checkpoint

If --model_dir is omitted, falls back to a fresh bert-base-uncased with a
random classification head purely so the UI is inspectable end-to-end before
you've trained anything -- predictions will be meaningless until you point it
at a real checkpoint.
"""
import argparse
import os
import sys

# Running this file directly (`python app/demo_app.py`) puts this file's own
# directory (app/) on sys.path, not the repo root -- so `from src...` below
# fails with "No module named 'src'" unless the repo root is added explicitly.
# (Observed in practice when launching from Colab; `python -m src...` CLIs
# elsewhere in this repo don't need this since `-m` adds the cwd instead.)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.interpretability.integrated_gradients import compute_ig

LABELS = {0: "Real", 1: "Fake"}


def build_predict_fn(model, tokenizer, device, use_ig: bool):
    def predict(headline: str):
        if not headline or not headline.strip():
            return "Enter a headline above.", None

        enc = tokenizer(headline, truncation=True, max_length=64, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1)[0]
        pred_class = int(torch.argmax(probs).item())
        confidence = float(probs[pred_class].item())

        summary = f"**Prediction: {LABELS[pred_class]}**  (confidence: {confidence:.1%})"

        highlighted = None
        if use_ig:
            try:
                result = compute_ig(
                    headline, model, tokenizer, target_class=pred_class,
                    device=device, n_steps=100, internal_batch_size=2,
                )
                pairs = [
                    (tok.replace("##", ""), score)
                    for tok, score in zip(result["tokens"], result["scores"])
                    if tok not in {"[CLS]", "[SEP]", "[PAD]"}
                ]
                highlighted = pairs
            except Exception as e:  # noqa: BLE001 -- surface IG errors in the UI, don't crash the app
                summary += f"\n\n_(Token attribution unavailable: {e})_"

        return summary, highlighted

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=None, help="Path to a save_pretrained() checkpoint from Task 3.")
    ap.add_argument("--base_model", type=str, default="bert-base-uncased", help="Fallback base model if --model_dir is not given.")
    ap.add_argument("--no_ig", action="store_true", help="Disable Integrated Gradients token highlighting (faster).")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = args.model_dir or args.base_model
    print(f"Loading model from: {source}")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2).to(device)
    model.eval()

    predict = build_predict_fn(model, tokenizer, device, use_ig=not args.no_ig)

    with gr.Blocks(title="Fake News Classifier -- BERT") as demo:
        gr.Markdown(
            "# Fake News Headline Classifier\n"
            "Fine-tuned BERT on the Fakeddit dataset, with Integrated Gradients "
            "token attribution showing which words drove the prediction."
        )
        headline_in = gr.Textbox(label="Reddit-style headline", placeholder="e.g. Scientists discover new species of deep-sea fish")
        btn = gr.Button("Classify", variant="primary")
        pred_out = gr.Markdown()
        attrib_out = gr.HighlightedText(label="Token attribution (darker = more important)")

        btn.click(fn=predict, inputs=headline_in, outputs=[pred_out, attrib_out])
        headline_in.submit(fn=predict, inputs=headline_in, outputs=[pred_out, attrib_out])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
