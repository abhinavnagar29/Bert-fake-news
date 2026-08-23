"""Pull misclassified examples and bucket them by simple heuristic categories.

The categories are intentionally simple heuristics (length, punctuation,
lexical cues) meant as a *starting point* for manual inspection -- the
report should still eyeball the actual misclassified examples and describe
the error patterns in prose (see NEXT_STEPS.md Step 5).
"""
import pandas as pd

CLICKBAIT_CUES = {
    "shocking", "you won't believe", "wont believe", "unbelievable",
    "must see", "goes viral", "insane", "outrageous", "amazing",
}


def categorize_error(text: str) -> str:
    t = text.lower()
    n_words = len(t.split())
    if n_words <= 4:
        return "very_short_title"
    if "?" in t:
        return "question_headline"
    if any(cue in t for cue in CLICKBAIT_CUES):
        return "clickbait_language"
    if t.count("!") >= 1:
        return "exclamatory_headline"
    if n_words >= 20:
        return "long_title"
    return "other"


def build_error_report(
    test_df: pd.DataFrame,
    preds,
    labels,
    text_col: str = "clean_title",
    top_k: int = 10,
) -> pd.DataFrame:
    """Returns a DataFrame of misclassified rows with a heuristic category,
    sorted so the first `top_k` rows are good candidates to paste into the
    report as worked examples."""
    df = test_df.reset_index(drop=True).copy()
    df["true_label"] = labels
    df["pred_label"] = preds
    errors = df[df["true_label"] != df["pred_label"]].copy()
    errors["error_category"] = errors[text_col].apply(categorize_error)
    errors = errors.sample(frac=1, random_state=42).reset_index(drop=True)
    return errors.head(top_k) if top_k else errors


def category_counts(errors: pd.DataFrame) -> pd.Series:
    return errors["error_category"].value_counts()
