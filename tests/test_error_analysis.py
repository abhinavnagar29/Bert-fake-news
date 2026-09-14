import pandas as pd

from src.evaluation.error_analysis import build_error_report, categorize_error, category_counts


def test_categorize_error_buckets():
    assert categorize_error("ok") == "very_short_title"
    assert categorize_error("is this really true right now?") == "question_headline"
    assert categorize_error("shocking discovery rocks the internet") == "clickbait_language"
    assert categorize_error("amazing news everyone loves it today") == "clickbait_language"
    assert categorize_error("breaking news reported by officials today") == "other"


def test_build_error_report_only_returns_mismatches():
    df = pd.DataFrame({"clean_title": ["a real story here", "a fake story here", "true story reported"]})
    preds = [0, 0, 1]  # row 0 correct, row 1 wrong, row 2 wrong
    labels = [0, 1, 0]
    errors = build_error_report(df, preds, labels, top_k=None)
    assert len(errors) == 2
    assert set(errors["true_label"]) <= {0, 1}
    assert "error_category" in errors.columns


def test_build_error_report_respects_top_k():
    df = pd.DataFrame({"clean_title": [f"story number {i} reported today" for i in range(20)]})
    preds = [0] * 20
    labels = [1] * 20  # all wrong
    errors = build_error_report(df, preds, labels, top_k=5)
    assert len(errors) == 5


def test_category_counts_sums_to_total_errors():
    df = pd.DataFrame({"clean_title": ["short", "is this true", "wow amazing shocking headline today"]})
    preds = [1, 1, 1]
    labels = [0, 0, 0]
    errors = build_error_report(df, preds, labels, top_k=None)
    counts = category_counts(errors)
    assert counts.sum() == len(errors)
