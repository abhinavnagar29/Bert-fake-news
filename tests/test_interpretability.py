from src.interpretability.integrated_gradients import top_k_tokens


def test_top_k_excludes_special_tokens_by_default():
    result = {
        "tokens": ["[CLS]", "shocking", "news", "today", "[SEP]"],
        "scores": [0.9, 0.5, 0.05, 0.3, 0.8],
    }
    top = top_k_tokens(result, k=2)
    tokens = [t for t, _ in top]
    assert "[CLS]" not in tokens
    assert "[SEP]" not in tokens
    assert tokens == ["shocking", "today"]  # ranked by |score|


def test_top_k_can_include_special_tokens():
    result = {
        "tokens": ["[CLS]", "shocking", "news"],
        "scores": [0.9, 0.5, 0.05],
    }
    top = top_k_tokens(result, k=1, exclude_special=False)
    assert top[0][0] == "[CLS]"


def test_top_k_respects_k():
    result = {
        "tokens": ["a", "b", "c", "d"],
        "scores": [0.1, 0.9, 0.4, 0.2],
    }
    assert len(top_k_tokens(result, k=2)) == 2
