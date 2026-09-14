import os

import pandas as pd
import pytest

from src.data.preprocess import balance, clean, load_and_combine, prepare_dataset_with_val, split_three_way


def _write_fake_tsv(path, n_fake, n_real, start_id=0):
    rows = []
    for i in range(n_fake):
        rows.append({"clean_title": f"shocking fake headline number {start_id + i}", "2_way_label": 1, "id": f"f{start_id+i}"})
    for i in range(n_real):
        rows.append({"clean_title": f"real reported news story number {start_id + i}", "2_way_label": 0, "id": f"r{start_id+i}"})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture
def tiny_data_dir(tmp_path):
    d = str(tmp_path)
    _write_fake_tsv(os.path.join(d, "all_train.tsv"), n_fake=100, n_real=100, start_id=0)
    _write_fake_tsv(os.path.join(d, "all_validate.tsv"), n_fake=40, n_real=40, start_id=1000)
    _write_fake_tsv(os.path.join(d, "all_test_public.tsv"), n_fake=40, n_real=40, start_id=2000)
    return d


def test_split_three_way_sizes(tiny_data_dir):
    raw = load_and_combine(tiny_data_dir)
    df = clean(raw)
    bal = balance(df, n_per_class=100, seed=42)  # 200 rows total
    train_df, val_df, test_df = split_three_way(bal, val_size=0.1, test_size=0.2, seed=42)
    assert len(test_df) == pytest.approx(0.2 * 200, abs=1)
    assert len(val_df) == pytest.approx(0.1 * 200, abs=2)
    assert len(train_df) + len(val_df) + len(test_df) == 200


def test_split_three_way_is_stratified(tiny_data_dir):
    raw = load_and_combine(tiny_data_dir)
    df = clean(raw)
    bal = balance(df, n_per_class=100, seed=42)
    train_df, val_df, test_df = split_three_way(bal, val_size=0.1, test_size=0.2, seed=42)
    for split_df in (train_df, val_df, test_df):
        n_fake = (split_df["label"] == 1).sum()
        n_real = (split_df["label"] == 0).sum()
        assert abs(n_fake - n_real) <= 2


def test_split_three_way_no_pairwise_leakage(tiny_data_dir):
    raw = load_and_combine(tiny_data_dir)
    df = clean(raw)
    bal = balance(df, n_per_class=100, seed=42)
    train_df, val_df, test_df = split_three_way(bal, val_size=0.1, test_size=0.2, seed=42)
    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert set(val_df.index).isdisjoint(set(test_df.index))


def test_split_three_way_rejects_invalid_sizes():
    df = pd.DataFrame({"clean_title": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError):
        split_three_way(df, val_size=0.6, test_size=0.5, seed=42)


def test_prepare_dataset_with_val_end_to_end(tiny_data_dir):
    train_df, val_df, test_df, stats = prepare_dataset_with_val(
        tiny_data_dir, n_per_class=100, val_size=0.1, test_size=0.2, seed=42
    )
    assert stats.balanced_total == 200
    assert stats.train_size + stats.val_size + stats.test_size == 200
    # test set is never touched by train_fraction sub-sampling
    train_df_25, val_df_25, test_df_25, stats_25 = prepare_dataset_with_val(
        tiny_data_dir, n_per_class=100, val_size=0.1, test_size=0.2, seed=42, train_fraction=0.25
    )
    assert stats_25.test_size == stats.test_size
    assert stats_25.val_size == stats.val_size
    assert stats_25.train_size < stats.train_size


def test_prepare_dataset_with_val_no_leakage_by_id(tiny_data_dir):
    train_df, val_df, test_df, _ = prepare_dataset_with_val(
        tiny_data_dir, n_per_class=100, val_size=0.1, test_size=0.2, seed=42
    )
    ids_train, ids_val, ids_test = set(train_df["id"]), set(val_df["id"]), set(test_df["id"])
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)
