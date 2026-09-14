import os

import pandas as pd
import pytest

from src.data.preprocess import balance, clean, load_and_combine, prepare_dataset, split


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
    _write_fake_tsv(os.path.join(d, "all_train.tsv"), n_fake=60, n_real=60, start_id=0)
    _write_fake_tsv(os.path.join(d, "all_validate.tsv"), n_fake=20, n_real=20, start_id=1000)
    _write_fake_tsv(os.path.join(d, "all_test_public.tsv"), n_fake=20, n_real=20, start_id=2000)
    return d


def test_load_and_combine_raises_on_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_combine(str(tmp_path))


def test_load_and_combine_row_count(tiny_data_dir):
    df = load_and_combine(tiny_data_dir)
    assert len(df) == 120 + 40 + 40


def test_clean_renames_and_drops_short_titles(tiny_data_dir):
    raw = load_and_combine(tiny_data_dir)
    # inject a too-short title to verify it gets dropped
    raw.loc[0, "clean_title"] = "hi"
    cleaned = clean(raw)
    assert "clean_title" in cleaned.columns and "label" in cleaned.columns
    assert (cleaned["clean_title"].str.len() > 3).all()
    assert len(cleaned) == len(raw) - 1


def test_balance_gives_equal_classes(tiny_data_dir):
    raw = load_and_combine(tiny_data_dir)
    df = clean(raw)
    bal = balance(df, n_per_class=50, seed=42)
    assert (bal["label"] == 1).sum() == 50
    assert (bal["label"] == 0).sum() == 50
    assert len(bal) == 100


def test_split_is_stratified(tiny_data_dir):
    raw = load_and_combine(tiny_data_dir)
    df = clean(raw)
    bal = balance(df, n_per_class=50, seed=42)
    train_df, test_df = split(bal, test_size=0.2, seed=42)
    assert len(train_df) == 80
    assert len(test_df) == 20
    # both splits should be ~balanced
    assert abs((train_df["label"] == 1).sum() - (train_df["label"] == 0).sum()) <= 1
    assert abs((test_df["label"] == 1).sum() - (test_df["label"] == 0).sum()) <= 1


def test_prepare_dataset_end_to_end(tiny_data_dir):
    train_df, test_df, stats = prepare_dataset(tiny_data_dir, n_per_class=50, seed=42)
    assert stats.balanced_total == 100
    assert stats.train_size == 80
    assert stats.test_size == 20
    assert set(train_df.columns) >= {"clean_title", "label"}


def test_prepare_dataset_train_fraction_ablation(tiny_data_dir):
    train_df_full, _, stats_full = prepare_dataset(tiny_data_dir, n_per_class=50, seed=42, train_fraction=1.0)
    train_df_half, _, stats_half = prepare_dataset(tiny_data_dir, n_per_class=50, seed=42, train_fraction=0.5)
    assert stats_half.train_size == pytest.approx(stats_full.train_size * 0.5, abs=1)
    # still stratified after sub-sampling
    assert abs((train_df_half["label"] == 1).sum() - (train_df_half["label"] == 0).sum()) <= 1


def test_no_train_test_leakage_by_id(tiny_data_dir):
    train_df, test_df, _ = prepare_dataset(tiny_data_dir, n_per_class=50, seed=42)
    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
