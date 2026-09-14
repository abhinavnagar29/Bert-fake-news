import torch

from src.calibration.temperature_scaling import TemperatureScaler, expected_calibration_error


def test_temperature_scaling_does_not_change_argmax():
    torch.manual_seed(0)
    logits = torch.randn(200, 2) * 3
    labels = (logits[:, 1] > logits[:, 0]).long()

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    raw_preds = logits.argmax(dim=-1)
    calibrated_preds = scaler.calibrated_probs(logits).argmax(dim=-1)
    assert torch.equal(raw_preds, calibrated_preds)


def test_temperature_scaling_fits_positive_temperature():
    torch.manual_seed(0)
    logits = torch.randn(200, 2) * 5  # exaggerated logits -> likely overconfident
    labels = torch.randint(0, 2, (200,))

    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)
    assert t > 0


def test_ece_zero_for_perfectly_calibrated_uniform_predictions():
    # 100 examples, model predicts 0.5/0.5 for every one, and is correct
    # exactly half the time -- this should be (close to) perfectly calibrated
    # in the 0.5 confidence bin.
    probs = torch.full((100, 2), 0.5)
    labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    result = expected_calibration_error(probs, labels, n_bins=10)
    assert result.ece < 0.05


def test_ece_high_for_overconfident_wrong_predictions():
    # Model is 99% confident and always wrong -> should show high ECE.
    probs = torch.tensor([[0.99, 0.01]] * 50 + [[0.01, 0.99]] * 50)
    labels = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(50, dtype=torch.long)])
    result = expected_calibration_error(probs, labels, n_bins=10)
    assert result.ece > 0.9


def test_ece_bin_counts_sum_to_n():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(123, 2), dim=-1)
    labels = torch.randint(0, 2, (123,))
    result = expected_calibration_error(probs, labels, n_bins=15)
    assert sum(result.bin_counts) == 123
