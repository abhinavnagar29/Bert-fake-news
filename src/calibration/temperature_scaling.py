"""Post-hoc confidence calibration via temperature scaling.

Implements the method from Guo, Pleiss, Sun & Weinberger, "On Calibration of
Modern Neural Networks" (ICML 2017, https://arxiv.org/abs/1706.04599): a
single scalar temperature T > 0 is fit, by minimizing negative log-likelihood
on a held-out validation set, to rescale logits before softmax:

    p_calibrated(y=i | x) = softmax(z(x) / T)_i

T is fit ONLY on the validation split, never on the test split. Accuracy is
unchanged by temperature scaling (it's a monotonic rescaling, so argmax is
identical) -- only the reported confidence changes. Guo et al. found
temperature scaling matches or beats more complex calibration methods
(vector/matrix scaling, histogram binning) on this kind of classification
task while being far less prone to overfitting on small validation sets --
relevant here since Fakeddit's validation split in this project is only a
few hundred examples.

This module was added because the existing demo (app/demo_app.py) reports
raw softmax confidence directly to end users with no calibration check at
all -- a known failure mode for fine-tuned transformers on small training
sets (Guo et al. 2017, Section 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Wraps a single learned scalar temperature applied to logits."""

    def __init__(self):
        super().__init__()
        # Start at T=1.0 (no-op) and optimize from there.
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """Fit T by minimizing NLL on (logits, labels). Both must come from
        the validation set only. Returns the fitted temperature."""
        logits = logits.detach()
        labels = labels.detach()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().clamp(min=1e-3).item())

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(logits), dim=-1)


@dataclass
class ECEResult:
    ece: float
    bin_accuracies: list
    bin_confidences: list
    bin_counts: list
    bin_edges: list


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> ECEResult:
    """Expected Calibration Error (Naeini et al. 2015), as used in Guo et al. 2017.

    probs: (N, n_classes) softmax probabilities (raw OR calibrated -- call
    this twice, once per set, to compare before/after).
    labels: (N,) integer class labels.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum().item())
        bin_counts.append(count)
        if count > 0:
            acc = accuracies[in_bin].float().mean().item()
            conf = confidences[in_bin].mean().item()
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / len(confidences)) * abs(acc - conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return ECEResult(
        ece=float(ece.item()),
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
        bin_edges=bin_edges.tolist(),
    )


def plot_reliability_diagram(result: ECEResult, title: str, ax=None):
    """Draws a standard reliability diagram (accuracy vs. confidence per bin)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    bin_centers = [(result.bin_edges[i] + result.bin_edges[i + 1]) / 2 for i in range(len(result.bin_accuracies))]
    ax.bar(bin_centers, result.bin_accuracies, width=1.0 / len(bin_centers), edgecolor="black", alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {result.ece:.4f}")
    ax.legend()
    return ax
