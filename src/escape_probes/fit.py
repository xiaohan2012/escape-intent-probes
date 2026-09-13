"""A regularised linear probe, on numpy.

Sixty lines rather than a scikit-learn dependency. The project's only numeric
requirement is numpy, `vllm` already pins a torch the `model` group does not
want, and a second resolver fight buys nothing here: this is logistic regression
with an L2 penalty, and the interesting decisions are all in `probe.py`.

Two properties the fit must have, both forced by the shape of the data rather
than by taste:

**The penalty dominates.** 5120 features against roughly 95 positives is p >> n
by a factor of fifty. An unregularised fit separates any such split perfectly
and reports a number about nothing. The penalty is therefore a swept parameter,
chosen on training folds, never a default left alone.

**Classes are weighted.** R2 yields about 95 positives against 888 negative
steps — 1:9. Unweighted, a probe that answers "never" scores 90% and has no
reason to look at its input.

Features are standardised on the training fold only. Fitting the scaler on
everything leaks the held-out fold's distribution, which at n=60 trajectories is
enough to move the answer.
"""

from __future__ import annotations

import numpy as np


def standardise(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale, returning statistics to apply to the held-out fold.

    A feature that is constant on the training fold gets a scale of 1 rather
    than 0. Dead residual dimensions are real at this sample size, and a single
    0/0 poisons the whole vector.
    """
    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return centre, scale


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """The rank statistic: P(score of a random positive > a random negative).

    Ties count a half, which matters at this sample size — a probe that has
    collapsed to a constant should report 0.5, not 1.0.
    """
    y = np.asarray(y)
    positives, negatives = scores[y == 1], scores[y == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("AUC needs both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties, which is what makes a constant score give 0.5.
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(values))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float(
        (ranks[y == 1].sum() - len(positives) * (len(positives) + 1) / 2)
        / (len(positives) * len(negatives))
    )


class LinearProbe:
    """Logistic regression by full-batch gradient descent.

    Deterministic — no shuffling, no random initialisation — so a reported number
    is reproducible without a seed threaded through every call site.
    """

    def __init__(
        self, l2: float = 1.0, balanced: bool = True, steps: int = 400, lr: float = 0.5
    ) -> None:
        self.l2 = l2
        self.balanced = balanced
        self.steps = steps
        self.lr = lr
        self.weights = np.zeros(0)
        self.bias = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearProbe:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(np.unique(y)) < 2:
            # A fold whose held-out instances contain no crossing. Skipping it
            # loudly beats reporting the 0.5 an all-one-class fit would give.
            raise ValueError("cannot fit on one class")

        weight = np.ones_like(y)
        if self.balanced:
            for value in (0.0, 1.0):
                weight[y == value] = len(y) / (2 * max((y == value).sum(), 1))

        self.weights = np.zeros(x.shape[1])
        self.bias = 0.0
        total = weight.sum()
        for _ in range(self.steps):
            error = _sigmoid(x @ self.weights + self.bias) - y
            scaled = weight * error
            self.weights -= self.lr * ((x.T @ scaled) / total + self.l2 * self.weights / len(y))
            self.bias -= self.lr * scaled.sum() / total
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(x, dtype=np.float64) @ self.weights + self.bias)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Split on sign so neither branch exponentiates a large positive number.
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


__all__ = ["LinearProbe", "roc_auc", "standardise"]
