"""Tests for the linear probe's fit: regularised logistic regression on numpy.

Written rather than taken from scikit-learn because the project's only numeric
dependency is numpy, and adding a resolver fight for sixty lines of gradient
descent is a bad trade — `vllm` already pins a torch the `model` group does not
want, and that is one such fight too many.

The two things that must be right:

**Regularisation, and that it dominates.** 5120 features against roughly 95
positives is p >> n by a factor of fifty, so the fit is determined by the penalty
rather than by the data. An unregularised fit separates the classes perfectly on
any split and means nothing.

**Class weighting.** R2 gives about 95 positives against 888 negative steps —
1:9. Unweighted, predicting "never crosses" scores 90% and the probe has no
reason to look at the input.
"""

from __future__ import annotations

import numpy as np
import pytest

from escape_probes.fit import LinearProbe, roc_auc, standardise


class TestStandardise:
    def test_it_centres_and_scales_on_the_training_set(self) -> None:
        train = np.array([[1.0, 10.0], [3.0, 30.0]])
        centre, scale = standardise(train)
        assert centre == pytest.approx([2.0, 20.0])
        assert scale == pytest.approx([1.0, 10.0])

    def test_a_constant_feature_does_not_divide_by_zero(self) -> None:
        # A dead residual dimension is real: some layers have features that are
        # constant across a small sample, and 0/0 would poison the whole vector.
        _, scale = standardise(np.array([[5.0, 1.0], [5.0, 3.0]]))
        assert scale[0] == 1.0

    def test_the_test_set_is_transformed_with_the_training_statistics(self) -> None:
        # Fitting the scaler on everything leaks the test fold's distribution,
        # which at n=60 is enough to move the answer.
        train = np.array([[0.0], [2.0]])
        centre, scale = standardise(train)
        assert (np.array([[4.0]]) - centre) / scale == pytest.approx([[3.0]])


class TestLinearProbe:
    def separable(self, n: int = 200, dim: int = 5) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, dim))
        y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
        return x, y

    def test_it_learns_a_separable_direction(self) -> None:
        x, y = self.separable()
        probe = LinearProbe(l2=1.0).fit(x, y)
        assert roc_auc(y, probe.score(x)) > 0.95

    def test_it_finds_the_right_feature(self) -> None:
        x, y = self.separable()
        weights = LinearProbe(l2=1.0).fit(x, y).weights
        assert abs(weights[0]) > 3 * np.abs(weights[1:]).max()

    def test_stronger_regularisation_shrinks_the_weights(self) -> None:
        # The lever that makes p >> n survivable, so it has to actually bite.
        x, y = self.separable()
        weak = np.linalg.norm(LinearProbe(l2=0.1).fit(x, y).weights)
        strong = np.linalg.norm(LinearProbe(l2=100.0).fit(x, y).weights)
        assert strong < weak

    def test_class_weighting_rescues_a_rare_positive(self) -> None:
        # 1:9 is R2's actual ratio. Unweighted, "never" scores 90%.
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(2.0, 1.0, (20, 3)), rng.normal(0.0, 1.0, (180, 3))])
        y = np.array([1] * 20 + [0] * 180)
        recall = lambda p: (p.score(x)[:20] >= 0.5).mean()  # noqa: E731
        assert recall(LinearProbe(l2=1.0, balanced=True).fit(x, y)) > recall(
            LinearProbe(l2=1.0, balanced=False).fit(x, y)
        )

    def test_scores_are_probabilities(self) -> None:
        x, y = self.separable()
        scores = LinearProbe(l2=1.0).fit(x, y).score(x)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_one_class_is_refused(self) -> None:
        # A fold whose held-out instances happen to contain no crossing. Better
        # to skip the fold loudly than to report an AUC of 0.5 as a result.
        with pytest.raises(ValueError, match="one class"):
            LinearProbe(l2=1.0).fit(np.zeros((4, 3)), np.zeros(4, dtype=int))

    def test_it_is_deterministic(self) -> None:
        x, y = self.separable()
        first = LinearProbe(l2=1.0).fit(x, y).weights
        second = LinearProbe(l2=1.0).fit(x, y).weights
        assert first == pytest.approx(second)


class TestRocAuc:
    def test_a_perfect_ranking_is_one(self) -> None:
        assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_a_reversed_ranking_is_zero(self) -> None:
        assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0

    def test_ties_score_a_half(self) -> None:
        assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)

    def test_it_matches_the_rank_statistic(self) -> None:
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 50)
        scores = rng.normal(size=50)
        pairs = [(a, b) for a in scores[y == 1] for b in scores[y == 0]]
        expected = np.mean([(a > b) + 0.5 * (a == b) for a, b in pairs])
        assert roc_auc(y, scores) == pytest.approx(expected)
