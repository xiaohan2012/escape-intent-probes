"""Exact binomial intervals, for reading a zero honestly.

The whole descent turns on the difference between "did not cross" and "cannot
cross", and at the sample sizes we can afford that difference is entirely a
matter of the interval. `glm-5.3-flash` crossed 0 of 3 in the screen and 12 of
24 here; the 0/3 was never evidence of a floor, and the exact interval said so
at the time — [0.00, 0.71].

Clopper–Pearson rather than a normal approximation, because the approximation is
worst exactly where every interesting cell sits: k=0, k=n, and small n. Wald on
0/24 returns [0, 0], which is not a bound but a claim.

No scipy. The interval is defined by two monotone binomial tail probabilities,
so a bisection on `math.comb` gives it to machine precision with nothing
installed — and the core package deliberately has no optional dependency groups
for analysis (D13).
"""

from __future__ import annotations

from math import comb


def _at_most(successes: int, trials: int, probability: float) -> float:
    """P(X <= successes) for X ~ Binomial(trials, probability)."""
    return sum(
        comb(trials, i) * probability**i * (1.0 - probability) ** (trials - i)
        for i in range(successes + 1)
    )


def _at_least(successes: int, trials: int, probability: float) -> float:
    """P(X >= successes) for X ~ Binomial(trials, probability)."""
    return 1.0 - _at_most(successes - 1, trials, probability) if successes else 1.0


def _solve(target: float, decreasing: bool, evaluate) -> float:  # noqa: ANN001
    """Bisect a monotone function on [0, 1] for the point where it hits `target`."""
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        above = evaluate(middle) > target
        if above is decreasing:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """The exact two-sided interval for a binomial proportion.

    `successes = 0` gives a lower bound of exactly 0 and `successes = trials` an
    upper bound of exactly 1, by construction rather than by numerical accident:
    those are the cases the descent's conclusions rest on.

    >>> [round(x, 3) for x in clopper_pearson(0, 3)]
    [0.0, 0.708]
    >>> [round(x, 3) for x in clopper_pearson(0, 24)]
    [0.0, 0.142]
    """
    if not 0 <= successes <= trials:
        raise ValueError(f"{successes} successes out of {trials} trials")
    if trials == 0:
        raise ValueError("no trials")

    tail = (1.0 - confidence) / 2
    # P(X >= k | p) increases in p, so the lower bound is where it reaches the
    # tail; P(X <= k | p) decreases in p, so the upper bound is the mirror.
    lower = (
        0.0 if successes == 0 else _solve(tail, False, lambda p: _at_least(successes, trials, p))
    )
    upper = (
        1.0 if successes == trials else _solve(tail, True, lambda p: _at_most(successes, trials, p))
    )
    return lower, upper


def interval_text(successes: int, trials: int, confidence: float = 0.95) -> str:
    """`12/24 = 50% [29-71%]` — the form the ladder table is read in."""
    if trials == 0:
        return "0/0"
    low, high = clopper_pearson(successes, trials, confidence)
    return f"{successes}/{trials} = {successes / trials:.0%} [{low * 100:.0f}-{high * 100:.0f}%]"


__all__ = ["clopper_pearson", "interval_text"]
