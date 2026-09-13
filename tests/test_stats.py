"""Tests for the exact binomial interval.

Checked against closed forms where one exists, because the implementation is a
bisection and a bisection that converges on the wrong root looks exactly like
one that converges on the right one. For `k = 0` the Clopper-Pearson upper bound
is `1 - (alpha/2)**(1/n)` in closed form, which pins the cases every conclusion
in D23 rests on.
"""

import pytest

from escape_probes.stats import clopper_pearson, interval_text


class TestClopperPearson:
    @pytest.mark.parametrize(
        ("successes", "trials", "expected"),
        [
            # The screen's nulls, and why they settled nothing.
            (0, 3, (0.0, 0.7076)),
            # The descent's sample, and why 24 was the number chosen.
            (0, 24, (0.0, 0.1425)),
            # The ablation, whose 108 is the only large-n null we have.
            (0, 108, (0.0, 0.0336)),
            # qwen3.6-35b-a3b: one crossing is existence, not a rate.
            (1, 24, (0.0011, 0.2113)),
            # glm-5.3-flash: the first cell with a usable rate.
            (12, 24, (0.2912, 0.7088)),
            # The screen's crossing cells, all n=3.
            (1, 3, (0.0084, 0.9057)),
            (2, 3, (0.0943, 0.9916)),
        ],
    )
    def test_matches_the_published_interval(
        self, successes: int, trials: int, expected: tuple[float, float]
    ) -> None:
        assert clopper_pearson(successes, trials) == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize("trials", [1, 3, 24, 108])
    def test_a_zero_matches_the_closed_form(self, trials: int) -> None:
        # 1 - (alpha/2)**(1/n), the only case with an elementary solution.
        expected = 1.0 - 0.025 ** (1 / trials)
        assert clopper_pearson(0, trials)[1] == pytest.approx(expected, abs=1e-6)

    def test_a_zero_has_a_lower_bound_of_exactly_zero(self) -> None:
        # Exactly, not approximately: the bound is quoted as "[0.00, …]".
        assert clopper_pearson(0, 24)[0] == 0.0

    def test_all_successes_has_an_upper_bound_of_exactly_one(self) -> None:
        assert clopper_pearson(24, 24)[1] == 1.0

    def test_the_interval_narrows_as_the_sample_grows(self) -> None:
        # The argument for spending money on n rather than on more cells.
        widths = [
            clopper_pearson(0, trials)[1] - clopper_pearson(0, trials)[0]
            for trials in (3, 12, 24, 48)
        ]
        assert widths == sorted(widths, reverse=True)

    def test_the_interval_contains_the_estimate(self) -> None:
        for trials in (3, 24):
            for successes in range(trials + 1):
                low, high = clopper_pearson(successes, trials)
                assert low <= successes / trials <= high

    def test_a_wider_confidence_gives_a_wider_interval(self) -> None:
        narrow = clopper_pearson(12, 24, confidence=0.80)
        wide = clopper_pearson(12, 24, confidence=0.99)
        assert wide[0] < narrow[0]
        assert wide[1] > narrow[1]

    @pytest.mark.parametrize(("successes", "trials"), [(-1, 3), (4, 3), (0, 0)])
    def test_a_nonsensical_count_raises(self, successes: int, trials: int) -> None:
        with pytest.raises(ValueError):
            clopper_pearson(successes, trials)


class TestIntervalText:
    def test_the_form_the_table_is_read_in(self) -> None:
        assert interval_text(12, 24) == "12/24 = 50% [29-71%]"

    def test_a_null_reads_as_an_upper_bound(self) -> None:
        assert interval_text(0, 24) == "0/24 = 0% [0-14%]"

    def test_an_empty_cell_does_not_divide_by_zero(self) -> None:
        # A cell whose every trajectory died on a parse failure has no trials.
        assert interval_text(0, 0) == "0/0"


class TestCellCensoring:
    """A parse death censors a trajectory — unless the crossing already happened.

    Both halves have bitten. Counting a dead trajectory as an observed zero
    deflates the rate (it never had the chance to cross); dropping a dead
    trajectory that *did* cross throws away an observed positive, which is worse,
    and one glm-5.3-flash trajectory is exactly that case — t*=13, dead at 17.
    """

    def labels(self, *specs: tuple[str, int | None]) -> list:
        from escape_probes.labels import Labels  # noqa: PLC0415

        return [
            Labels(
                instance_id=f"i{index}",
                condition="impossible",
                seed=0,
                outcome=outcome,
                t_star=t_star,
                n_steps=25,
            )
            for index, (outcome, t_star) in enumerate(specs)
        ]

    def cell(self, *specs: tuple[str, int | None]):  # noqa: ANN202
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from report_ladder import Cell  # noqa: PLC0415

        return Cell("c", self.labels(*specs))

    def test_a_death_before_any_crossing_is_dropped(self) -> None:
        cell = self.cell(("parse_failed", None), ("max_steps", None))
        assert len(cell.live) == 1
        assert cell.dead == 1

    def test_a_death_after_a_crossing_is_kept(self) -> None:
        cell = self.cell(("parse_failed", 13))
        assert len(cell.live) == 1
        assert cell.dead == 0
        assert cell.count("crossed") == 1

    def test_the_rate_uses_the_uncensored_denominator(self) -> None:
        # One crossing, one honest zero, one censored: 1/2, not 1/3.
        cell = self.cell(("max_steps", 5), ("max_steps", None), ("parse_failed", None))
        assert (cell.count("crossed"), len(cell.live)) == (1, 2)
