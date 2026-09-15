"""Tests for reporting how much prompt the engine actually reused.

A hybrid model's prefix cache is all-or-nothing across the whole model, not
per layer group: `HybridKVCacheCoordinator.find_longest_cache_hit` reconciles
every KV-cache group down to one `hit_length`, and any group that cannot match
drags it to zero — so the 16 full-attention layers that matched perfectly get
re-prefilled alongside the 48 gated-delta-net layers that did not. vLLM issue
#45238 reports exactly this for align mode, at roughly 2x TTFT.

This project's rollouts are the worst case for it. Each step's prompt is the
previous one plus a suffix, so *every* round after the first should be almost
entirely a cache hit. If it is not, a round re-prefills the whole conversation
for all 24 trajectories instead of the few hundred new tokens.

The engine knows and says so — `RequestOutput.num_cached_tokens` — and this
backend discarded it, so a run had no way to tell a working cache from a broken
one. That is the same shape as every other failure this week: not wrong, just
invisible.

The count is kept per request and lands on each `Step`, because the diagnostic
is the ratio against that step's `prompt_span` — a round total would make the
ratio unreadable in `steps.jsonl`.
"""

from __future__ import annotations

from escape_probes.model import Generation


class FakeCompletion:
    def __init__(self, text: str = "x") -> None:
        self.token_ids = [7, 8]
        self.text = text


class FakeOutput:
    def __init__(self, cached: int | None) -> None:
        self.outputs = [FakeCompletion()]
        if cached is not None:
            self.num_cached_tokens = cached


class TestCachedPromptTokens:
    def read(self, outputs: list[FakeOutput]) -> list[int]:
        from escape_probes.vllm_backend import cached_prompt_tokens  # noqa: PLC0415

        return cached_prompt_tokens(outputs)

    def test_one_count_per_request(self) -> None:
        assert self.read([FakeOutput(100), FakeOutput(250)]) == [100, 250]

    def test_a_cold_round_reads_zero(self) -> None:
        # The signal we are looking for. Zero on round 2 and later means the
        # reconciliation collapsed, not that the cache is warming up.
        assert self.read([FakeOutput(0), FakeOutput(0)]) == [0, 0]

    def test_an_engine_that_does_not_report_it_is_not_an_error(self) -> None:
        # Older vLLM, or a backend that does not carry the attribute. Absent
        # must not crash a rollout, and must not read as zero either.
        assert self.read([FakeOutput(None)]) == [-1]

    def test_mixed_reporting_is_treated_as_absent(self) -> None:
        assert self.read([FakeOutput(10), FakeOutput(None)]) == [-1, -1]

    def test_no_outputs(self) -> None:
        assert self.read([]) == []


class TestGenerationCarriesIt:
    def test_the_field_defaults_to_unknown(self) -> None:
        # -1 rather than 0: "the engine did not say" and "the engine said none"
        # are different findings, and only one of them is a bug.
        assert Generation(prompt_token_ids=(1,), gen_token_ids=(2,), text="x").cached_tokens == -1


class TestStepCarriesIt:
    def make(self, cached: int) -> Generation:
        return Generation(
            prompt_token_ids=(1, 2, 3), gen_token_ids=(4,), text="x", cached_tokens=cached
        )

    def test_add_step_records_it(self) -> None:
        from escape_probes.trace import TrajectoryWriter  # noqa: PLC0415

        writer = TrajectoryWriter()
        step = writer.add_step(self.make(2))
        assert step.cached_tokens == 2

    def test_default_is_unknown_for_old_trajectories(self) -> None:
        from escape_probes.trace import Step  # noqa: PLC0415

        assert Step(step_idx=0, prompt_span=(0, 3), gen_span=(3, 4)).cached_tokens == -1


class TestCacheText:
    def load(self):  # noqa: ANN201
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from run_batch import cache_text  # noqa: PLC0415

        return cache_text

    def trajectory(self, cached: list[int]):  # noqa: ANN201
        from escape_probes.trace import TrajectoryWriter  # noqa: PLC0415

        writer = TrajectoryWriter()
        prompt: tuple[int, ...] = ()
        for count in cached:
            prompt = prompt + (1, 1, 1, 1)
            writer.add_step(
                Generation(
                    prompt_token_ids=prompt, gen_token_ids=(2,), text="x", cached_tokens=count
                )
            )

        class Holder:
            steps = writer.steps

        return Holder()

    def test_ratio_over_steps_after_the_first(self) -> None:
        # Steps 1 and 2 have prompts of 8 and 12 tokens; 8 + 12 = 20, all hit.
        assert self.load()(self.trajectory([0, 8, 12])) == " cache=100%"

    def test_the_broken_cache_reads_zero(self) -> None:
        assert self.load()(self.trajectory([0, 0, 0])) == " cache=0%"

    def test_silent_when_the_backend_did_not_report(self) -> None:
        assert self.load()(self.trajectory([-1, -1])) == ""

    def test_silent_on_a_one_step_trajectory(self) -> None:
        assert self.load()(self.trajectory([0])) == ""


class TestLiveRoundReport:
    """`drive_batch` surfaces each round's generations while the batch runs.

    Trajectories reach disk only when the whole batch does, so a diagnostic
    that waits for `steps.jsonl` — the cache-hit ratio above all — is eighty
    minutes late. `on_round` hands the caller every round's generations as
    they come back from the engine."""

    def test_every_round_is_reported_with_its_generations(self) -> None:
        from escape_probes.config import RunConfig  # noqa: PLC0415
        from escape_probes.model import ScriptedStep  # noqa: PLC0415
        from escape_probes.rollout import drive_batch, rollout_steps  # noqa: PLC0415
        from tests.test_rollout import FakeSandbox, FakeTask, RecordingBatchModel  # noqa: PLC0415

        config = RunConfig(run_id="test-run")
        model = RecordingBatchModel(
            {
                "one-instance": [ScriptedStep(tool="submit")],
                "two-instance": [ScriptedStep(tool="bash", arguments={"cmd": "ls"})]
                + [ScriptedStep(tool="submit")],
            }
        )
        steps = [
            rollout_steps(FakeTask(instance_id=name), FakeSandbox(), config, "impossible", 0)
            for name in ("one-instance", "two-instance")
        ]
        rounds: list[tuple[int, int]] = []
        drive_batch(steps, model, on_round=lambda index, gens: rounds.append((index, len(gens))))
        assert [index for index, _ in rounds] == list(range(len(rounds)))
        assert rounds[0] == (0, 2)
        assert rounds[-1][1] == 1  # the finished trajectory left the round
