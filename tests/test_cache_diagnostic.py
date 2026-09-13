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
    def read(self, outputs: list[FakeOutput]) -> int:
        from escape_probes.vllm_backend import cached_prompt_tokens  # noqa: PLC0415

        return cached_prompt_tokens(outputs)

    def test_it_sums_across_the_round(self) -> None:
        assert self.read([FakeOutput(100), FakeOutput(250)]) == 350

    def test_a_cold_round_reads_zero(self) -> None:
        # The signal we are looking for. Zero on round 2 and later means the
        # reconciliation collapsed, not that the cache is warming up.
        assert self.read([FakeOutput(0), FakeOutput(0)]) == 0

    def test_an_engine_that_does_not_report_it_is_not_an_error(self) -> None:
        # Older vLLM, or a backend that does not carry the attribute. Absent
        # must not crash a rollout, and must not read as zero either.
        assert self.read([FakeOutput(None)]) == -1

    def test_mixed_reporting_is_treated_as_absent(self) -> None:
        assert self.read([FakeOutput(10), FakeOutput(None)]) == -1

    def test_no_outputs(self) -> None:
        assert self.read([]) == -1


class TestGenerationCarriesIt:
    def test_the_field_defaults_to_unknown(self) -> None:
        # -1 rather than 0: "the engine did not say" and "the engine said none"
        # are different findings, and only one of them is a bug.
        assert Generation(prompt_token_ids=(1,), gen_token_ids=(2,), text="x").cached_tokens == -1
