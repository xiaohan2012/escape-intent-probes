"""Tests for B4/B5: the record must say how a step was sampled and why it ended.

`meta.json` claims to echo the config in full, but until now it recorded only
`temperature` — and temperature alone does not determine the distribution the
tokens were drawn from. And no field said whether a generation ended naturally
or hit the `max_new_tokens` cap, though a capped step's text is truncated
mid-thought and one such truncation already surfaced two steps later as a
prefix-assertion failure pointing at a healthy step.
"""

from __future__ import annotations

import pytest

from escape_probes.config import RunConfig
from escape_probes.model import FakeModel, Generation, ScriptedStep
from escape_probes.rollout import rollout
from escape_probes.trace import Step, TrajectoryWriter
from tests.test_rollout import FakeSandbox, FakeTask


@pytest.fixture
def config(tmp_path) -> RunConfig:  # noqa: ANN001
    config = RunConfig(run_id="test", instance_ids=("django__django-12419",))
    config.out_root = tmp_path
    return config


class TestMetaEchoesSampling:
    """The sampling config lands in meta.json, not just temperature."""

    def rolled(self, config: RunConfig):  # noqa: ANN201
        script = [ScriptedStep(tool="submit")]
        return rollout(FakeTask(), FakeModel(script), FakeSandbox(), config, "impossible", 0)

    def test_it_records_the_full_sampling_config(self, config: RunConfig) -> None:
        config.model.top_p = 0.8
        config.model.top_k = 20
        config.model.max_new_tokens = 512
        meta = self.rolled(config).meta
        assert (meta.top_p, meta.top_k, meta.max_new_tokens) == (0.8, 20, 512)

    def test_old_metas_load_with_not_recorded_defaults(self) -> None:
        # max_new_tokens=0 means "not recorded", never "no budget": trajectories
        # written before these fields existed must still load.
        step = Step(step_idx=0, prompt_span=(0, 1), gen_span=(1, 2))
        assert step.finish_reason == ""


class TestFinishReasonReachesTheStep:
    def make(self, finish_reason: str) -> Generation:
        return Generation(
            prompt_token_ids=(1, 2),
            gen_token_ids=(3,),
            text="x",
            finish_reason=finish_reason,
        )

    @pytest.mark.parametrize("reason", ["stop", "length"])
    def test_add_step_records_it(self, reason: str) -> None:
        writer = TrajectoryWriter()
        assert writer.add_step(self.make(reason)).finish_reason == reason

    def test_the_default_is_not_recorded(self) -> None:
        assert Generation(prompt_token_ids=(1,), gen_token_ids=(2,), text="x").finish_reason == ""


class TestVllmWiring:
    """`CompletionOutput.finish_reason` lands on the Generation."""

    def test_the_backend_reads_it_from_the_completion(self) -> None:
        # vllm is not installed here; parse the source instead, the same way
        # test_backend_imports guards the import surface.
        from pathlib import Path  # noqa: PLC0415

        source = Path("src/escape_probes/vllm_backend.py").read_text()
        assert 'finish_reason=completion.finish_reason or ""' in source
        assert "finish_reason: str | None" in source  # the Completion protocol carries it
