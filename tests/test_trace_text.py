"""Tests for persisting the generation's text, not only its token span.

`add_step` stored `prompt_span`, `gen_span` and the token ids, which is
everything Pass 2 needs when the rollout ran locally — the ids *are* the frozen
asset (D12). On a hosted backend there are no ids, and the spans are both
`(0, 0)`, so what survived was the parsed tool call and the observation and
nothing else.

That is not enough to reconstruct the conversation. `qwen3.8-27b` emits roughly
700 completion tokens a step against about 30 in a rendered tool call, so 95% of
what the model wrote — its reasoning, its plan, its reading of the cue — was
discarded. A probe trained on the remainder is reading a conversation the model
never had, with the part most likely to carry intent removed.

Cheap to fix and worth storing even locally, where it is redundant with the ids:
it makes a trajectory readable without a tokenizer, which is what every
inspection in this project has actually needed.
"""

from escape_probes.model import Generation
from escape_probes.trace import TrajectoryWriter


class TestGenerationText:
    def writer_with(self, *texts: str) -> TrajectoryWriter:
        writer = TrajectoryWriter()
        for index, text in enumerate(texts):
            writer.add_step(
                Generation(
                    prompt_token_ids=tuple(range(index + 1)),
                    gen_token_ids=(100 + index,),
                    text=text,
                )
            )
        return writer

    def test_the_text_is_kept(self) -> None:
        writer = self.writer_with("I should look at grader.yaml first.")
        assert writer.steps[0].generation_text == "I should look at grader.yaml first."

    def test_each_step_keeps_its_own(self) -> None:
        writer = self.writer_with("first", "second", "third")
        assert [step.generation_text for step in writer.steps] == ["first", "second", "third"]

    def test_a_hosted_step_with_no_ids_still_keeps_the_text(self) -> None:
        # The case this exists for: both spans are (0, 0) and the ids are empty,
        # so the text is the only record of what the model said.
        writer = TrajectoryWriter()
        writer.add_step(Generation(prompt_token_ids=(), gen_token_ids=(), text="reasoning …"))
        step = writer.steps[0]
        assert (step.prompt_span, step.gen_span) == ((0, 0), (0, 0))
        assert step.generation_text == "reasoning …"

    def test_it_defaults_to_empty_so_old_trajectories_still_load(self) -> None:
        # 158 trajectories on disk predate the field. They must keep loading,
        # and must be distinguishable from new ones rather than silently filled.
        from escape_probes.trace import Step  # noqa: PLC0415

        step = Step(step_idx=0, prompt_span=(0, 3), gen_span=(3, 5))
        assert step.generation_text == ""
