"""The fast path: vLLM, with a batch entry point for the lock-step runner.

Why replace the `transformers` path for rollouts, measured rather than guessed:
model time was **94%** of wall clock, sandbox 6%, decode **16 tok/s** at **34%**
GPU utilisation. The card was idle. Decode is memory-bandwidth bound, so reading
the weights for one token costs about the same whether one sequence is decoding
or sixteen — which means the idle fraction is not a tuning problem but an
unused batch dimension.

Two things this buys beyond speed, both of which matter to D12:

* vLLM's offline API takes `TokensPrompt(prompt_token_ids=...)` and returns
  `token_ids`, so a step never round-trips through text. The HuggingFace path
  tokenises a rendered string and then decodes the output again; this one hands
  over the ids we rendered and gets ids back.
* `generate_batch` exists because the loop's concurrency belongs in the caller,
  not here. One `llm.generate([...])` call with N prompts lets vLLM do its own
  continuous batching, so there is no padding or ragged-length bookkeeping to
  write — the lock-step runner just collects one prompt per live trajectory and
  hands over the round.

Not a replacement for `hf_backend`: Pass 2 needs forward hooks on the residual
stream, which vLLM does not expose. Both render prompts through `chat.py` so a
trajectory produced here is tokenised exactly as Pass 2 will re-tokenise it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from escape_probes.chat import render_prompt, select_tool_schemas
from escape_probes.config import DEFAULT_TOOLS, ModelConfig
from escape_probes.model import (
    THINK_OPEN,
    Generation,
    Message,
    command_token_index,
    split_reasoning,
)


class Completion(Protocol):
    """The part of vLLM's `CompletionOutput` this module reads.

    `text` is already detokenised without special tokens, matching what the
    HuggingFace path stores, so the two backends' `Generation.text` are
    comparable.
    """

    token_ids: Sequence[int]
    text: str
    finish_reason: str | None


def cached_prompt_tokens(outputs: Sequence[Any]) -> list[int]:
    """Prompt tokens the engine reused, one count per request, or -1s if it
    did not say.

    Per request rather than a round total, because the counts land in
    `steps.jsonl` next to each step's `prompt_span` — the ratio of the two is
    the diagnostic, and a round total would make that ratio unreadable.

    On a hybrid model — 48 gated-delta-net layers here against 16 full-attention
    ones — a prefix-cache hit is reconciled across every KV-cache group into one
    length, and a group that cannot match drags it to zero. The 16 layers that
    matched perfectly are then re-prefilled anyway (vLLM #45238, roughly 2x TTFT
    in their report).

    Every round after the first should be almost entirely a hit here, because
    each prompt is the previous one plus a suffix. A zero from round 2 onwards
    means the whole conversation is being re-prefilled for every trajectory —
    which costs a great deal and says nothing in any log, since this backend
    used to discard the number.
    """
    counts = [getattr(output, "num_cached_tokens", None) for output in outputs]
    if any(count is None for count in counts):
        return [-1] * len(counts)
    return counts  # ty: ignore


class VLLMModel:
    """A local causal LM served by vLLM's offline engine.

    Loading is eager, as with `HFModel`: on four cards the engine takes a while
    to come up, and it should fail before a batch starts rather than on the
    first trajectory.
    """

    def __init__(
        self,
        config: ModelConfig,
        tools: tuple[str, ...] = DEFAULT_TOOLS,
    ) -> None:
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        self.config = config
        self.tool_schemas = select_tool_schemas(tools)
        self.llm = LLM(
            model=config.model_id,
            revision=config.revision,
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            enable_prefix_caching=True,
            # Every step of a trajectory re-sends the whole conversation with
            # one exchange appended, so the prefix is shared across steps and
            # its prefill is pure waste without this. Cheap on these boxes:
            # the cache lives in the same block pool as the KV cache.
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._sampling = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k or -1,
            # vLLM spells "no cutoff" as -1 where `transformers` spells it 0,
            # and our config follows `transformers` (D20).
            max_tokens=self.config.max_new_tokens,
        )

    def generate(self, messages: Sequence[Message]) -> Generation:
        return self.generate_batch([messages])[0]

    def generate_batch(self, conversations: Sequence[Sequence[Message]]) -> list[Generation]:
        """One engine call for N conversations, results in the order given.

        The order matters more than it looks: the lock-step runner maps results
        back onto trajectories positionally, and vLLM is free to finish requests
        in any order internally. `llm.generate` returns outputs aligned to its
        input list, which is the property being relied on here.
        """
        from vllm import TokensPrompt  # noqa: PLC0415

        prompts = [
            render_prompt(self.tokenizer, messages, self.tool_schemas) for messages in conversations
        ]
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
            self._sampling,
        )
        cached = cached_prompt_tokens(outputs)
        return [
            self._to_generation(prompt_ids, output.outputs[0], count)
            for prompt_ids, output, count in zip(prompts, outputs, cached, strict=True)
        ]

    def _to_generation(
        self, prompt_ids: list[int], completion: Completion, cached: int = -1
    ) -> Generation:
        gen_ids = list(completion.token_ids)
        text = completion.text
        # Whether the prompt ends inside a think block is a property of this
        # model's template, and this is the only place that has rendered it. A
        # reasoning model whose generation hit the token cap mid-thought emits
        # no closing tag, and calling that text `content` makes the next step's
        # prompt stop being a prefix (D24).
        opened = self.tokenizer.decode(prompt_ids[-4:]).endswith(THINK_OPEN + "\n")
        reasoning, answer = split_reasoning(text, think_opened=opened)
        return Generation(
            prompt_token_ids=tuple(prompt_ids),
            gen_token_ids=tuple(gen_ids),
            text=text,
            reasoning=reasoning,
            answer=answer,
            finish_reason=completion.finish_reason or "",
            cached_tokens=cached,
            tool_start_token_idx=command_token_index(gen_ids, text, self._decode),
        )

    def _decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


__all__ = ["VLLMModel"]
