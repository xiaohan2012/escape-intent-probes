"""A model backend on HuggingFace `transformers`, one step at a time.

Serial and unbatched, per D13 — make it run first, then let a measurement decide
whether concurrency is needed. That measurement came back badly (16 tok/s, 34%
GPU utilisation), so `vllm_backend` is now the path for rollouts. This one stays
for two reasons that are not nostalgia:

* Pass 2 needs forward hooks on the residual stream, which vLLM does not expose.
  Whatever captures activations will be a `transformers` model, and it must
  tokenise a conversation exactly as the rollout did — so the rendering code is
  shared (`chat.py`) rather than duplicated.
* It is the reference implementation when a vLLM trajectory looks wrong.

What this class must get right, because Pass 2 depends on it:

* **Token ids, not text.** `prompt_token_ids` are the ids actually fed to the
  model, and `gen_token_ids` the ids it actually produced. Re-tokenising text
  later is the mismatch E4 exists to catch, and it is AgentLens's one weakness
  (D14) — they store conversations as text and re-render at extraction time.
* **Probe position (b).** The offset of the command string inside the
  generation, computed where the tokenizer is at hand.
"""

from __future__ import annotations

from collections.abc import Sequence

from escape_probes.chat import render_prompt, select_tool_schemas
from escape_probes.config import DEFAULT_TOOLS, ModelConfig
from escape_probes.model import Generation, Message, command_token_index


class HFModel:
    """A local causal LM driven one step at a time.

    Loading is eager: a failure to load should happen before a batch starts, not
    on the first trajectory.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: str | None = None,
        tools: tuple[str, ...] = DEFAULT_TOOLS,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.config = config
        self.tool_schemas = select_tool_schemas(tools)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            dtype=getattr(torch, config.dtype),
            device_map=device or config.device_map,
        )
        self.model.eval()
        self._torch = torch

    def generate(self, messages: Sequence[Message]) -> Generation:
        prompt_ids = render_prompt(self.tokenizer, messages, self.tool_schemas)
        inputs = self._torch.tensor([prompt_ids], device=self.model.device)

        with self._torch.inference_mode():
            output = self.model.generate(
                inputs,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature or None,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                # Explicit, because `generate()` fills unspecified knobs from
                # the checkpoint's generation_config — Qwen ships 1.05 there —
                # while the vLLM path's explicit SamplingParams uses 1.0. A
                # reference implementation must sample the same distribution.
                repetition_penalty=1.0,
            )

        gen_ids = output[0].tolist()[len(prompt_ids) :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return Generation(
            prompt_token_ids=tuple(prompt_ids),
            gen_token_ids=tuple(gen_ids),
            text=text,
            # `generate()` returns no finish reason; a generation that used the
            # whole budget is taken as capped. It can also have ended naturally
            # at exactly the cap, which vLLM would distinguish and this cannot.
            finish_reason="length" if len(gen_ids) >= self.config.max_new_tokens else "stop",
            tool_start_token_idx=command_token_index(gen_ids, text, self._decode),
        )

    def _decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


__all__ = ["HFModel"]
