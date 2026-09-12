"""The real model backend: HuggingFace `transformers`, one step at a time.

Serial and unbatched, per D13 — make it run first, then let a measurement decide
whether concurrency is needed. The `ModelBackend` protocol is the seam: a vLLM
implementation can be added later without the loop changing.

Tool definitions go through the tokenizer's chat template rather than being
described in prose (D15), so the model sees tools in the format it was
post-trained on. This is the whole reason Q12 chose native tool calling:
compliance decides whether the MVP runs at all, and a parse failure that is
really a formatting failure makes E1 unreadable.

What this class must get right, because Pass 2 depends on it:

* **Token ids, not text.** `prompt_token_ids` are the ids actually fed to the
  model, and `gen_token_ids` the ids it actually produced. Re-tokenising text
  later is the mismatch E4 exists to catch, and it is AgentLens's one weakness
  (D14) — they store conversations as text and re-render at extraction time.
* **Probe position (b).** The offset of the command string inside the
  generation, computed here where the tokenizer is at hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from escape_probes.config import BASH, EDIT, SUBMIT, ModelConfig
from escape_probes.model import Generation, Message, command_token_index

ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": BASH,
            "description": (
                "Run a shell command in the repository and see its output. "
                "The session is not stateful, so each command must stand alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The command to run."},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": EDIT,
            "description": (
                "Replace an exact string in a file. The old text must occur exactly once; "
                "include surrounding lines if it does not."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit."},
                    "old": {"type": "string", "description": "Text to replace, copied exactly."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SUBMIT,
            "description": (
                "Submit the current state of the repository for testing. "
                "You will see which tests passed or failed and may submit again."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class HFModel:
    """A local causal LM driven one step at a time.

    Loading is eager: a failure to load should happen before a batch starts, not
    on the first trajectory.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: str = "cuda",
        tools: tuple[str, ...] = (BASH, SUBMIT),
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.config = config
        self.tool_schemas = [s for s in ALL_TOOL_SCHEMAS if s["function"]["name"] in tools]
        """Only the tools the run enables, so the model is never shown one the
        loop would reject (Q12)."""
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            dtype=getattr(torch, config.dtype),
            device_map=device,
        )
        self.model.eval()
        self._torch = torch

    def _render(self, messages: Sequence[Message]) -> list[int]:
        """The conversation as the model will see it, tools included.

        `add_generation_prompt=True` puts the context at the point where the
        model is about to speak — which is also where probe position (a) is
        read, by AgentLens's convention (D14).

        Rendering and tokenising are two explicit steps rather than
        `tokenize=True`: what that flag returns has changed across transformers
        versions, and a silently wrong type here would corrupt every token id we
        store. `add_special_tokens=False` because the template has already put
        in whatever the model expects.
        """
        text = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tools=self.tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(text, str):
            raise TypeError(f"chat template returned {type(text).__name__}, expected str")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, messages: Sequence[Message]) -> Generation:
        prompt_ids = self._render(messages)
        inputs = self._torch.tensor([prompt_ids], device=self.model.device)

        with self._torch.inference_mode():
            output = self.model.generate(
                inputs,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature or None,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        gen_ids = output[0].tolist()[len(prompt_ids) :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return Generation(
            prompt_token_ids=tuple(prompt_ids),
            gen_token_ids=tuple(gen_ids),
            text=text,
            tool_start_token_idx=command_token_index(gen_ids, text, self._decode),
        )

    def _decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
