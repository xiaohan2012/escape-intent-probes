"""What every real backend needs: tool schemas, prompt rendering, position (b).

Extracted from `hf_backend` when the vLLM backend arrived. None of this is
backend-specific — it is knowledge about the wire format and about the
tokenizer, and both backends must agree on it exactly or the two are not
measuring the same thing. In particular, a prompt rendered differently by the
two would make a vLLM trajectory and a HuggingFace trajectory incomparable
while looking identical in the logs.

The tool definitions go through the tokenizer's chat template rather than being
described in prose (D15), so the model sees tools in the format it was
post-trained on. This is the whole reason Q12 chose native tool calling:
compliance decides whether the MVP runs at all, and a parse failure that is
really a formatting failure makes E1 unreadable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from escape_probes.config import BASH, EDIT, SUBMIT
from escape_probes.model import Message, command_char_offset

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


def select_tool_schemas(tools: Sequence[str]) -> list[dict[str, Any]]:
    """Only the tools the run enables, so the model is never shown one the loop
    would reject (Q12)."""
    return [s for s in ALL_TOOL_SCHEMAS if s["function"]["name"] in tools]


class ChatTokenizer(Protocol):
    """The slice of a HuggingFace tokenizer both backends use.

    vLLM hands out the same object from `llm.get_tokenizer()`, which is why the
    rendering code below can be shared rather than reimplemented.
    """

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Any: ...

    def decode(self, token_ids: Any, **kwargs: Any) -> str: ...


def render_prompt(
    tokenizer: ChatTokenizer,
    messages: Sequence[Message],
    tool_schemas: Sequence[dict[str, Any]],
) -> list[int]:
    """The conversation as the model will see it, tools included.

    `add_generation_prompt=True` puts the context at the point where the model
    is about to speak — which is also where probe position (a) is read, by
    AgentLens's convention (D14).

    Rendering and tokenising are two explicit steps rather than `tokenize=True`:
    what that flag returns has changed across transformers versions, and a
    silently wrong type here would corrupt every token id we store.
    `add_special_tokens=False` because the template has already put in whatever
    the model expects.
    """
    text = tokenizer.apply_chat_template(
        [message.model_dump() for message in messages],
        tools=list(tool_schemas),
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(text, str):
        raise TypeError(f"chat template returned {type(text).__name__}, expected str")
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def command_token_index(
    gen_ids: Sequence[int],
    text: str,
    decode: Callable[[Sequence[int]], str],
) -> int | None:
    """Probe position (b): the first token of the tool call's command string.

    The character offset comes from the shared format helper; mapping it to a
    token index means finding the token that *contains* the character at that
    offset. Decoding prefixes is not free, but it is the only mapping that
    survives a tokenizer merging `"` with the word after it — the alignment
    failure the fake backend's character tokenizer was changed to avoid, and the
    one E4 checks for on real data.

    The comparison is strict. Token `k` begins at character `len(decode(ids[:k]))`,
    so the token containing `offset` is the first `k` whose *following* prefix
    passes it. A `>=` here — which is what the HuggingFace backend carried until
    this code was extracted and tested — returns the token before the command,
    shifting every position-(b) read one token early. Nothing downstream would
    have complained: the index is in range and the probe would simply have been
    trained on the wrong activation.
    """
    offset = command_char_offset(text)
    if offset is None:
        return None
    for index in range(1, len(gen_ids) + 1):
        if len(decode(gen_ids[:index])) > offset:
            return index - 1
    return None


__all__ = [
    "ALL_TOOL_SCHEMAS",
    "ChatTokenizer",
    "command_token_index",
    "render_prompt",
    "select_tool_schemas",
]
