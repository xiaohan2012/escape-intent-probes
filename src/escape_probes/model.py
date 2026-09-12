"""Model backends for the rollout loop.

Two things live here: the shape of one generation step, and a scripted fake
backend that produces those generations without loading any weights.

The fake backend exists so the whole loop — tool parsing, container execution,
persistence, t* labelling — can be developed and regression-tested with no GPU
(stage0-plan Step 0.3). Bugs in those parts are the bulk of Stage 0's
engineering, and debugging them while a 30B model occupies a rented GPU is
waste. It also keeps failures attributable: once the real model is attached, a
bad trajectory is known not to be the loop's fault (D3).

A `Generation` carries token ids rather than text because D12 makes the token
sequence the frozen asset: Pass 2 teacher-forces exactly these ids, so anything
that re-tokenises text in between is a defect.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass(frozen=True)
class ParsedCall:
    """A tool call as it came off the wire, before any validation."""

    name: str | None
    arguments: dict[str, str]
    value_offset: int | None
    """Character offset of the first argument's value — probe position (b)."""


class Message(BaseModel):
    """One turn of the conversation handed to the model."""

    role: str
    content: str


class Generation(BaseModel):
    """What the model produced at one step, in the form Pass 2 will need.

    `prompt_token_ids` and `gen_token_ids` concatenate, in that order, into the
    slice of the trajectory's token stream belonging to this step.
    """

    prompt_token_ids: tuple[int, ...]
    gen_token_ids: tuple[int, ...]
    text: str

    tool_start_token_idx: int | None = None
    """Index into `gen_token_ids` of the first token of the tool call's command
    string — probe position (b). `None` when the generation has no parsable tool
    call. Recorded here because the backend knows the boundary for free, while
    recovering it later means re-parsing JSON through a tokenizer (D12)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    """What the request actually cost, as the server counted it. Zero on a local
    backend, where the counts are `len(prompt_token_ids)` and
    `len(gen_token_ids)` and storing them again would be a second copy that can
    disagree with the first.

    They exist for the hosted path, which has no token ids at all (D22). The
    screen's budget rests on an estimate that is quadratic in the step count,
    because every step resends the conversation — the kind of number that is
    wrong by a factor of two with nothing looking wrong. The server reports the
    real one on every response."""

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.gen_token_ids


class ModelBackend(Protocol):
    """What the rollout loop needs from a model, fake or real."""

    def generate(self, messages: Sequence[Message]) -> Generation: ...


class CharTokenizer:
    """A character-level tokenizer with a growing vocabulary.

    Only for the fake backend. It is not a stand-in for a real tokenizer — it
    exists so fake trajectories carry genuine integer ids, which means the
    persistence layer and the Pass 1 / Pass 2 index arithmetic are exercised by
    tests that need neither weights nor `transformers`.

    Character level, not word level, so that a character offset into the text is
    exactly a token index. A word tokenizer would glue the opening quote onto
    the command (`"cat`), putting the command's first token boundary inside a
    token and making position (b) unrepresentable — a small rehearsal of the
    real alignment problem that E4 exists to catch.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> tuple[int, ...]:
        ids = []
        for char in text:
            if char not in self._vocab:
                self._vocab[char] = len(self._vocab)
            ids.append(self._vocab[char])
        return tuple(ids)

    def decode(self, ids: Sequence[int]) -> str:
        inverse = {i: c for c, i in self._vocab.items()}
        return "".join(inverse[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


def find_tool_payload(text: str) -> tuple[str, int] | None:
    """The first tool call's payload, and where it starts in `text`.

    One definition of the wire format, shared by the parser and by whatever
    computes probe position (b). Two copies drifted once already — one accepted
    an unclosed block that the other rejected — and the failure mode of that
    drift is a silently wrong position (b) rather than a crash.
    """
    start = text.find(TOOL_CALL_OPEN)
    if start == -1:
        return None
    payload_start = start + len(TOOL_CALL_OPEN)
    end = text.find(TOOL_CALL_CLOSE, payload_start)
    if end == -1:
        return None
    return text[payload_start:end], payload_start


_FUNCTION = re.compile(r"<function=([^>\s]+)\s*>")
_PARAMETER = re.compile(r"<parameter=([^>\s]+)\s*>\n?(.*?)\n?</parameter>", re.DOTALL)


def decode_payload(payload: str, payload_start: int) -> ParsedCall | None:
    """Read a tool call in whichever wire format the model emitted.

    Two are supported because models differ and the choice is not ours to make:
    a tool call must be parsed in the format the model was post-trained to
    produce, or every step fails to parse and E1 measures formatting rather than
    capability (D15). Qwen3-Coder emits a tag form:

        <function=bash>
        <parameter=cmd>
        ls -la
        </parameter>
        </function>

    while other families emit JSON. Neither is a fallback for the other; both
    are first-class.

    `value_offset` is the character offset, in the enclosing generation, of the
    first argument's value — probe position (b), the last moment before the
    action is named.
    """
    function = _FUNCTION.search(payload)
    if function is not None:
        arguments = {}
        value_offset = None
        for match in _PARAMETER.finditer(payload):
            key, value = match.group(1), match.group(2)
            arguments[key] = value
            if value_offset is None:
                value_offset = payload_start + match.start(2)
        return ParsedCall(name=function.group(1), arguments=arguments, value_offset=value_offset)

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    arguments = decoded.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    arguments = {str(k): str(v) for k, v in arguments.items()}
    value_offset = None
    if arguments:
        first = next(iter(arguments.values()))
        found = payload.find(first)
        value_offset = None if found == -1 else payload_start + found
    return ParsedCall(name=decoded.get("name"), arguments=arguments, value_offset=value_offset)


def render_tool_call(name: str, arguments: dict[str, str]) -> str:
    """Emit a tool call in Qwen3-Coder's tag form.

    The fake backend speaks the same dialect as the model we actually run, so a
    change to parsing is exercised by the fast tests rather than discovered on
    the GPU.
    """
    lines = [TOOL_CALL_OPEN, f"<function={name}>"]
    for key, value in arguments.items():
        lines.append(f"<parameter={key}>\n{value}\n</parameter>")
    lines += ["</function>", TOOL_CALL_CLOSE]
    return "\n".join(lines)


class ScriptedStep(BaseModel):
    """One entry of a fake model's script."""

    reasoning: str = ""
    """Free text emitted before the tool call, as a real model would. The
    keyword baseline and the N3 audit both read this part of the output."""

    tool: str | None = None
    arguments: dict[str, str] = Field(default_factory=dict)

    raw: str | None = None
    """Bypass `tool`/`arguments` and emit this verbatim — for testing malformed
    output and the parse-retry path."""

    def render(self) -> str:
        if self.raw is not None:
            return self.raw
        if self.tool is None:
            return self.reasoning
        call = render_tool_call(self.tool, self.arguments)
        return f"{self.reasoning}\n{call}" if self.reasoning else call


class FakeModel:
    """Replays a fixed script of steps, ignoring the conversation.

    Exhausting the script raises, so a loop that runs longer than the script
    expects fails loudly instead of silently repeating its last action.
    """

    def __init__(self, script: Sequence[ScriptedStep]) -> None:
        if not script:
            raise ValueError("a fake model needs at least one scripted step")
        self._script = list(script)
        self._cursor = 0
        self.tokenizer = CharTokenizer()

    @property
    def steps_remaining(self) -> int:
        return len(self._script) - self._cursor

    def generate(self, messages: Sequence[Message]) -> Generation:
        if self._cursor >= len(self._script):
            raise IndexError(
                f"fake model script exhausted after {len(self._script)} steps; "
                "extend the script or lower max_steps"
            )
        step = self._script[self._cursor]
        self._cursor += 1

        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        text = step.render()
        prompt_ids = self.tokenizer.encode(prompt)
        gen_ids = self.tokenizer.encode(text)

        return Generation(
            prompt_token_ids=prompt_ids,
            gen_token_ids=gen_ids,
            text=text,
            tool_start_token_idx=_command_token_index(text, self.tokenizer),
        )


def command_char_offset(text: str) -> int | None:
    """Character offset of the tool call's first argument value.

    Probe position (b) is the last moment before the action is emitted. A
    backend maps this offset through its own tokenizer; the offset itself is
    format knowledge, not tokenizer knowledge, so it lives here once.
    """
    found = find_tool_payload(text)
    if found is None:
        return None
    call = decode_payload(*found)
    return None if call is None else call.value_offset


def command_token_index(
    gen_ids: Sequence[int],
    text: str,
    decode: Callable[[Sequence[int]], str],
) -> int | None:
    """Probe position (b): the first token of the tool call's command string.

    One implementation, called by every backend. There used to be two — this
    module's, for the fake backend's character tokenizer, and a copy inside
    `HFModel` — and only this one was tested. The copy compared decoded prefix
    length with `>=` and so returned the token *before* the command, in range
    and without raising, which would have put every probe read one token early.

    The comparison is strict because token `k` begins at character
    `len(decode(ids[:k]))`, so the token containing `offset` is the first `k`
    whose *following* prefix passes it.

    Decoding prefixes is not free, but it is the only mapping that survives a
    tokenizer merging `"` with the word after it — the alignment failure the
    fake backend's character tokenizer was changed to avoid, and the one E4
    checks for on real data.
    """
    offset = command_char_offset(text)
    if offset is None:
        return None
    for index in range(1, len(gen_ids) + 1):
        if len(decode(gen_ids[:index])) > offset:
            return index - 1
    return None


def _command_token_index(text: str, tokenizer: CharTokenizer) -> int | None:
    return command_token_index(tokenizer.encode(text), text, tokenizer.decode)
