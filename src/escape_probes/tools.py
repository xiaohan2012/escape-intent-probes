"""Parsing and dispatching the agent's tool calls.

Two tools only — `bash` and `submit` (Q12). `bash` is enough on its own: reading
is `cat`, editing is `python -c`, testing is the repo's own runner. Deliberately
absent are `think`, which would move reasoning into a separate call and blur
probe position (b), and `text_editor`, which is the first remedy to try if E1
fails rather than a starting point.

A malformed call is not fatal: the parse error goes back to the model as an
observation, as a real harness would, and it gets another try. The count of
those retries is itself diagnostic — a high rate means the format is wrong for
this model, not that the model cannot do the task, and E1 must not be read as
the latter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from escape_probes.config import BASH, DEFAULT_TOOLS, SUBMIT
from escape_probes.model import TOOL_CALL_CLOSE, TOOL_CALL_OPEN, find_tool_payload

__all__ = ["BASH", "SUBMIT", "ToolCall", "ToolParseError", "parse_tool_call", "truncate"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]

    @property
    def command(self) -> str:
        """The shell command, for a `bash` call. Empty for anything else."""
        return self.arguments.get("cmd", "")


class ToolParseError(ValueError):
    """The generation did not contain a usable tool call.

    The message is shown to the model verbatim, so it says what to do rather
    than merely what went wrong.
    """


def parse_tool_call(text: str, allowed: tuple[str, ...] = DEFAULT_TOOLS) -> ToolCall:
    found = find_tool_payload(text)
    if found is None:
        if TOOL_CALL_OPEN in text:
            raise ToolParseError(f"Tool call was not closed with {TOOL_CALL_CLOSE}.")
        raise ToolParseError(
            f"No tool call found. Emit exactly one {TOOL_CALL_OPEN}...{TOOL_CALL_CLOSE} block "
            'containing JSON like {"name": "bash", "arguments": {"cmd": "ls -la"}}.'
        )

    try:
        payload = json.loads(found[0])
    except json.JSONDecodeError as exc:
        raise ToolParseError(f"Tool call was not valid JSON: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ToolParseError("Tool call JSON must be an object.")

    name = payload.get("name")
    if name not in allowed:
        raise ToolParseError(f"Unknown tool {name!r}. Available tools: {', '.join(allowed)}.")

    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ToolParseError("`arguments` must be an object.")
    if name == BASH and not arguments.get("cmd"):
        raise ToolParseError('A bash call needs a non-empty "cmd" argument.')

    return ToolCall(name=name, arguments={str(k): str(v) for k, v in arguments.items()})


def truncate(output: str, limit: int) -> str:
    """Shorten an observation for the context window.

    The middle goes rather than the tail: a failing test run puts the summary at
    the end, and that is the part driving the agent's next move. The full text
    is persisted untruncated regardless (D12), so no analysis depends on this.
    """
    if len(output) <= limit:
        return output
    head = limit // 2
    tail = limit - head
    omitted = len(output) - limit
    return f"{output[:head]}\n... [{omitted} characters omitted] ...\n{output[-tail:]}"
