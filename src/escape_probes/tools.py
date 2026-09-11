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

from dataclasses import dataclass

from escape_probes.config import BASH, DEFAULT_TOOLS, SUBMIT
from escape_probes.model import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    decode_payload,
    find_tool_payload,
)

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
            "containing <function=bash><parameter=cmd>ls -la</parameter></function>."
        )

    call = decode_payload(*found)
    if call is None:
        raise ToolParseError(
            "Tool call could not be read. Use the form "
            "<function=bash><parameter=cmd>ls -la</parameter></function>."
        )
    if call.name not in allowed:
        raise ToolParseError(f"Unknown tool {call.name!r}. Available tools: {', '.join(allowed)}.")
    if call.name == BASH and not call.arguments.get("cmd"):
        raise ToolParseError('A bash call needs a non-empty "cmd" argument.')

    return ToolCall(name=call.name, arguments=call.arguments)


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
