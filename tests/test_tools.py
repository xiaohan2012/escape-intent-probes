import pytest

from escape_probes.model import render_tool_call
from escape_probes.tools import (
    BASH,
    SUBMIT,
    ToolCall,
    ToolParseError,
    parse_tool_call,
    truncate,
)


class TestParseToolCall:
    """Test reading a tool call out of a generation."""

    def test_reads_a_bash_call(self) -> None:
        call = parse_tool_call(render_tool_call(BASH, {"cmd": "ls -la"}))
        assert call == ToolCall(name=BASH, arguments={"cmd": "ls -la"})

    def test_reads_a_submit_call(self) -> None:
        assert parse_tool_call(render_tool_call(SUBMIT, {})).name == SUBMIT

    def test_ignores_reasoning_before_the_call(self) -> None:
        text = "Let me look around first.\n" + render_tool_call(BASH, {"cmd": "ls"})
        assert parse_tool_call(text).command == "ls"

    def test_takes_the_first_call_when_several_are_emitted(self) -> None:
        text = render_tool_call(BASH, {"cmd": "first"}) + render_tool_call(BASH, {"cmd": "second"})
        assert parse_tool_call(text).command == "first"

    def test_preserves_a_command_containing_quotes_and_newlines(self) -> None:
        command = "python -c 'print(\"hi\")'\ngrep -r 'x' ."
        assert parse_tool_call(render_tool_call(BASH, {"cmd": command})).command == command

    def test_coerces_argument_values_to_strings(self) -> None:
        # Models sometimes emit a number where a string belongs.
        call = parse_tool_call(
            '<tool_call>\n{"name": "bash", "arguments": {"cmd": 42}}\n</tool_call>'
        )
        assert call.command == "42"

    @pytest.mark.parametrize(
        "text, expected_message",
        [
            ("I think I should look around.", "No tool call found"),
            ('<tool_call>\n{"name": "bash"}', "not closed"),
            ("<tool_call>\nnot json\n</tool_call>", "not valid JSON"),
            ("<tool_call>\n[1, 2, 3]\n</tool_call>", "must be an object"),
            ('<tool_call>\n{"name": "rm_rf"}\n</tool_call>', "Unknown tool"),
            ('<tool_call>\n{"name": "bash", "arguments": "ls"}\n</tool_call>', "must be an object"),
            ('<tool_call>\n{"name": "bash", "arguments": {}}\n</tool_call>', "non-empty"),
        ],
        ids=[
            "no-call",
            "unclosed",
            "bad-json",
            "json-not-object",
            "unknown-tool",
            "arguments-not-object",
            "bash-without-command",
        ],
    )
    def test_rejects_malformed_calls(self, text: str, expected_message: str) -> None:
        with pytest.raises(ToolParseError, match=expected_message):
            parse_tool_call(text)

    def test_error_message_shows_the_expected_shape(self) -> None:
        # The message goes back to the model as an observation, so it has to say
        # what to do, not only what went wrong.
        with pytest.raises(ToolParseError) as excinfo:
            parse_tool_call("no call here")
        assert "<tool_call>" in str(excinfo.value)
        assert '"name": "bash"' in str(excinfo.value)

    def test_honours_a_restricted_tool_set(self) -> None:
        with pytest.raises(ToolParseError, match="Unknown tool"):
            parse_tool_call(render_tool_call(SUBMIT, {}), allowed=(BASH,))


class TestToolCall:
    """Test the parsed call."""

    def test_command_is_empty_for_a_non_bash_call(self) -> None:
        assert ToolCall(name=SUBMIT, arguments={}).command == ""


class TestTruncate:
    """Test shortening an observation for the context."""

    def test_leaves_short_output_alone(self) -> None:
        assert truncate("short", 100) == "short"

    def test_keeps_the_tail(self) -> None:
        # A failing test run puts its summary at the end, and that is what
        # drives the agent's next move.
        output = "x" * 500 + "FAILED test_foo"
        assert truncate(output, 100).endswith("FAILED test_foo")

    def test_keeps_the_head(self) -> None:
        output = "the command echoed this first\n" + "x" * 500
        assert truncate(output, 100).startswith("the command echoed")

    def test_says_how_much_it_dropped(self) -> None:
        assert "400 characters omitted" in truncate("x" * 500, 100)

    @pytest.mark.parametrize("limit", [10, 100, 1000])
    def test_result_is_only_slightly_longer_than_the_limit(self, limit: int) -> None:
        # The elision notice is the only overhead.
        result = truncate("x" * 5000, limit)
        assert len(result) <= limit + 60
