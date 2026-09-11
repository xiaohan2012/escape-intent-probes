import pytest

from escape_probes.model import render_tool_call
from escape_probes.tools import (
    BASH,
    EDIT,
    SUBMIT,
    ToolCall,
    ToolParseError,
    apply_edit,
    parse_tool_call,
    truncate,
)

QWEN_SAMPLE = (
    "I'll help implement the requested feature. Let me first explore the repository "
    "structure.\n"
    "<tool_call>\n"
    "<function=bash>\n"
    "<parameter=cmd>\n"
    'find . -type f -name "*.py" | grep -E "(settings|middleware)" | head -20\n'
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
"""Captured verbatim from Qwen3-Coder-30B-A3B on the first real trajectory. The
first implementation parsed JSON only and every step failed, which makes E1
measure formatting rather than capability (D15)."""


class TestParseToolCall:
    """Test reading a tool call out of a generation."""

    def test_reads_the_format_the_model_actually_emits(self) -> None:
        call = parse_tool_call(QWEN_SAMPLE)
        assert call.name == BASH
        assert call.command.startswith("find . -type f")

    def test_preserves_shell_metacharacters(self) -> None:
        # Pipes and quotes inside the command must survive the tag form.
        assert "|" in parse_tool_call(QWEN_SAMPLE).command
        assert '"*.py"' in parse_tool_call(QWEN_SAMPLE).command

    def test_reads_json_from_other_model_families(self) -> None:
        text = '<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls -la"}}\n</tool_call>'
        assert parse_tool_call(text).command == "ls -la"

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

    def test_offset_points_at_the_command_in_the_real_sample(self) -> None:
        from escape_probes.model import command_char_offset

        offset = command_char_offset(QWEN_SAMPLE)
        assert offset is not None
        assert QWEN_SAMPLE[offset:].startswith("find . -type f")

    @pytest.mark.parametrize(
        "text, expected_message",
        [
            ("I think I should look around.", "No tool call found"),
            ('<tool_call>\n{"name": "bash"}', "not closed"),
            ("<tool_call>\nnot a call at all\n</tool_call>", "could not be read"),
            ("<tool_call>\n[1, 2, 3]\n</tool_call>", "could not be read"),
            ('<tool_call>\n{"name": "rm_rf"}\n</tool_call>', "Unknown tool"),
            ("<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>", "Unknown tool"),
            ("<tool_call>\n<function=bash>\n</function>\n</tool_call>", "non-empty"),
        ],
        ids=[
            "no-call",
            "unclosed",
            "unreadable",
            "json-not-object",
            "unknown-tool-json",
            "unknown-tool-tags",
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
        assert "<function=bash>" in str(excinfo.value)

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


class TestApplyEdit:
    """Test the edit tool — the first remedy if E1 fails (Q12)."""

    @property
    def content(self) -> str:
        return "def f(x):\n    return x + 1\n\ndef g(y):\n    return y\n"

    def sandbox(self, content: str | None = None):
        from tests.test_rollout import FakeSandbox

        return FakeSandbox({"cat /testbed/m.py": self.content if content is None else content})

    def call(self, **arguments: str) -> ToolCall:
        return ToolCall(name=EDIT, arguments={"path": "/testbed/m.py", **arguments})

    def test_replaces_an_exact_match(self) -> None:
        sandbox = self.sandbox()
        result = apply_edit(self.call(old="return x + 1", new="return x + 2"), sandbox)
        assert "Edited" in result
        assert any("return x + 2" in command for command in sandbox.commands)

    def test_refuses_when_nothing_matches(self) -> None:
        # A mistaken assumption about the file, not an edit to guess at.
        result = apply_edit(self.call(old="return z", new="return 0"), self.sandbox())
        assert "No match" in result

    def test_refuses_an_ambiguous_match(self) -> None:
        # Applying one of two would show up much later as an unexplained failure.
        sandbox = self.sandbox("return x\nreturn x\n")
        result = apply_edit(self.call(old="return x", new="return y"), sandbox)
        assert "appears 2 times" in result

    def test_keeps_the_rest_of_the_file(self) -> None:
        sandbox = self.sandbox()
        apply_edit(self.call(old="return x + 1", new="return x + 2"), sandbox)
        written = next(c for c in sandbox.commands if "cat >" in c)
        assert "def g(y):" in written


class TestEditParsing:
    """Test validation of an edit call."""

    def rendered(self, **arguments: str) -> str:
        return render_tool_call(EDIT, arguments)

    def test_accepts_a_complete_call(self) -> None:
        text = self.rendered(path="m.py", old="a", new="b")
        call = parse_tool_call(text, allowed=(BASH, SUBMIT, EDIT))
        assert call.arguments == {"path": "m.py", "old": "a", "new": "b"}

    @pytest.mark.parametrize("missing", ["path", "old", "new"])
    def test_rejects_an_incomplete_call(self, missing: str) -> None:
        arguments = {"path": "m.py", "old": "a", "new": "b"}
        del arguments[missing]
        with pytest.raises(ToolParseError, match=missing):
            parse_tool_call(self.rendered(**arguments), allowed=(BASH, SUBMIT, EDIT))

    def test_is_unavailable_unless_enabled(self) -> None:
        # Not in the default tool set: it is a remedy, not a starting point.
        with pytest.raises(ToolParseError, match="Unknown tool"):
            parse_tool_call(self.rendered(path="m.py", old="a", new="b"))
