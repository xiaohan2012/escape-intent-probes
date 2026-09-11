"""Tests for SWE-bench instance setup and scoring.

`swebench` and `datasets` are optional (the `tasks` group), so the tests that
need them are skipped rather than failing a core install. Everything that can be
checked without them — which patch each condition selects, which files a patch
touches, what a failed submission tells the agent — is checked unconditionally.
"""

from typing import Any

import pytest

from escape_probes.sandbox import ExecResult
from escape_probes.tasks import (
    MUTATIONS,
    SweBenchTask,
    TaskError,
    _failure_report,
    _test_files,
    load_instances,
)

swebench = pytest.importorskip("swebench", reason="needs the `tasks` dependency group")


class RecordingSandbox:
    """Records commands and returns scripted results."""

    image = "swebench/sweb.eval.x86_64.fake:latest"

    def __init__(self, failures: dict[str, int] | None = None, output: str = "") -> None:
        self.commands: list[str] = []
        self.failures = failures or {}
        self.output = output

    def exec(self, command: str, timeout: int = 120, workdir: str | None = None) -> ExecResult:
        self.commands.append(command)
        for fragment, code in self.failures.items():
            if fragment in command:
                return ExecResult(stdout="", stderr=f"{fragment} failed", exit_code=code)
        return ExecResult(stdout=self.output, stderr="", exit_code=0)

    def ran(self, fragment: str) -> bool:
        return any(fragment in command for command in self.commands)

    def index_of(self, fragment: str) -> int:
        return next(i for i, c in enumerate(self.commands) if fragment in c)


TEST_PATCH = """\
diff --git a/tests/test_thing.py b/tests/test_thing.py
--- a/tests/test_thing.py
+++ b/tests/test_thing.py
@@ -1 +1 @@
-assert x == "right"
+assert x == "wrong"
"""

ORIGINAL_TEST_PATCH = TEST_PATCH.replace('"wrong"', '"right"')


@pytest.fixture
def row() -> dict[str, Any]:
    return {
        "instance_id": "django__django-12419",
        "repo": "django/django",
        "version": "3.1",
        "base_commit": "abc123",
        "patch": "diff --git a/django/x.py b/django/x.py\n",
        "test_patch": TEST_PATCH,
        "original_test_patch": ORIGINAL_TEST_PATCH,
        "problem_statement": "Something is broken.",
        "FAIL_TO_PASS": ["test_thing"],
        "PASS_TO_PASS": ["test_other"],
    }


class TestTestFiles:
    """Test reading the touched files out of a patch."""

    def test_finds_the_files(self) -> None:
        assert _test_files(TEST_PATCH) == ["tests/test_thing.py"]

    def test_finds_several(self) -> None:
        patch = TEST_PATCH + TEST_PATCH.replace("test_thing", "test_other")
        assert _test_files(patch) == ["tests/test_thing.py", "tests/test_other.py"]

    def test_empty_patch_touches_nothing(self) -> None:
        assert _test_files("") == []


class TestConditionSelectsTheTests:
    """Test the one thing that differs between conditions (design doc 6.2)."""

    def test_benign_uses_the_original_tests(self, row: dict[str, Any]) -> None:
        assert SweBenchTask(row=row, condition="benign").test_patch == ORIGINAL_TEST_PATCH

    def test_impossible_uses_the_mutated_tests(self, row: dict[str, Any]) -> None:
        assert SweBenchTask(row=row, condition="impossible").test_patch == TEST_PATCH

    def test_everything_else_is_shared(self, row: dict[str, Any]) -> None:
        benign = SweBenchTask(row=row, condition="benign")
        impossible = SweBenchTask(row=row, condition="impossible")
        assert benign.instance_id == impossible.instance_id
        assert benign.problem_statement == impossible.problem_statement
        assert benign.gold_patch == impossible.gold_patch

    def test_test_command_comes_from_swebench(self, row: dict[str, Any]) -> None:
        # Guessing this per repo is how scoring goes silently wrong.
        command = SweBenchTask(row=row, condition="benign").test_command
        assert "runtests.py" in command


class TestSetup:
    """Test putting the repository into its starting state."""

    @property
    def condition(self) -> str:
        return "impossible"

    def task(self, row: dict[str, Any]) -> SweBenchTask:
        return SweBenchTask(row=row, condition="impossible")

    def test_resets_to_the_base_commit_first(self, row: dict[str, Any]) -> None:
        sandbox = RecordingSandbox()
        self.task(row).setup(sandbox)
        assert sandbox.index_of("git reset --hard abc123") == 0

    def test_applies_the_condition_tests(self, row: dict[str, Any]) -> None:
        sandbox = RecordingSandbox()
        self.task(row).setup(sandbox)
        assert sandbox.ran("git apply")

    def test_commits_the_tests(self, row: dict[str, Any]) -> None:
        # Otherwise the injected tests would show up in the agent's own diff.
        sandbox = RecordingSandbox()
        self.task(row).setup(sandbox)
        assert sandbox.index_of("commit") > sandbox.index_of("git apply")

    @pytest.mark.parametrize(
        "failing_command",
        ["git reset", "git checkout", "git apply", "commit"],
    )
    def test_reports_which_step_failed(self, row: dict[str, Any], failing_command: str) -> None:
        sandbox = RecordingSandbox(failures={failing_command: 1})
        with pytest.raises(TaskError, match="django__django-12419"):
            self.task(row).setup(sandbox)

    def test_rejects_a_patch_touching_no_files(self, row: dict[str, Any]) -> None:
        row["test_patch"] = "not a patch"
        with pytest.raises(TaskError, match="touches no files"):
            self.task(row).setup(RecordingSandbox())


class TestScore:
    """Test the graded run."""

    def task(self, row: dict[str, Any]) -> SweBenchTask:
        return SweBenchTask(row=row, condition="impossible")

    def test_restores_the_tests_before_running_them(self, row: dict[str, Any]) -> None:
        # This is what makes the tests read-only (D8): an agent may edit them,
        # and the edit simply never reaches the grader.
        sandbox = RecordingSandbox()
        self.task(row).score(sandbox)
        assert sandbox.index_of("git checkout HEAD --") < sandbox.index_of("runtests.py")

    def test_a_crashed_run_does_not_pass(self, row: dict[str, Any]) -> None:
        # An unparsable log means no evidence, which must not read as success.
        passed, _ = self.task(row).score(RecordingSandbox(output="Segmentation fault"))
        assert not passed


class TestFailureReport:
    """Test what a failed submission tells the agent."""

    def test_names_the_tests_still_failing(self) -> None:
        report = _failure_report("log output", ["test_a"], [])
        assert "test_a" in report

    def test_distinguishes_regressions(
        self,
    ) -> None:
        report = _failure_report("log output", [], ["test_b"])
        assert "Regressed" in report

    def test_puts_the_summary_last(self) -> None:
        # The loop truncates the middle, so the summary has to be at an end.
        report = _failure_report("log output", ["test_a"], [])
        assert report.strip().endswith("test_a")

    def test_caps_how_many_names_it_lists(self) -> None:
        report = _failure_report("log", [f"test_{n}" for n in range(50)], [])
        assert report.count("test_") <= 11


class TestLoadInstances:
    """Test dataset access."""

    def test_rejects_an_unknown_split(self) -> None:
        with pytest.raises(TaskError, match="unknown split"):
            load_instances("original")

    def test_mutations_are_the_two_impossible_splits(self) -> None:
        assert set(MUTATIONS) == {"conflicting", "oneoff"}
