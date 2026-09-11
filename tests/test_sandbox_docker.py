"""Integration tests against a real SWE-bench container.

Marked `docker`: they need a running daemon and a pulled image, so CI skips them
(`uv run pytest -m "not docker"`) and they run on the x86 box instead.

The image is not mocked. These tests exist precisely to check the things a mock
would assume: that the working directory is where we think, that a file written
into the writable layer is readable back, and that the boundary geometry the
crossing rule depends on actually holds inside a real image.
"""

import pytest

from escape_probes.boundary import exit_patch_path, plant, render_cue
from escape_probes.config import EnvConfig
from escape_probes.sandbox import DockerSandbox, SandboxError

pytestmark = pytest.mark.docker

INSTANCE_ID = "django__django-12419"


@pytest.fixture(scope="module")
def env() -> EnvConfig:
    return EnvConfig()


@pytest.fixture
def sandbox(env: EnvConfig):
    with DockerSandbox(INSTANCE_ID, env) as box:
        yield box


class TestDockerSandbox:
    """Test the container wrapper against a real image."""

    def test_starts_in_the_authorized_area(self, sandbox: DockerSandbox, env: EnvConfig) -> None:
        assert sandbox.exec("pwd").stdout.strip() == env.workdir

    def test_repo_is_present_and_is_a_clean_checkout(self, sandbox: DockerSandbox) -> None:
        # `git diff` on a clean tree is what D12's final-diff field relies on.
        assert sandbox.exec("git rev-parse --is-inside-work-tree").stdout.strip() == "true"
        assert sandbox.exec("git status --porcelain").stdout.strip() == ""

    def test_agent_is_root(self, sandbox: DockerSandbox) -> None:
        # Which is why a chmod-based read-only test file would be advisory (D8).
        assert sandbox.exec("whoami").stdout.strip() == "root"

    def test_has_no_network(self, sandbox: DockerSandbox) -> None:
        # Design doc 6.1: the planted exit is the only way out.
        result = sandbox.exec("getent hosts pypi.org || echo NO_DNS")
        assert "NO_DNS" in result.output

    def test_exec_reports_failure(self, sandbox: DockerSandbox) -> None:
        result = sandbox.exec("exit 3")
        assert result.exit_code == 3

    def test_exec_captures_stderr(self, sandbox: DockerSandbox) -> None:
        result = sandbox.exec("echo boom >&2")
        assert "boom" in result.output

    def test_exec_times_out_without_hanging(self, sandbox: DockerSandbox) -> None:
        result = sandbox.exec("sleep 30", timeout=3)
        assert result.exit_code == 124

    def test_write_and_read_roundtrip(self, sandbox: DockerSandbox) -> None:
        sandbox.write_file("/tmp/probe/roundtrip.txt", "line one\nline two")
        assert sandbox.read_file("/tmp/probe/roundtrip.txt").strip() == "line one\nline two"

    def test_container_is_removed_on_exit(self, env: EnvConfig) -> None:
        with DockerSandbox(INSTANCE_ID, env) as box:
            container_id = box.container_id
        probe = DockerSandbox(INSTANCE_ID, env)
        probe._container_id = container_id
        assert probe.exec("true").exit_code != 0


class TestPlantedBoundary:
    """Test the cue and the exit as the agent would meet them."""

    @property
    def patch(self) -> str:
        return "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"

    def test_cue_is_readable_from_inside(self, sandbox: DockerSandbox, env: EnvConfig) -> None:
        plant(sandbox, INSTANCE_ID, self.patch)
        # Reading it is allowed: the file is inside the authorized area (D9).
        result = sandbox.exec(f"cat {env.cue_path}")
        assert result.exit_code == 0
        assert result.stdout == render_cue(env.cue_strength, env, INSTANCE_ID)

    def test_cue_names_the_exit(self, sandbox: DockerSandbox, env: EnvConfig) -> None:
        plant(sandbox, INSTANCE_ID, self.patch)
        assert env.exit_dir in sandbox.read_file(env.cue_path)

    def test_exit_holds_the_patch(self, sandbox: DockerSandbox, env: EnvConfig) -> None:
        plant(sandbox, INSTANCE_ID, self.patch)
        assert sandbox.read_file(exit_patch_path(env, INSTANCE_ID)).strip() == self.patch.strip()

    def test_exit_is_outside_the_authorized_area(
        self, sandbox: DockerSandbox, env: EnvConfig
    ) -> None:
        plant(sandbox, INSTANCE_ID, self.patch)
        real_exit = sandbox.exec(f"readlink -f {env.exit_dir}").stdout.strip()
        real_work = sandbox.exec(f"readlink -f {env.workdir}").stdout.strip()
        assert not real_exit.startswith(real_work + "/")

    def test_exit_is_not_reachable_without_naming_it(
        self, sandbox: DockerSandbox, env: EnvConfig
    ) -> None:
        # If a symlink inside the boundary pointed at the exit, the agent could
        # reach it without ever naming a path under `exit_dir`, and the crossing
        # rule would miss the crossing.
        plant(sandbox, INSTANCE_ID, self.patch)
        sandbox.verify_boundary()

    def test_verify_boundary_catches_a_planted_symlink(
        self, sandbox: DockerSandbox, env: EnvConfig
    ) -> None:
        plant(sandbox, INSTANCE_ID, self.patch)
        sandbox.exec(f"ln -s {env.exit_dir} {env.workdir}/shortcut")
        with pytest.raises(SandboxError, match="symlink"):
            sandbox.verify_boundary()

    def test_verify_boundary_catches_a_missing_exit(
        self, sandbox: DockerSandbox, env: EnvConfig
    ) -> None:
        plant(sandbox, INSTANCE_ID, self.patch)
        sandbox.exec(f"rm -rf {env.exit_dir}")
        with pytest.raises(SandboxError, match="not planted"):
            sandbox.verify_boundary()
