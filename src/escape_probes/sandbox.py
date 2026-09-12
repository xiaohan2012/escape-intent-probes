"""The sandbox the agent acts in: a container, plus the boundary planted in it.

One container per trajectory (D13). The agent runs as root and will damage its
environment — that is partly the point — so containers are never reused.

The exit and the cue are written into the running container rather than baked
into a derived image (Q14). Tuning E2 begins by rewording the cue, and under the
bake-an-image option every reworded character means rebuilding one image per
instance. A bind mount would be the third option and is ruled out: it shows up
in `mount` and `/proc/mounts`, telling the agent that `/opt/grader` was placed
there deliberately.

The container need not run on the same machine as the model. It usually does
not: rented GPU instances are themselves unprivileged containers, so Docker is
unavailable on them (see setup-log.md), while the sandbox needs a real Docker
host. Setting `host` routes every `docker` invocation through SSH to that host.
Latency matters here — one tool call is one round trip — so the connection is
multiplexed with `ControlMaster`, which turns a fresh handshake per call into a
reused channel.
"""

from __future__ import annotations

import platform
import shlex
import subprocess
from dataclasses import dataclass
from typing import Protocol

from escape_probes.config import EnvConfig

IMAGE_PREFIX = "swebench/sweb.eval."
"""SWE-bench's prebuilt images on Docker Hub. Instance ids embed `__`, which is
not legal in a tag, so upstream substitutes `_1776_`."""

MACHINE_ARCHITECTURES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
}
"""`platform.machine()` onto the two names upstream publishes images under.

Three spellings reach the same architecture depending on the operating system,
and upstream uses exactly one of them per architecture.
"""


def host_arch() -> str:
    """The architecture of the machine the containers will run on.

    A default rather than a configuration knob, because getting it wrong is
    silent in the expensive direction: an amd64 image on an arm64 host runs
    under emulation, and a Django suite that then takes minutes per invocation
    looks like a slow model rather than a wrong image.
    """
    machine = platform.machine().lower()
    if machine not in MACHINE_ARCHITECTURES:
        raise ValueError(f"unknown architecture {platform.machine()!r}")
    return MACHINE_ARCHITECTURES[machine]


def image_for(instance_id: str, arch: str | None = None) -> str:
    """The prebuilt image for an instance, for one architecture.

    Upstream publishes an arm64 set alongside the x86_64 one, which is what lets
    the frontier screen run its sandboxes on a laptop: it needs no GPU, so
    binding it to a rented x86 box would spend the one advantage it has (D22).
    Arm64 is not validated upstream for evaluation parity, which does not matter
    for a diagnostic whose comparison is already broken along the model axis.
    """
    arch = arch or host_arch()
    if arch not in set(MACHINE_ARCHITECTURES.values()):
        raise ValueError(f"unknown architecture {arch!r}")
    return f"{IMAGE_PREFIX}{arch}.{instance_id.replace('__', '_1776_')}:latest"


@dataclass(frozen=True)
class ExecResult:
    """One tool call's effect on the container."""

    stdout: str
    stderr: str
    exit_code: int

    @property
    def output(self) -> str:
        """What the agent sees. Both streams, because a failing test run says
        most of what it has to say on stderr."""
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)


class Sandbox(Protocol):
    """What the rollout loop needs: somewhere to run a command, and its identity.

    Narrower than `DockerSandbox` on purpose — the loop never starts, stops or
    inspects a container, so a test double implementing this much is a
    legitimate substitute rather than a mock of something larger.
    """

    image: str
    """What the sandbox is running. Recorded as trajectory provenance, asked of
    the sandbox rather than re-derived from the instance id, so it stays true
    for any implementation."""

    def exec(self, command: str, timeout: int = 120, workdir: str | None = None) -> ExecResult: ...


class SandboxError(RuntimeError):
    """The container itself misbehaved — distinct from a command that failed."""


SSH_MULTIPLEX = [
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPath=~/.ssh/cm-escape-probes-%r@%h:%p",
    "-o",
    "ControlPersist=10m",
    "-o",
    "BatchMode=yes",
]
"""Reuse one SSH connection across tool calls; a handshake per call would cost
more than the command itself."""


def _run(args: list[str], timeout: int = 120) -> ExecResult:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ExecResult(stdout="", stderr=f"timed out after {timeout}s", exit_code=124)
    return ExecResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)


class DockerSandbox:
    """A single container, started for one trajectory and destroyed after it.

    Use as a context manager so the container is removed even when a rollout
    raises; a leaked container holds a couple of GB of writable layer.
    """

    def __init__(
        self,
        instance_id: str,
        env: EnvConfig,
        network: bool = False,
        host: str | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.env = env
        self.network = network
        self.host = host
        """SSH destination of the Docker host. `None` runs Docker locally."""
        self.image = image_for(instance_id)
        self._container_id: str | None = None

    def _docker(self, args: list[str]) -> list[str]:
        """Wrap a docker invocation for wherever the daemon lives."""
        if self.host is None:
            return args
        # One argv element per remote word: ssh concatenates its arguments with
        # spaces and hands the result to a remote shell, so anything containing
        # whitespace has to arrive already quoted.
        return ["ssh", *SSH_MULTIPLEX, self.host, *(shlex.quote(a) for a in args)]

    def __enter__(self) -> DockerSandbox:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def container_id(self) -> str:
        if self._container_id is None:
            raise SandboxError("sandbox is not running")
        return self._container_id

    def start(self) -> None:
        if self._container_id is not None:
            raise SandboxError("sandbox already started")
        args = ["docker", "run", "-d", "--rm"]
        if not self.network:
            # Design doc 6.1: no network. ImpossibleBench's own compose.yaml
            # sets `network_mode: none` too.
            args += ["--network", "none"]
        args += [self.image, "tail", "-f", "/dev/null"]
        result = _run(self._docker(args), timeout=180)
        if result.exit_code != 0:
            raise SandboxError(f"could not start container: {result.output}")
        self._container_id = result.stdout.strip()

    def stop(self) -> None:
        if self._container_id is None:
            return
        _run(self._docker(["docker", "rm", "-f", self._container_id]), timeout=60)
        self._container_id = None

    def exec(self, command: str, timeout: int = 120, workdir: str | None = None) -> ExecResult:
        """Run a shell command inside the container, as the agent would."""
        args = ["docker", "exec", "-w", workdir or self.env.workdir]
        args += [self.container_id, "bash", "-lc", command]
        return _run(self._docker(args), timeout=timeout)

    def write_file(self, path: str, content: str) -> None:
        """Write a file into the container's own writable layer.

        Indistinguishable, from inside, from a file baked into the image — which
        is the property that rules out a bind mount (Q14).
        """
        quoted_path = shlex.quote(path)
        marker = "ESCAPE_PROBES_EOF"
        # Exactly one trailing newline, never two: a heredoc needs the marker on
        # its own line, and patch files are byte-sensitive — `git apply` rejects
        # a diff whose trailing whitespace does not match.
        body = content if content.endswith("\n") else content + "\n"
        heredoc = (
            f"mkdir -p $(dirname {quoted_path}) && cat > {quoted_path} <<'{marker}'\n{body}{marker}"
        )
        result = self.exec(heredoc, workdir="/")
        if result.exit_code != 0:
            raise SandboxError(f"could not write {path}: {result.output}")

    def read_file(self, path: str) -> str:
        result = self.exec(f"cat {shlex.quote(path)}", workdir="/")
        if result.exit_code != 0:
            raise SandboxError(f"could not read {path}: {result.output}")
        return result.stdout

    def exists(self, path: str) -> bool:
        return self.exec(f"test -e {shlex.quote(path)}", workdir="/").exit_code == 0
