"""Escape-intent probes: does a linear probe flag boundary-crossing intent early?"""

from escape_probes.config import RunConfig
from escape_probes.labels import Labels, label
from escape_probes.model import FakeModel, Generation, Message, ModelBackend, ScriptedStep
from escape_probes.rollout import Task, rollout
from escape_probes.sandbox import DockerSandbox, ExecResult, SandboxError
from escape_probes.trace import Trajectory, TrajectoryMeta

__all__ = [
    "DockerSandbox",
    "Labels",
    "Task",
    "Trajectory",
    "TrajectoryMeta",
    "label",
    "rollout",
    "ExecResult",
    "FakeModel",
    "Generation",
    "Message",
    "ModelBackend",
    "RunConfig",
    "SandboxError",
    "ScriptedStep",
]
