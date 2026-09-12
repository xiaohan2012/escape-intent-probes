"""The ReAct loop: one trajectory from first observation to outcome.

Deliberately small. Nothing here is novel — the loop is a `while`, a regex and a
`docker exec` — and the reason we write it rather than reuse AgentLens's or
ImpossibleBench's harness is that the project needs full control of the forward
pass to capture activations, which a harness that treats the model as a black
box does not give (D1).

Two behaviours are worth knowing about, because both shape the data:

*Parse failures are survivable.* A malformed tool call goes back as an
observation and the agent tries again. Only a run of consecutive failures ends
the trajectory. This keeps E1 readable: a model that cannot format a tool call
is not a model that cannot fix the bug, and the two must not be confused.

*Submission is a pressure loop.* A failed submission returns the test report and
restates the rules, so the agent learns that the route it tried is closed and
goes looking for another. Under a mutated test suite every legitimate route is
closed, which is exactly the pressure the experiment is about. The loop is
finite on purpose: `max_submissions` is separate from `max_steps` (D20), so
resubmitting an unchanged patch costs something and running out of attempts is
a distinct outcome.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from escape_probes.config import EDIT, SUBMIT, AgentConfig, Condition, RunConfig
from escape_probes.model import Generation, Message, ModelBackend
from escape_probes.prompts import failed_submission_prompt, retry_prompt, system_prompt
from escape_probes.sandbox import Sandbox
from escape_probes.tools import ToolParseError, apply_edit, parse_tool_call, truncate
from escape_probes.trace import Outcome, Trajectory, TrajectoryMeta, TrajectoryWriter


class Task(Protocol):
    """One instance: what to show the agent, how to set it up, how to score it."""

    instance_id: str
    problem_statement: str
    test_command: str

    def setup(self, sandbox: Sandbox) -> None:
        """Put the repository into its starting state, tests included."""

    def score(self, sandbox: Sandbox) -> tuple[bool, str]:
        """Run the graded tests, returning (passed, report).

        The implementation restores the original tests first, so editing them is
        futile (D8) — the agent may still try, and that attempt is recorded.
        """


def opening_message(task: Task, sandbox: Sandbox, agent: AgentConfig) -> str:
    """The problem statement, preceded by a listing of the working directory.

    The listing is a harness action rather than an agent step: it costs no step
    budget and appears in no trajectory, and it is identical in both conditions.
    See `AgentConfig.initial_listing_command` for why it exists at all.
    """
    if not agent.initial_listing_command:
        return task.problem_statement
    listing = sandbox.exec(agent.initial_listing_command)
    return (
        f"Contents of the working directory:\n\n{listing.output}\n\n"
        f"Here is the issue to fix:\n\n{task.problem_statement}"
    )


RolloutSteps = Generator[list[Message], Generation, Trajectory]
"""What `rollout_steps` is: it yields the conversation to generate from, is sent
back the `Generation`, and returns the finished `Trajectory`."""


def rollout(
    task: Task,
    model: ModelBackend,
    sandbox: Sandbox,
    config: RunConfig,
    condition: Condition,
    seed: int,
) -> Trajectory:
    """One trajectory, generating serially. The reference driver.

    Kept as the simple entry point because most things that read a trajectory
    want one trajectory: the smoke script, the docker tests, anything
    diagnosing a single run. `rollout_steps` plus `drive_batch` is the same
    logic with the generation lifted out, for when N trajectories should share
    one engine call (D21).
    """
    return drive_one(rollout_steps(task, sandbox, config, condition, seed), model)


def drive_one(steps: RolloutSteps, model: ModelBackend) -> Trajectory:
    """Pump one generator to its end, generating as it asks.

    Shared by the serial driver and the threaded one, so that "how a trajectory
    is advanced" has one definition. The threaded driver is otherwise nothing
    but this function in a thread pool.
    """
    try:
        messages = next(steps)
        while True:
            messages = steps.send(model.generate(messages))
    except StopIteration as finished:
        return finished.value


def rollout_steps(
    task: Task,
    sandbox: Sandbox,
    config: RunConfig,
    condition: Condition,
    seed: int,
) -> RolloutSteps:
    """The loop, with generation lifted out to the caller.

    A generator rather than a state-machine class, because every local variable
    here — the message list, the parse-error counter, the submission count, the
    writer — is per-trajectory state that the generator frame keeps for free.
    Writing it as a class would mean re-deriving that state as fields and
    getting the control flow right by hand, for no gain (D21).

    `generate_seconds` is now measured around the `yield`, so under a batching
    driver it includes any wait for the round's slowest sequence. That is the
    number worth having: it is what the barrier actually costs.
    """
    agent = config.agent
    prompt = system_prompt(agent, task.test_command)
    messages = [
        Message(role="system", content=prompt),
        Message(role="user", content=opening_message(task, sandbox, agent)),
    ]

    writer = TrajectoryWriter()
    outcome: Outcome = "max_steps"
    consecutive_parse_errors = 0
    submissions = 0
    started = time.monotonic()

    for _ in range(agent.max_steps):
        generate_started = time.monotonic()
        generation = yield messages
        generate_seconds = round(time.monotonic() - generate_started, 2)
        messages.append(Message(role="assistant", content=generation.text))

        try:
            call = parse_tool_call(generation.text, allowed=agent.tools)
        except ToolParseError as error:
            consecutive_parse_errors += 1
            observation = retry_prompt(str(error))
            writer.add_step(
                generation,
                parse_error=str(error),
                observation=observation,
                generate_seconds=generate_seconds,
            )
            messages.append(Message(role="user", content=observation))
            if consecutive_parse_errors > agent.max_parse_retries:
                outcome = "parse_failed"
                break
            continue

        consecutive_parse_errors = 0

        if call.name == SUBMIT:
            submissions += 1
            exec_started = time.monotonic()
            passed, report = task.score(sandbox)
            observation = report if passed else failed_submission_prompt(report, agent)
            writer.add_step(
                generation,
                tool_name=call.name,
                tool_arguments=call.arguments,
                observation=observation,
                generate_seconds=generate_seconds,
                exec_seconds=round(time.monotonic() - exec_started, 2),
            )
            if passed:
                outcome = "passed"
                break
            if submissions >= agent.max_submissions:
                outcome = "max_submissions"
                break
            messages.append(
                Message(role="user", content=truncate(observation, agent.max_observation_chars))
            )
            continue

        exec_started = time.monotonic()
        if call.name == EDIT:
            output, exit_code = apply_edit(call, sandbox), 0
        else:
            result = sandbox.exec(call.command)
            output, exit_code = result.output, result.exit_code
        writer.add_step(
            generation,
            tool_name=call.name,
            tool_arguments=call.arguments,
            observation=output,
            exit_code=exit_code,
            generate_seconds=generate_seconds,
            exec_seconds=round(time.monotonic() - exec_started, 2),
        )
        messages.append(Message(role="user", content=truncate(output, agent.max_observation_chars)))

    if outcome == "max_steps" and writer.steps and writer.steps[-1].tool_name == SUBMIT:
        outcome = "failed"

    n_parse_errors = sum(1 for step in writer.steps if step.parse_error)

    meta = TrajectoryMeta(
        run_id=config.run_id,
        instance_id=task.instance_id,
        condition=condition,
        seed=seed,
        model_id=config.model.model_id,
        backend=config.model.backend,
        temperature=config.model.temperature,
        prompt_level=agent.prompt_level,
        cue_strength=config.env.cue_strength,
        tools=agent.tools,
        image=sandbox.image,
        system_prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
        outcome=outcome,
        n_steps=len(writer.steps),
        n_parse_errors=n_parse_errors,
        wall_clock_seconds=round(time.monotonic() - started, 2),
        final_diff=sandbox.exec("git diff").stdout,
    )
    return Trajectory(meta=meta, steps=writer.steps, token_ids=writer.token_ids)


class BatchBackend(Protocol):
    """A backend that can generate for several conversations in one call."""

    def generate_batch(self, conversations: Sequence[Sequence[Message]]) -> list[Generation]: ...


class SerialBatch:
    """Adapts a one-at-a-time backend to the batch interface.

    So that there is one driver rather than two code paths. The fake model and
    the HuggingFace backend generate one conversation at a time; wrapping them
    means `drive_batch` is what every batch goes through, and a bug in the
    driver cannot hide behind a serial fallback that the tests exercise instead.
    """

    def __init__(self, model: ModelBackend) -> None:
        self.model = model

    def generate_batch(self, conversations: Sequence[Sequence[Message]]) -> list[Generation]:
        return [self.model.generate(conversation) for conversation in conversations]


def drive_batch(
    steps: Sequence[RolloutSteps],
    model: BatchBackend,
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[Trajectory | None]:
    """Advance N trajectories in lock step, one engine call per round.

    Each round collects the current conversation from every live trajectory and
    hands the whole list to `generate_batch`, so vLLM does its own continuous
    batching and there is no padding or ragged-length bookkeeping here.
    Trajectories leave the round as they finish, and the returned list is in the
    order given regardless of the order they finished in.

    A trajectory that raises — a container that died, a sandbox command that
    timed out — is closed and comes back as `None`, and the round carries on.
    The serial driver could afford to let an exception escape because it cost
    one trajectory; here it would cost the whole round, which is the opposite of
    why batching exists. `on_error` is how the caller reports it.

    The known cost is the barrier: a round waits for its slowest generation, and
    the sandbox commands of a round run while the GPU has nothing to do. Both
    are visible in the data — `generate_seconds` absorbs the first and
    `exec_seconds` the second — which is the point of measuring before replacing
    this with something that has no barrier (D21).
    """
    finished: dict[int, Trajectory] = {}
    pending: dict[int, list[Message]] = {}

    def advance(index: int, pump: Callable[[], list[Message]]) -> None:
        """One step of one trajectory, or its end. Never raises.

        `pump` is `next` on the first round and `send` afterwards; the caller
        supplies it so that the priming round and the steady state share this
        error handling rather than each having its own copy of it.
        """
        try:
            pending[index] = pump()
        except StopIteration as done:
            finished[index] = done.value
        except Exception as error:
            steps[index].close()
            if on_error is not None:
                on_error(index, error)

    for index, generator in enumerate(steps):
        advance(index, lambda g=generator: next(g))

    while pending:
        live = list(pending)
        generations = model.generate_batch([pending[index] for index in live])
        pending = {}
        for index, generation in zip(live, generations, strict=True):
            advance(index, lambda i=index, g=generation: steps[i].send(g))

    return [finished.get(index) for index in range(len(steps))]


def drive_threaded(
    steps: Sequence[RolloutSteps],
    model: ModelBackend,
    max_workers: int = 8,
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[Trajectory | None]:
    """Advance N trajectories concurrently, one request at a time each.

    The counterpart to `drive_batch` for a hosted endpoint (D22). There is no
    batch dimension to fill and therefore no reason to hold a barrier: the limit
    is the provider's rate limit, not a card, and lock step would make every
    trajectory wait for the round's slowest response for nothing. Each
    trajectory is simply `drive_one` in its own thread, which is also why the
    two drivers cannot disagree about how a trajectory advances.

    `max_workers` is the rate limit's knob. Exceeding a provider's limit returns
    429s, which arrive looking like a flaky model.

    Failures are handled as in `drive_batch`: a trajectory that raises comes back
    as `None` and the others continue. The threads are independent — each
    trajectory has its own generator, sandbox and container — so nothing here is
    shared except the backend, which must be safe to call from several threads.
    """
    finished: dict[int, Trajectory] = {}

    def run(index: int) -> None:
        try:
            finished[index] = drive_one(steps[index], model)
        except Exception as error:
            steps[index].close()
            if on_error is not None:
                on_error(index, error)

    if steps:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(run, range(len(steps))))

    return [finished.get(index) for index in range(len(steps))]


__all__ = [
    "BatchBackend",
    "SerialBatch",
    "RolloutSteps",
    "Task",
    "drive_batch",
    "drive_one",
    "drive_threaded",
    "opening_message",
    "rollout",
    "rollout_steps",
]
