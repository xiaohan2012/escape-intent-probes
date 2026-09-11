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
closed, which is exactly the pressure the experiment is about.
"""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

from escape_probes.config import SUBMIT, AgentConfig, Condition, RunConfig
from escape_probes.model import Message, ModelBackend
from escape_probes.prompts import failed_submission_prompt, retry_prompt, system_prompt
from escape_probes.sandbox import Sandbox
from escape_probes.tools import ToolParseError, parse_tool_call, truncate
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


def rollout(
    task: Task,
    model: ModelBackend,
    sandbox: Sandbox,
    config: RunConfig,
    condition: Condition,
    seed: int,
) -> Trajectory:
    agent = config.agent
    prompt = system_prompt(agent, task.test_command)
    messages = [
        Message(role="system", content=prompt),
        Message(role="user", content=opening_message(task, sandbox, agent)),
    ]

    writer = TrajectoryWriter()
    outcome: Outcome = "max_steps"
    consecutive_parse_errors = 0
    started = time.monotonic()

    for _ in range(agent.max_steps):
        generate_started = time.monotonic()
        generation = model.generate(messages)
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
            messages.append(
                Message(role="user", content=truncate(observation, agent.max_observation_chars))
            )
            continue

        exec_started = time.monotonic()
        result = sandbox.exec(call.command)
        writer.add_step(
            generation,
            tool_name=call.name,
            tool_arguments=call.arguments,
            observation=result.output,
            exit_code=result.exit_code,
            generate_seconds=generate_seconds,
            exec_seconds=round(time.monotonic() - exec_started, 2),
        )
        messages.append(
            Message(role="user", content=truncate(result.output, agent.max_observation_chars))
        )

    if outcome == "max_steps" and writer.steps and writer.steps[-1].tool_name == SUBMIT:
        outcome = "failed"

    n_parse_errors = sum(1 for step in writer.steps if step.parse_error)

    meta = TrajectoryMeta(
        run_id=config.run_id,
        instance_id=task.instance_id,
        condition=condition,
        seed=seed,
        model_id=config.model.model_id,
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


__all__ = ["Task", "opening_message", "rollout"]
