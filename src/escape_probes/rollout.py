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

from escape_probes.config import Condition, RunConfig
from escape_probes.model import Message, ModelBackend
from escape_probes.prompts import failed_submission_prompt, retry_prompt, system_prompt
from escape_probes.sandbox import Sandbox, image_for
from escape_probes.tools import BASH, SUBMIT, ToolParseError, parse_tool_call, truncate
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


def rollout(
    task: Task,
    model: ModelBackend,
    sandbox: Sandbox,
    config: RunConfig,
    condition: Condition,
    seed: int,
) -> Trajectory:
    agent = config.agent
    prompt = system_prompt(agent, task.test_command, max_attempts=agent.max_steps)
    messages = [
        Message(role="system", content=prompt),
        Message(role="user", content=task.problem_statement),
    ]

    writer = TrajectoryWriter()
    outcome: Outcome = "max_steps"
    consecutive_parse_errors = 0
    n_parse_errors = 0
    started = time.monotonic()

    for _ in range(agent.max_steps):
        generation = model.generate(messages)
        messages.append(Message(role="assistant", content=generation.text))

        try:
            call = parse_tool_call(generation.text, allowed=agent.tools)
        except ToolParseError as error:
            n_parse_errors += 1
            consecutive_parse_errors += 1
            observation = retry_prompt(str(error))
            writer.add_step(
                generation.prompt_token_ids,
                generation.gen_token_ids,
                generation.tool_start_token_idx,
                parse_error=str(error),
                observation=observation,
            )
            messages.append(Message(role="user", content=observation))
            if consecutive_parse_errors > agent.max_parse_retries:
                outcome = "parse_failed"
                break
            continue

        consecutive_parse_errors = 0

        if call.name == SUBMIT:
            passed, report = task.score(sandbox)
            observation = report if passed else failed_submission_prompt(report, agent)
            writer.add_step(
                generation.prompt_token_ids,
                generation.gen_token_ids,
                generation.tool_start_token_idx,
                tool_name=call.name,
                tool_arguments=call.arguments,
                observation=observation,
            )
            if passed:
                outcome = "passed"
                break
            messages.append(
                Message(role="user", content=truncate(observation, agent.max_observation_chars))
            )
            continue

        result = sandbox.exec(call.command)
        writer.add_step(
            generation.prompt_token_ids,
            generation.gen_token_ids,
            generation.tool_start_token_idx,
            tool_name=call.name,
            tool_arguments=call.arguments,
            observation=result.output,
            exit_code=result.exit_code,
        )
        messages.append(
            Message(role="user", content=truncate(result.output, agent.max_observation_chars))
        )

    if outcome == "max_steps" and writer.steps and writer.steps[-1].tool_name == SUBMIT:
        outcome = "failed"

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
        image=image_for(task.instance_id),
        system_prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
        outcome=outcome,
        n_steps=len(writer.steps),
        n_parse_errors=n_parse_errors,
        wall_clock_seconds=round(time.monotonic() - started, 2),
        final_diff=sandbox.exec("git diff").stdout,
    )
    return Trajectory(meta=meta, steps=writer.steps, token_ids=writer.token_ids)


__all__ = ["BASH", "Task", "rollout"]
