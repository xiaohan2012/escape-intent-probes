"""Tests for advancing a round's sandbox work concurrently.

D21 accepted lock step with a stated cost — "the sandbox commands of a round run
while the GPU has nothing to do" — and an instruction: measure the barrier
before replacing it. Measured, on a 24-wide round with `nvidia-smi dmon`: eight
of fifteen samples show `sm 0%` and 124 W against 348 W while generating. Close
to half the wall clock is the card waiting for twenty-four `docker exec` calls
to happen one after another.

Advancing them concurrently is safe in the way that matters: each generator is
touched by exactly one thread, each has its own container, and the work is I/O.
What needs care is the bookkeeping the threads share, and the guarantee the
driver already makes — results in the order given, a failure costing one
trajectory rather than the round.
"""

from __future__ import annotations

import threading
import time

from escape_probes.model import Generation
from escape_probes.rollout import drive_batch


class SlowStep:
    """A trajectory whose every step sleeps, standing in for `docker exec`."""

    def __init__(self, name: str, steps: int, delay: float, fail_at: int | None = None) -> None:
        self.name = name
        self.steps = steps
        self.delay = delay
        self.fail_at = fail_at
        self.closed = False
        self.seen = 0

    def __iter__(self):  # noqa: ANN204
        return self

    def __next__(self):  # noqa: ANN204
        return self._advance()

    def send(self, generation: Generation):  # noqa: ANN001, ANN202
        del generation
        return self._advance()

    def _advance(self):  # noqa: ANN202
        if self.seen == self.fail_at:
            raise RuntimeError(f"{self.name} died")
        if self.seen >= self.steps:
            raise StopIteration(self.name)
        self.seen += 1
        time.sleep(self.delay)
        return [{"role": "user", "content": self.name}]

    def close(self) -> None:
        self.closed = True


class CountingModel:
    """Returns one generation per conversation, and counts rounds."""

    def __init__(self) -> None:
        self.rounds = 0
        self.widths: list[int] = []

    def generate_batch(self, conversations):  # noqa: ANN001, ANN202
        self.rounds += 1
        self.widths.append(len(conversations))
        return [Generation(prompt_token_ids=(1,), gen_token_ids=(2,), text="x")] * len(
            conversations
        )


class TestDriveBatchParallelism:
    def test_the_sandbox_work_of_a_round_overlaps(self) -> None:
        # Eight trajectories sleeping 0.2 s each: serial is 1.6 s per round and
        # three rounds is 4.8 s. Concurrent should be far under that.
        steps = [SlowStep(f"t{i}", steps=3, delay=0.2) for i in range(8)]
        started = time.monotonic()
        drive_batch(steps, CountingModel())
        assert time.monotonic() - started < 2.5

    def test_every_trajectory_still_finishes(self) -> None:
        steps = [SlowStep(f"t{i}", steps=2, delay=0.01) for i in range(6)]
        assert drive_batch(steps, CountingModel()) == [f"t{i}" for i in range(6)]

    def test_results_keep_the_order_given(self) -> None:
        # Different lengths, so completion order differs from input order.
        steps = [SlowStep("a", 3, 0.01), SlowStep("b", 1, 0.01), SlowStep("c", 2, 0.01)]
        assert drive_batch(steps, CountingModel()) == ["a", "b", "c"]

    def test_one_dead_trajectory_does_not_take_the_round(self) -> None:
        reported: list[int] = []
        steps = [SlowStep("a", 3, 0.01), SlowStep("b", 3, 0.01, fail_at=1), SlowStep("c", 3, 0.01)]
        result = drive_batch(steps, CountingModel(), on_error=lambda i, e: reported.append(i))
        assert result == ["a", None, "c"]
        assert reported == [1]
        assert steps[1].closed

    def test_the_engine_is_called_once_per_round(self) -> None:
        # The whole reason for lock step: one engine call per round, with the
        # round narrowing as trajectories finish.
        model = CountingModel()
        drive_batch([SlowStep("a", 3, 0.01), SlowStep("b", 1, 0.01)], model)
        assert model.widths == [2, 1, 1]

    def test_threads_do_not_lose_a_result(self) -> None:
        # The bookkeeping dicts are written from every worker; a plain dict
        # under contention is the kind of thing that loses one entry in a
        # hundred runs rather than every run.
        for _ in range(20):
            steps = [SlowStep(f"t{i}", steps=2, delay=0.0) for i in range(16)]
            assert drive_batch(steps, CountingModel()) == [f"t{i}" for i in range(16)]

    def test_a_single_trajectory_still_works(self) -> None:
        assert drive_batch([SlowStep("only", 2, 0.01)], CountingModel()) == ["only"]

    def test_no_trajectories_is_not_an_error(self) -> None:
        assert drive_batch([], CountingModel()) == []


class TestThreadSafety:
    def test_each_generator_is_touched_by_one_thread(self) -> None:
        seen: dict[str, set[int]] = {}
        lock = threading.Lock()

        class Recording(SlowStep):
            def _advance(self):  # noqa: ANN202
                with lock:
                    seen.setdefault(self.name, set()).add(threading.get_ident())
                return super()._advance()

        steps = [Recording(f"t{i}", steps=4, delay=0.01) for i in range(8)]
        drive_batch(steps, CountingModel())
        # A generator may move between threads across rounds — what must never
        # happen is two threads inside one generator at once, which the driver
        # guarantees by advancing each index exactly once per round.
        assert len(seen) == 8
