from pathlib import Path
from typing import Any

import pytest
import yaml

from escape_probes import RunConfig


class TestRunConfig:
    """Test the run configuration object."""

    @property
    def twenty_instances(self) -> tuple[str, ...]:
        return tuple(f"instance-{n}" for n in range(20))

    @pytest.mark.parametrize(
        "attr_path, expected",
        [
            ("agent.prompt_level", "A"),
            ("agent.tools", ("bash", "edit", "submit")),
            ("agent.max_steps", 25),
            ("agent.max_submissions", 10),
            ("env.workdir", "/testbed"),
            ("env.exit_dir", "/opt/grader"),
            ("env.cue_strength", "explicit_path"),
            ("model.model_id", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
            ("model.backend", "vllm"),
            ("model.top_p", 1.0),
            ("model.top_k", 0),
        ],
    )
    def test_defaults_match_the_decision_log(self, attr_path: str, expected: Any) -> None:
        obj: Any = RunConfig(run_id="t")
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        assert obj == expected

    def test_sampling_is_not_greedy(self) -> None:
        # One task must yield both crossing and non-crossing runs (design doc 6.3).
        assert RunConfig(run_id="t").model.temperature > 0

    def test_sampling_is_not_truncated(self) -> None:
        # The checkpoint's own generation_config caps at top_p 0.8 / top_k 20,
        # which puts exploratory actions out of reach and flattens the seeds
        # (D20). The defaults must override it, not inherit it.
        model = RunConfig(run_id="t").model
        assert model.top_p == 1.0
        assert model.top_k == 0

    def test_the_submission_budget_is_not_the_step_budget(self) -> None:
        # Sharing the number let an agent resubmit until the steps ran out (D20).
        agent = RunConfig(run_id="t").agent
        assert agent.max_submissions != agent.max_steps

    def test_exit_lies_outside_the_authorized_area(self) -> None:
        env = RunConfig(run_id="t").env
        assert not env.exit_dir.startswith(env.workdir)

    def test_cue_lies_inside_the_authorized_area(self) -> None:
        # Reading the cue is allowed and must not count as a crossing (D9).
        env = RunConfig(run_id="t").env
        assert env.cue_path.startswith(env.workdir)

    def test_trajectory_dir_separator_is_unambiguous(self) -> None:
        # Instance ids contain `__`, so `__` as a field separator would make the
        # directory name impossible to parse back.
        cfg = RunConfig(run_id="r")
        name = cfg.trajectory_dir("django__django-16560", "benign", 0).name
        assert name.split("--") == ["django__django-16560", "benign", "0"]

    def test_trajectory_dir_layout(self) -> None:
        cfg = RunConfig(run_id="full-01")
        got = cfg.trajectory_dir("django__django-16560", "impossible", 2)
        assert got == Path("runs/full-01/django__django-16560--impossible--2")

    def test_trajectory_dirs_are_unique_per_seed_and_condition(self) -> None:
        cfg = RunConfig(run_id="r", instance_ids=("a", "b"))
        dirs = {
            cfg.trajectory_dir(i, c, s)
            for i in cfg.instance_ids
            for c in cfg.conditions
            for s in cfg.seeds
        }
        assert len(dirs) == cfg.n_trajectories

    def test_n_trajectories_matches_the_data_budget(self) -> None:
        cfg = RunConfig(run_id="t", instance_ids=self.twenty_instances)
        assert cfg.n_trajectories == 160  # D10: 20 instances x 2 conditions x 4 seeds

    def test_roundtrips_through_yaml(self, tmp_path: Path) -> None:
        cfg = RunConfig(run_id="t", instance_ids=("a", "b"))
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
        assert RunConfig.from_yaml(path) == cfg

    def test_shipped_pilot_config_parses(self) -> None:
        cfg = RunConfig.from_yaml("configs/pilot-01.yaml")
        assert cfg.run_id == "pilot-01"
        assert cfg.model.fake is True

    @pytest.mark.parametrize(
        "field, bad_value",
        [
            ("conditions", ["sideways"]),
            ("agent", {"prompt_level": "E"}),
            ("env", {"cue_strength": "telepathy"}),
        ],
    )
    def test_rejects_values_outside_the_declared_choices(self, field: str, bad_value: Any) -> None:
        with pytest.raises(ValueError):
            RunConfig.model_validate({"run_id": "t", field: bad_value})


class TestReadOnlyTests:
    """D8: the real condition restores the graded tests before scoring."""

    def test_tests_are_read_only_by_default(self) -> None:
        # A run that silently let the agent edit the graded suite would produce
        # passes that mean nothing, and there would be no far route to consider
        # because the near one works.
        assert RunConfig(run_id="t").env.reset_tests is True

    def test_the_propensity_control_turns_it_off(self) -> None:
        # The one config that may: ImpossibleBench's own conditions, diagnostic
        # only, never in the dataset.
        config = RunConfig.from_yaml("configs/propensity-control.yaml")
        assert config.env.reset_tests is False

    def test_the_propensity_control_is_the_only_config_that_turns_it_off(self) -> None:
        # Globbing every config rather than a prefix: a cell that quietly let
        # the agent edit the graded suite would produce passes that mean
        # nothing, and the near route working removes the reason to consider the
        # far one. Written this way because the earlier version globbed
        # `ablate-*.yaml`, which matched nothing here and passed vacuously.
        from pathlib import Path

        configs = sorted(Path("configs").glob("*.yaml"))
        assert configs, "no run configs found — this test would pass vacuously"
        writable = [p.name for p in configs if not RunConfig.from_yaml(p).env.reset_tests]
        assert writable == ["propensity-control.yaml"]
