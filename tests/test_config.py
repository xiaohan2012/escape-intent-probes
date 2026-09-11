from pathlib import Path

import pytest

from escape_probes import RunConfig


def test_defaults_match_the_decision_log():
    cfg = RunConfig(run_id="t")
    assert cfg.agent.prompt_level == "B"  # D13
    assert cfg.agent.tools == ("bash", "submit")  # Q12
    assert cfg.agent.max_steps == 25
    assert cfg.env.cue_strength == "explicit_path"  # D13
    assert cfg.model.temperature > 0  # sampling, not greedy


def test_exit_is_outside_the_authorized_area():
    cfg = RunConfig(run_id="t")
    assert not cfg.env.exit_dir.startswith(cfg.env.workdir)
    assert cfg.env.cue_path.startswith(cfg.env.workdir)


def test_trajectory_layout():
    cfg = RunConfig(run_id="full-01")
    got = cfg.trajectory_dir("django__django-16560", "impossible", 2)
    assert got == Path("runs/full-01/django__django-16560__impossible__2")


def test_n_trajectories():
    cfg = RunConfig(run_id="t", instance_ids=tuple(f"i{n}" for n in range(20)))
    assert cfg.n_trajectories == 160  # D10: 20 x 2 x 4


def test_roundtrip_through_yaml(tmp_path):
    import yaml

    cfg = RunConfig(run_id="t", instance_ids=("a", "b"))
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert RunConfig.from_yaml(path) == cfg


def test_shipped_pilot_config_parses():
    cfg = RunConfig.from_yaml("configs/pilot-01.yaml")
    assert cfg.run_id == "pilot-01"
    assert cfg.model.fake is True


def test_unknown_condition_rejected():
    # Via model_validate so the bad value is a runtime concern, not a static one.
    with pytest.raises(ValueError):
        RunConfig.model_validate({"run_id": "t", "conditions": ["sideways"]})
