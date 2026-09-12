"""Tests for the backend factory.

Neither real backend can be constructed without weights, so what is checkable
here is the dispatch and the failure mode. That is worth a test anyway: the
factory exists because two entry scripts had diverging copies of this choice,
and a silent disagreement about the default is exactly the bug it removes.
"""

import pytest

from escape_probes.backends import build_model
from escape_probes.config import ModelConfig


class TestBuildModel:
    def test_the_default_is_the_rollout_path(self) -> None:
        # vLLM, per D21. A default of `hf` would silently give back the 16 tok/s
        # path, and the speed work would look like it had not landed.
        assert ModelConfig().backend == "vllm"

    def test_the_config_layer_rejects_an_unknown_backend(self) -> None:
        # First line of defence: `Backend` is a Literal, so a typo in a YAML run
        # config fails at load rather than after the images are pulled.
        with pytest.raises(Exception, match="backend"):
            ModelConfig.model_validate({"backend": "trt"})

    def test_the_factory_refuses_one_too(self) -> None:
        # Second line: `model_construct` skips validation, which is how a value
        # could reach the factory anyway. It must raise rather than fall through
        # to `None`, and it must raise *before* importing anything — otherwise a
        # typo on a fresh box reports a missing dependency instead.
        config = ModelConfig.model_construct(backend="trt")
        with pytest.raises(ValueError, match="unknown backend"):
            build_model(config, tools=("bash",))

    def test_whitespace_is_not_quietly_accepted(self) -> None:
        config = ModelConfig.model_construct(backend="hf ")
        with pytest.raises(ValueError, match="unknown backend"):
            build_model(config, tools=("bash",))
