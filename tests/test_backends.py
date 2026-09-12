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
    def test_an_unknown_backend_is_refused(self) -> None:
        config = ModelConfig()
        config.backend = "trt"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="unknown backend"):
            build_model(config, tools=("bash",))

    def test_the_default_is_the_rollout_path(self) -> None:
        # vLLM, per D21. A default of `hf` would silently give back the 16 tok/s
        # path and the speed work would look like it had not landed.
        assert ModelConfig().backend == "vllm"

    def test_a_bad_backend_fails_before_any_import(self) -> None:
        # The check must not depend on `vllm` or `transformers` being installed,
        # or a typo on a fresh box reports a missing dependency instead.
        config = ModelConfig()
        config.backend = "hf "  # type: ignore[assignment]
        with pytest.raises(ValueError):
            build_model(config, tools=("bash",))
