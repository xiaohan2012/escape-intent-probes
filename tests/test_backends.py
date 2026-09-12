"""Tests for the backend factory.

Neither real backend can be constructed without weights, so what is checkable
here is the dispatch and the failure mode. That is worth a test anyway: the
factory exists because two entry scripts had diverging copies of this choice,
and a silent disagreement about the default is exactly the bug it removes.
"""

import pytest

from escape_probes.backends import build_model, prefers_concurrency
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

    def test_the_api_path_is_reachable_by_name(self) -> None:
        # The screen's backend (D22). Dispatch is worth a test on its own: a
        # YAML typo must fail at config load, not after the images are pulled.
        assert ModelConfig(backend="openrouter").backend == "openrouter"


class TestPrefersConcurrency:
    """Test which driver a backend asks for.

    The question is asked of the backend object because `config.backend` is
    already switched on once, in `build_model`; answering it a second time from
    the string is two switches that can drift.
    """

    def test_a_hosted_backend_asks_for_threads(self) -> None:
        from escape_probes.api_backend import APIModel

        assert APIModel.concurrent_requests is True

    def test_a_backend_that_says_nothing_is_driven_in_lock_step(self) -> None:
        # The safe default: threads over one card fight rather than overlap, so
        # a backend must opt in rather than out.
        class Local:
            def generate(self, messages):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        assert prefers_concurrency(Local()) is False

    def test_a_batch_call_is_not_what_decides_it(self) -> None:
        # Duck-typing on `generate_batch` would send the HuggingFace backend,
        # which has none, down the threaded path.
        class LocalBatched:
            def generate(self, messages):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def generate_batch(self, conversations):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        assert prefers_concurrency(LocalBatched()) is False
