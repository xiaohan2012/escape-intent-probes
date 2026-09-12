"""One place that decides which real backend a run uses.

Both entry scripts had their own copy of this choice, which was fine while there
was only one backend and stops being fine now: a run whose `meta.json` says
`model_id` but not *how* the model was served is not reproducible, and two
scripts that disagree about the default are a bug nobody notices until the
numbers differ.

Imports are deferred to call time so that neither `transformers` nor `vllm` is
needed to run the loop against the fake model.
"""

from __future__ import annotations

from escape_probes.config import ModelConfig
from escape_probes.model import ModelBackend


def prefers_concurrency(model: ModelBackend) -> bool:
    """Whether this backend is better driven by `drive_threaded` than `drive_batch`.

    A trait of the backend rather than a second reading of `config.backend`:
    that string is already switched on once, here, and deciding the same
    question again elsewhere means two switches that can drift. Duck-typing on
    `generate_batch` is not enough either — the HuggingFace backend has no batch
    call and still must not be driven concurrently, because threads on one card
    fight rather than overlap. What the driver needs to know is whether requests
    are served somewhere else, which only the backend knows.
    """
    return bool(getattr(model, "concurrent_requests", False))


def build_model(config: ModelConfig, tools: tuple[str, ...]) -> ModelBackend:
    """The backend named by `config.backend`.

    `vllm` is the default because it is the rollout path (D21). `hf` stays
    reachable for two reasons: Pass 2 needs forward hooks, which vLLM does not
    expose, and it is the reference implementation when a vLLM trajectory looks
    wrong. `openrouter` needs no weights at all and produces no probe data; it
    is the frontier screen's path (D22).
    """
    if config.backend == "vllm":
        from escape_probes.vllm_backend import VLLMModel  # noqa: PLC0415

        return VLLMModel(config, tools=tools)
    if config.backend == "hf":
        from escape_probes.hf_backend import HFModel  # noqa: PLC0415

        return HFModel(config, tools=tools)
    if config.backend == "openrouter":
        from escape_probes.api_backend import APIModel  # noqa: PLC0415

        return APIModel(config, tools=tools)
    raise ValueError(f"unknown backend {config.backend!r}")


__all__ = ["build_model", "prefers_concurrency"]
