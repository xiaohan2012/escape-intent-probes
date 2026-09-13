"""Check every hosted cell's provider pin against the live endpoint, before a run.

    uv run python scripts/check_pins.py configs/descent-02-*.yaml

Exists because this failure has now cost two runs, and both times the log was
unreadable. A pin that resolves to zero endpoints comes back as a bare `404`,
which is indistinguishable from a model that will not answer — the exact signal
these runs are looking for. D22 fixed the *reporting* (the body is in the
exception now). This fixes the *timing*: the pin is validated in one second per
cell instead of discovered at trajectory 1 of 24.

Two traps, both of which produced a 404 in practice:

* **The aggregate `/models` listing is not a validator.** It shows every
  endpoint that exists, including the ones this account's data policy excludes
  for training on paid requests. Only a request with `provider.only` set
  actually exercises the routing.
* **`provider/quantization` is a filter, not a request.** `novita/bf16` matches
  nothing when Novita serves that model at fp8, and fails closed.

So the check sends the real thing: a tool-enabled completion, pinned, capped at
a few tokens. Costs a fraction of a cent per cell.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.config import RunConfig  # noqa: E402

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}
"""A tool schema, because tool support is part of what is being checked.

A provider can serve a model and not serve its tool protocol, and the loop's
first step would then fail to parse on every trajectory.
"""


def check(config: RunConfig, client: object) -> tuple[bool, str]:
    """One pinned request. Returns whether it worked and what to print."""
    model = config.model
    if model.backend != "openrouter":
        return True, f"skipped ({model.backend} is local)"
    payload: dict[str, object] = {
        "model": model.model_id,
        "messages": [{"role": "user", "content": "say ok"}],
        "tools": [PROBE_TOOL],
        "max_tokens": 8,
    }
    if model.providers:
        payload["provider"] = {"only": list(model.providers)}

    response = client.post(  # type: ignore[attr-defined]
        f"{model.api_base.rstrip('/')}/chat/completions", json=payload
    )
    body = response.json()
    if response.is_error:
        error = body.get("error", {})
        available = error.get("metadata", {}).get("available_providers")
        detail = error.get("message", response.text)[:200]
        if available:
            detail = f"{detail} | available: {', '.join(available)}"
        return False, detail
    return True, f"served by {body.get('provider', '?')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()

    import httpx

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    failures = 0
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(60.0)
    ) as client:
        for path in sorted(args.configs):
            config = RunConfig.from_yaml(path)
            pin = ",".join(config.model.providers) or "(unpinned)"
            ok, detail = check(config, client)
            failures += not ok
            print(
                f"{'ok  ' if ok else 'FAIL'} {config.run_id:<40} "
                f"{config.model.model_id:<36} {pin:<18} {detail}",
                flush=True,
            )

    if failures:
        print(f"\n{failures} cell(s) would fail at trajectory 1. Fix the pin.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
