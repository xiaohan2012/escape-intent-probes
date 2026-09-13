"""Every backend module's top-level names must resolve, with no heavy imports.

`vllm_backend` imported `command_token_index` from `escape_probes.chat` after
the probe-position refactor had moved it to `escape_probes.model`. Nothing
caught it: the vLLM path had not run since, and the suite does not import the
module because `vllm` is not installed in CI. It surfaced on a rented H100, at
the first line of the first smoke run, as `ImportError`.

Checked by parsing rather than importing, so it costs nothing and needs neither
torch nor vllm. This finds exactly one class of bug — a name imported from the
wrong module — and that is the class that cost us a card's first ten minutes.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "escape_probes"
MODULES = sorted(path.name for path in SOURCE.glob("*.py") if path.name != "__init__.py")


@pytest.mark.parametrize("filename", MODULES)
def test_every_internal_import_resolves(filename: str) -> None:
    tree = ast.parse((SOURCE / filename).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
            "escape_probes"
        ):
            continue
        module = importlib.import_module(node.module)
        missing = [alias.name for alias in node.names if not hasattr(module, alias.name)]
        assert not missing, f"{filename} imports {missing} from {node.module}, which lacks them"
