import ast
from collections import Counter
from pathlib import Path

import wavelet.orchestrator.verifiers as verifier_rollout_module
from wavelet.orchestrator import scheduler


def test_verifier_rollout_function_path_resolves_to_scheduler() -> None:
    assert verifier_rollout_module is scheduler


def test_modules_do_not_shadow_top_level_definitions() -> None:
    root = Path(__file__).parents[1] / "wavelet"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        assert not duplicates, f"{path}: {duplicates}"
