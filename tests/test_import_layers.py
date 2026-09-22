"""Executable guardrails for the package dependency direction."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOTS = {
    "configs",
    "contracts",
    "transport",
    "trainer",
    "inference",
    "orchestrator",
    "launch",
    "data",
    "kernels",
    "utils",
    "dashboard",
    "deployment",
    "tools",
    "entrypoints",
}

ALLOWED_IMPORTS: dict[str, set[str]] = {
    "configs": {"utils"},
    "contracts": {"configs", "utils"},
    "transport": {"configs", "contracts", "utils"},
    "trainer": {
        "configs",
        "contracts",
        "data",
        "kernels",
        "transport",
        "utils",
    },
    "inference": {"configs", "contracts", "data", "transport", "utils"},
    "orchestrator": {
        "configs",
        "contracts",
        "dashboard",
        "data",
        "inference",
        "transport",
        "utils",
    },
    "launch": PACKAGE_ROOTS - {"launch"},
    "data": {"configs", "utils"},
    "kernels": {"configs", "utils"},
    "utils": {"configs"},
    "dashboard": {"configs", "orchestrator", "transport", "utils"},
    "deployment": {"configs", "inference", "launch", "orchestrator", "utils"},
    "tools": {"configs", "data", "trainer", "utils"},
    "entrypoints": PACKAGE_ROOTS - {"entrypoints"},
}

KNOWN_VIOLATIONS: set[tuple[str, str]] = {
    ("trainer/trainer.py", "deployment"),
}


def _wavelet_edges() -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    wavelet_dir = Path(__file__).parents[1] / "wavelet"
    for path in wavelet_dir.rglob("*.py"):
        relative_path = path.relative_to(wavelet_dir).as_posix()
        owner = relative_path.split("/", 1)[0]
        if owner not in PACKAGE_ROOTS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module is not None:
                    imported_modules = [node.module]
            elif isinstance(node, ast.Call):
                function = node.func
                is_import_module = (
                    isinstance(function, ast.Name) and function.id == "import_module"
                ) or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "import_module"
                )
                if (
                    is_import_module
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    imported_modules = [node.args[0].value]
            for module in imported_modules:
                parts = module.split(".")
                if (
                    len(parts) >= 2
                    and parts[0] == "wavelet"
                    and parts[1] in PACKAGE_ROOTS
                    and parts[1] != owner
                ):
                    edges.add((relative_path, parts[1]))
    return edges


def test_import_edges_follow_the_layer_allowlist() -> None:
    edges = _wavelet_edges()
    unexpected = {
        edge
        for edge in edges
        if edge[1] not in ALLOWED_IMPORTS[edge[0].split("/", 1)[0]]
        and edge not in KNOWN_VIOLATIONS
    }
    assert not unexpected, f"Unexpected cross-layer imports: {sorted(unexpected)}"
    missing = KNOWN_VIOLATIONS - edges
    assert not missing, f"Stale known import violations: {sorted(missing)}"


def test_trainer_transport_orchestrator_import_without_vllm() -> None:
    code = """
import sys
sys.modules["vllm"] = None
import wavelet.trainer
import wavelet.transport
import wavelet.orchestrator
import wavelet.trainer.rl
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_no_sys_modules_module_aliases() -> None:
    wavelet_dir = Path(__file__).parents[1] / "wavelet"
    aliases: list[str] = []
    for path in wavelet_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "sys"
                and target.value.attr == "modules"
                and isinstance(target.slice, ast.Name)
                and target.slice.id == "__name__"
                for target in node.targets
            ):
                aliases.append(path.relative_to(wavelet_dir).as_posix())
    assert not aliases, f"Found sys.modules aliases: {sorted(aliases)}"
