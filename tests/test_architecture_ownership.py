import ast
from collections import Counter
from pathlib import Path

import wavelet.distributed.world as legacy_world
import wavelet.orchestrator.verifiers as legacy_verifiers
import wavelet.trainer.lora as legacy_lora
from wavelet.orchestrator import envs, scheduler
from wavelet.trainer import distributed, losses, model
from wavelet.trainer import lm_head as legacy_lm_head
from wavelet.trainer import rl_loss as legacy_rl_loss


def test_module_aliases_share_canonical_monkeypatch_state() -> None:
    assert legacy_lora is model
    assert legacy_world is distributed
    assert legacy_verifiers is scheduler
    assert legacy_verifiers._eval_metrics is envs._eval_metrics
    assert legacy_lm_head is losses
    assert legacy_rl_loss is losses


def test_canonical_modules_do_not_import_compatibility_paths() -> None:
    root = Path(__file__).parents[1] / "wavelet"
    compatibility_imports = (
        "from wavelet.data.loading",
        "from wavelet.data.rl_dataset",
        "from wavelet.data.tokenization",
        "from wavelet.distributed.world",
        "from wavelet.distributed.parallel_dims",
        "from wavelet.orchestrator.queue",
        "from wavelet.orchestrator.verifiers",
        "from wavelet.trainer.lora",
    )
    canonical_files = (
        root / "monitor.py",
        root / "data" / "sft.py",
        root / "orchestrator" / "state_server.py",
        root / "trainer" / "rl.py",
        root / "trainer" / "trainer.py",
        root / "transport" / "policy.py",
    )

    for path in canonical_files:
        source = path.read_text(encoding="utf-8")
        assert not any(item in source for item in compatibility_imports), path


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
