from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from wavelet.orchestrator.algorithms import Algorithm


_REGISTERED_NAME_ATTRIBUTE = "__wavelet_algorithm_name__"
FactoryT = TypeVar("FactoryT", bound=Callable[..., object])


def register_algorithm(name: str) -> Callable[[FactoryT], FactoryT]:
    """Give a class or factory a file-local algorithm name."""
    if not name.strip():
        raise ValueError("Custom algorithm registration name must not be empty.")
    if name != name.strip():
        raise ValueError(
            "Custom algorithm registration name must not have surrounding whitespace."
        )

    def decorator(factory: FactoryT) -> FactoryT:
        setattr(factory, _REGISTERED_NAME_ATTRIBUTE, name)
        return factory

    return decorator


def load_custom_algorithm(
    file_path: str | Path,
    algorithm_name: str,
    *,
    kwargs: dict[str, object] | None = None,
) -> Algorithm:
    """Load, construct, and validate an algorithm from a Python file."""
    path = _resolve_file(file_path)
    module = _load_module(path)
    factory = _find_factory(module, algorithm_name, path=path)
    instance = factory(**(kwargs or {}))
    return _validate_instance(instance, name=algorithm_name, path=path)


def _resolve_file(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Custom algorithm file does not exist: {path}")
    return path


def _load_module(path: Path) -> ModuleType:
    module_name = _module_name(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom algorithm file: {path}")

    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        _restore_module(module_name, previous_module)
        raise
    return module


def _module_name(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    return f"_wavelet_custom_algorithm_{digest}"


def _restore_module(module_name: str, previous_module: ModuleType | None) -> None:
    if previous_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous_module


def _find_factory(
    module: ModuleType,
    algorithm_name: str,
    *,
    path: Path,
) -> Callable[..., object]:
    registered = [
        value
        for value in vars(module).values()
        if getattr(value, _REGISTERED_NAME_ATTRIBUTE, None) == algorithm_name
    ]
    if len(registered) > 1:
        raise ValueError(
            f"Custom algorithm name {algorithm_name!r} is registered more than once "
            f"in {path}"
        )
    factory = registered[0] if registered else getattr(module, algorithm_name, None)
    if not callable(factory):
        raise TypeError(
            f"Custom algorithm {algorithm_name!r} is not callable in {path}"
        )
    return factory


def _validate_instance(instance: object, *, name: str, path: Path) -> Algorithm:
    missing_hooks = [
        hook
        for hook in ("score_rollout", "score_group")
        if not callable(getattr(instance, hook, None))
    ]
    if missing_hooks:
        joined = ", ".join(missing_hooks)
        raise TypeError(
            f"Custom algorithm {name!r} from {path} is missing callable hooks: "
            f"{joined}."
        )
    return cast("Algorithm", instance)
