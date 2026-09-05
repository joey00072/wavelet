from __future__ import annotations

from pathlib import Path


def test_dockerfile_builds_locked_cuda_runtime_as_non_root() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04" in dockerfile
    assert "FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv" in dockerfile
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in dockerfile
    assert dockerfile.count("uv sync --python /usr/bin/python3.12 --locked") == 2
    assert "--extra flash-attn --no-install-project" in dockerfile
    assert "userdel --remove ubuntu" in dockerfile
    assert "USER wavelet" in dockerfile
    assert 'ENTRYPOINT ["wavelet"]' in dockerfile


def test_docker_context_excludes_secrets_artifacts_and_references() -> None:
    ignored = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".env", ".env.*", "ref", "outputs", "wandb"} <= ignored
