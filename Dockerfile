ARG UV_VERSION=0.11.17
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# The locked Linux dependencies use CUDA 13 wheels. Keep the toolkit available
# at runtime because vLLM and attention kernels may compile code on first use.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV CUDA_HOME=/usr/local/cuda \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/wavelet/.venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/wavelet/.venv/bin:/usr/local/cuda/bin:$PATH \
    HF_HUB_ETAG_TIMEOUT=500 \
    HF_HUB_DOWNLOAD_TIMEOUT=300 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libnuma1 \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /opt/wavelet

# Resolve third-party packages in a source-independent cache layer. The
# prebuilt FlashAttention wheel is matched to Python 3.12, CUDA 13.0, and the
# PyTorch version in uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --python /usr/bin/python3.12 --locked --no-dev \
        --extra flash-attn --no-install-project

COPY . .
RUN userdel --remove ubuntu 2>/dev/null || true
RUN groupdel ubuntu 2>/dev/null || true
RUN uv sync --python /usr/bin/python3.12 --locked --no-dev --extra flash-attn \
    && groupadd --gid "${GROUP_ID}" wavelet \
    && useradd --uid "${USER_ID}" --gid wavelet --create-home wavelet \
    && mkdir -p /opt/wavelet/outputs \
    && chown -R wavelet:wavelet /opt/wavelet/outputs

USER wavelet

ENTRYPOINT ["wavelet"]
