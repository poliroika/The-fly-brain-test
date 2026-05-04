# Multi-stage build: Rust core compiled inside the image, then a slim runtime.
#
# Stage 1: build the maturin wheel.
# Stage 2: install runtime deps and the wheel into a Python slim image.

ARG PYTHON_VERSION=3.11
ARG RUST_VERSION=1.83

FROM rust:${RUST_VERSION}-slim-bookworm AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_VERSION=${PYTHON_VERSION}

# Debian bookworm ships python3.11 as the default python3 — that's why we pin
# this base image instead of `slim` (which tracks trixie / python3.13).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl pkg-config libssl-dev \
        python3 python3-dev python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/buildvenv \
 && /opt/buildvenv/bin/pip install --upgrade pip "maturin>=1.7,<2"

ENV PATH="/opt/buildvenv/bin:${PATH}"

WORKDIR /build
COPY . .

RUN maturin build --release --strip --interpreter /opt/buildvenv/bin/python


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /build/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl \
 && pip install \
        "pydantic>=2.6,<3" \
        "hydra-core>=1.3,<2" \
        "omegaconf>=2.3,<3" \
        "yandex-ai-studio-sdk>=0.20,<1" \
        "tenacity>=8.2,<9" \
        "tqdm>=4.66" \
        "rich>=13.7" \
        "numpy>=1.26,<3"

# Hydra reads YAML at runtime; copy them so users can `flybrain run --config configs/...`.
COPY configs ./configs
COPY scripts ./scripts

ENTRYPOINT ["python", "-m", "flybrain.cli"]
CMD ["info"]
