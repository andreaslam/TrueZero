# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS builder

ARG RUST_VERSION=1.95.0
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    TZ_DATA_ROOT=/opt/truezero

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential clang libclang-dev pkg-config \
    libssl-dev python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/truezero
COPY requirements-tch-only.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools && \
    /opt/venv/bin/pip install -r requirements-tch-only.txt
ENV PATH="/opt/venv/bin:${PATH}" \
    LIBTORCH="$(python3 -c 'import torch; print(torch.__path__[0])')"

COPY src/rust/Cargo.toml src/rust/Cargo.lock src/rust/
COPY src/rust /opt/truezero/src/rust
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/opt/truezero/src/rust/target \
    cargo build --manifest-path src/rust/Cargo.toml --release --locked --bins

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS runtime

ARG APP_UID=10001
ARG APP_GID=10001
ENV DEBIAN_FRONTEND=noninteractive \
    TZ_DATA_ROOT=/data \
    TZ_DEVICE=auto \
    TZ_MODEL_PATH=/data/nets/tz_0.pt \
    LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/local/cuda/lib64 \
    RUST_BACKTRACE=1

RUN groupadd --gid ${APP_GID} truezero && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /usr/sbin/nologin truezero && \
    mkdir -p /opt/truezero/bin /opt/libtorch/lib /data && \
    chown -R truezero:truezero /opt/truezero /data

COPY --from=builder /opt/truezero/src/rust/target/release/ /opt/truezero/bin/
COPY --from=builder /opt/venv/lib/python3.10/site-packages/torch/lib/ /opt/libtorch/lib/
COPY --from=builder /opt/truezero/src/python /opt/truezero/src/python
COPY docker/entrypoint.sh /usr/local/bin/truezero

RUN chmod 0755 /usr/local/bin/truezero && \
    chown -R truezero:truezero /opt/truezero /opt/libtorch

WORKDIR /opt/truezero
USER truezero
VOLUME ["/data"]
EXPOSE 38475
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD test -d /data || exit 1
ENTRYPOINT ["/usr/local/bin/truezero"]
CMD ["ucimain"]
