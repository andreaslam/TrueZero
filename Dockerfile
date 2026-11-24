# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS builder

ARG RUST_VERSION=1.95.0
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    TZ_DATA_ROOT=/opt/truezero

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential clang libclang-dev pkg-config \
    libssl-dev python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/

WORKDIR /opt/truezero
COPY pyproject.toml uv.lock ./
# The `tch` group pins the torch wheel the Rust tch crate links against (see pyproject.toml).
# The venv uses the system python3 so it still works in the runtime stage.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group tch
ENV PATH="/opt/venv/bin:${PATH}" \
    LIBTORCH="$(python3 -c 'import torch; print(torch.__path__[0])')"

# The commit being built (.git is not in the build context); pins runs (run.example.toml).
ARG TZ_BUILD_GIT_SHA=unknown
ENV TZ_BUILD_GIT_SHA=${TZ_BUILD_GIT_SHA}
COPY src/rust/Cargo.toml src/rust/Cargo.lock src/rust/
COPY src/rust /opt/truezero/src/rust
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cargo build --manifest-path src/rust/Cargo.toml --release --locked --bins
COPY src/python /opt/truezero/src/python

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS runtime

ARG APP_UID=10001
ARG APP_GID=10001
ARG TZ_BUILD_GIT_SHA=unknown
ENV DEBIAN_FRONTEND=noninteractive \
    TZ_BUILD_GIT_SHA=${TZ_BUILD_GIT_SHA} \
    TZ_DATA_ROOT=/data \
    TZ_DEVICE=auto \
    TZ_MODEL_PATH=/data/nets/tz_0.pt \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64 \
    RUST_BACKTRACE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${APP_GID} truezero && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /usr/sbin/nologin truezero && \
    mkdir -p /opt/truezero/bin /opt/libtorch/lib /data && \
    chown -R truezero:truezero /opt/truezero /data

COPY --from=builder /opt/truezero/src/rust/target/release/ /opt/truezero/bin/
COPY --from=builder /opt/venv /opt/venv
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
