# Container deployment

TrueZero is built in a pinned CUDA 12.1.1 development image and runs in the
matching CUDA runtime image. The Rust binaries are compiled with the checked-in
`Cargo.lock` and `rust-toolchain.toml`; PyTorch supplies the LibTorch libraries
used by `tch-rs`. The runtime is non-root and contains no credentials or
checkpoints.

## Prerequisites

For GPU servers, install Docker Engine and the NVIDIA Container Toolkit, and
verify the host driver with:

```sh
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

The host driver must support CUDA 12.1 or newer. Host CUDA development headers
and LibTorch are not required. CPU execution is available with `TZ_DEVICE=cpu`
and without `--gpus`.

## Published image

On a server, copy `.env.example` to `.env`, set `TRUEZERO_DATA_DIR` to a
persistent directory, and place a TorchScript model at
`$TRUEZERO_DATA_DIR/nets/tz_0.pt`. Then:

```sh
docker login ghcr.io
cp .env.example .env
./scripts/update.sh
```

The image is published as `sha-<commit>`, `latest` for successful `main`
builds, and immutable `vX.Y.Z` tags for version tags. Pull requests build and
test without publishing. A failed workflow cannot move an existing tag.

Use an explicit version when reproducing an experiment:

```sh
./scripts/update.sh --tag v1.0.0
```

`update.sh` records the prior image, refuses to replace a running `main`,
`trainenginetest`, or training/client process unless `--force` is supplied,
preserves `/data`, and verifies that the replacement is running. Roll back with:

```sh
./scripts/update.sh --rollback
```

Rollback changes only the image reference; it never removes volumes or files.
Do not update a container during active training unless interruption is
explicitly acceptable.

## Local build and commands

The same image can be built locally:

```sh
docker build -t truezero:local .
mkdir -p .truezero-data/nets
cp /path/to/tz_0.pt .truezero-data/nets/tz_0.pt
TZ_DEVICE=cpu TRUEZERO_IMAGE=truezero:local docker compose up
```

`TZ_COMMAND` selects a Rust binary. Useful commands include `ucimain`, `main`,
`server`, `getinference`, and `lichess`. The training client can be run with:

```sh
docker compose run --rm truezero python src/python/client.py
```

For the normal generator/trainer workflow, start the provided services with:

```sh
docker compose up server selfplay training
```

They share the same persistent data directory and use the Compose service name
`server` for TCP communication on port 38475. Start only `truezero` for a
single UCI/inference process.

The following variables are runtime configuration and do not require rebuilding:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRUEZERO_DATA_DIR` | `./.truezero-data` | Persistent host data directory |
| `TZ_DATA_ROOT` | `/data` | Container data root |
| `TZ_MODEL_PATH` | `/data/nets/tz_0.pt` | TorchScript model |
| `TZ_DEVICE` | `auto` in the image, `cuda` in Compose | `auto`, `cpu`, `cuda`, or `mps` |
| `TZ_COMMAND` | `ucimain` | Rust binary to execute |

`TZ_DEVICE=cuda` fails explicitly if CUDA is unavailable. `auto` selects CUDA,
then Apple MPS, then CPU. `TZ_DATA_ROOT` is honored by both Rust and Python,
so checkpoints, game data, logs, and experiment outputs survive image updates.

## GPU smoke test

On a GPU host with a model mounted in the data directory:

```sh
TRUEZERO_DATA_DIR=/srv/truezero/data ./scripts/gpu-smoke-test.sh
```

This executes the real Rust inference binary on CUDA. It cannot be verified on
the macOS development host without NVIDIA hardware.

## HPC and Apptainer

Docker is often unavailable on compute nodes. The example
[`hpc/slurm-truezero.sbatch`](hpc/slurm-truezero.sbatch) uses Apptainer with
`--nv`, a persistent bind mount, one GPU, and a Slurm log. Adapt the partition,
account, image cache policy, and storage path. On restricted clusters, pull the
OCI image on a login node with `apptainer pull` and reference the local `.sif`
file in the job script.

## Compatibility and upgrades

The compatibility source of truth is the CUDA base image in `Dockerfile`,
`requirements-tch-only.txt`, `src/rust/Cargo.toml`, `src/rust/Cargo.lock`, and
`rust-toolchain.toml`. Upgrade CUDA and PyTorch together, rebuild from a clean
cache, run Rust tests and the GPU smoke test, then publish a new version tag.
Do not install host NVIDIA drivers into the image.

Common failures:

* `could not load libcudart`: install/configure NVIDIA Container Toolkit and
  use `--gpus all`; do not add host driver files to the image.
* `TZ_DEVICE=cuda ... CUDA is unavailable`: check `nvidia-smi`, the toolkit,
  and that the container was started with GPU access.
* model-not-found: place the model under the persistent data directory or set
  `TZ_MODEL_PATH`.
* LibTorch version errors: rebuild after changing the pinned PyTorch version;
  the image sets `LIBTORCH_USE_PYTORCH=1` and does not use a stale host install.

## GitHub configuration

The `container.yml` workflow needs no repository secret: GitHub's
`GITHUB_TOKEN` logs in to GHCR with the job's `packages: write` permission.
The package may need to be made public in the repository's Packages settings if
servers should pull it without authentication. For a private package, run
`docker login ghcr.io` on the deployment server with a token that has
read-only package access.

## Reproducible experiments

Record the image digest printed by `update.sh`, the immutable `sha-<commit>` or
version tag, the model path, and the persistent data directory. A later run
using the same image tag/digest and copied data directory restores the same
application and experiment inputs without baking checkpoints into the image.
