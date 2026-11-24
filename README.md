
<div align="center">

<img src="https://github.com/andreaslam/TrueZero/blob/main/TrueZero.png" alt="TrueZero logo">

### A Python and Rust chess engine that starts from Zero.
<img src="https://img.shields.io/badge/Powered%20by-Rust-b7410e" alt="Powered by Rust">
<img src="https://img.shields.io/badge/Powered%20by-Python-306998" alt="Powered by Python">
<img src="https://badgen.net/github/commits/andreaslam/TrueZero/main" alt="Total commits">
</div>

## About the Engine 
Instead of using hand-crafted evaluations (HCE), this AI learns how to play through playing against itself, starting with zero prior knowledge except for the rules of chess.

The chess engine will then play games against itself using the evaluation to evaluate chess positions, done using [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).

This project is still very much work-in-progress.

## Engine setup

Firstly, download this repository onto your computer. 

```
git clone https://github.com/andreaslam/TrueZero
```

Make sure you have Rust installed. If not, follow the instructions [here](https://doc.rust-lang.org/book/ch01-01-installation.html). 
Make sure you have Python installed. If not, download the latest version [here]([https://doc.rust-lang.org/book/ch01-01-installation.html](https://www.python.org/downloads/)). 
Configure `tch-rs` from the instructions [here](https://github.com/LaurentMazare/tch-rs/blob/main/README.md). For now, the neural net for this project is not provided but the NN architecture is available [here](https://github.com/andreaslam/TrueZero/blob/main/network.py) for reference.

Navigate to the Rust crate (which lives in `src/rust`):

```
cd TrueZero/src/rust
```

Then, build using `cargo`:

```
cargo build --release
```

Then choose a binary to run!

## Docker deployment

TrueZero has a reproducible CUDA/LibTorch container workflow for local,
NVIDIA GPU, and HPC deployments. It builds and tests automatically on pushes
to `main`, publishing successful images to GHCR. See
[`DOCKER.md`](DOCKER.md) for prerequisites, persistent model storage, GPU
execution, one-command updates, rollback, and Slurm/Apptainer usage.

> **Project layout:** Rust binaries live in `src/rust/src/bin/` and are auto-discovered by Cargo. Data folders (`nets/`, `games/`, `hidden/`, `python_client_games/`, `experiment_nets/`, `frames/`) live at the **project root** (`TrueZero/`) and are resolved automatically — both the Rust code (`tzrust::data_path()`) and the Python code (`src/python/paths.py`) anchor paths to the project root, so you can run everything from any working directory.

## Setting environment variables


## Python environment

Python dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`). From the project root:

```
uv sync
```

This creates `.venv/` with Python 3.12 (from `.python-version`) and every default dependency group: `torch`, `gui` (training plots, GUI), `viz` (visualisation, TensorBoard) and `tools`. Run scripts with `uv run`, e.g. `uv run python client.py`, or activate `.venv` first. To install only the libtorch version the Rust `tch` crate is built against (what the Docker image uses), run `uv sync --no-default-groups --group tch`.

## Setting environment variables

Set the following environment variables to configure `tch-rs` before running anything (or `source script.sh`).

```
export LIBTORCH_USE_PYTORCH=1
PYTORCH_PATH=$(uv run python -c "import torch; print(torch.__path__[0])")
export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"
```

On Windows, it's
```
$env:LIBTORCH_USE_PYTORCH=1
$PYTORCH_PATH = uv run python -c "import torch; print(torch.__path__[0])"
$env:PATH = "$PYTORCH_PATH\lib;$env:PATH"
```

## Running Data Generation and Training

Before data generation, ensure that the loaded copy of the Repository is up-to-date.

```
git pull
```

To run data generation, simply run the python training client `client.py`, the `main` binary and the `server` binary as follows. Open a new terminal window for each.

Python side (from `src/python`):

```
cd src/python
uv run python client.py
```

Rust side (from `src/rust`):

```
cd src/rust
cargo run --bin main --release
```

```
cd src/rust
cargo run --bin server --release
```

### Lichess bot

`lichess` plays on Lichess with the networks supplied by the training server.
It connects to the server like `main` (a RustDataGen, honouring
`TZ_SERVER_HOST`/`TZ_SERVER_PORT`) and evaluates positions with the same
batched executors. Start `server` (and `client.py`, which provides the
networks) as above, export `LICHESS_API_KEY`, then run:

```
cd src/rust
cargo run --bin lichess --release
```

The bot waits until the first network is loaded, then accepts incoming
standard-chess challenges, offers a rematch after each game, and challenges
randomly chosen online bots whenever it has no game in progress. Later
networks are loaded as the server sends them.

**Training on Lichess games is off by default.** The bot then only plays: it
searches without exploration noise and sends no games to the server. To use
its games as training data, set:

```powershell
$env:TZ_LICHESS_COLLECT_DATA = "1"
```

With collection on, the positions where TrueZero was to move are written with
the game's Lichess result (including resignations, timeouts and agreed draws)
and sent to the server every 100 games, like self-play data. Aborted games are
not recorded.

Other settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TZ_NUM_EXECUTORS` | `2` | Executor threads |
| `TZ_BATCH_SIZE` | `1024` | Executor batch size |
| `TZ_TENSOR_QUEUE_CAPACITY` | executors × batch size | Evaluation queue length |
| `TZ_MAX_CONCURRENT_GAMES` | `4` | Games played at once; extra challenges are declined |

All data (`nets/`, `games/`, `hidden/`, `python_client_games/`, `log.npz`) is read from and written to the project root automatically, regardless of where you launch each process.

## Training runs: run.toml, telemetry and the anchor ladder

A training run is a data root (the project root, or `$env:TZ_DATA_ROOT`) holding
`run.toml`, `nets/`, `games/`, `hidden/` and `python_client_games/`. Before the first start:

```powershell
Copy-Item run.example.toml run.toml    # into the data root; edit run_id and anything else
Copy-Item observe.example.toml observe.toml   # optional: W&B, log tailing, ladder
```

**run.toml is what the run is.** Self-play (`main`) and the trainer (`client.py`) both read
it; every key is required. The first process to start writes `hidden/run_lock.json` with the
run_id, the SHA-256 of run.toml and the git commit. Later starts are refused if run.toml
changed, the code is at another commit or has uncommitted changes (commit first), or the
run_id differs. `--allow-config-change` / `--allow-code-change` override the first two and
are recorded in `hidden/run_log.jsonl`. The old `TZ_NUM_GENERATORS`, `TZ_VIRTUAL_LEAVES`,
`TZ_PCR_*` and `TZ_LR_*` variables are ignored with a warning (`Remove-Item
Env:TZ_PCR_FULL_PROB` silences it).
`TZ_NUM_EXECUTORS` and `TZ_BATCH_TIMEOUT_US` stay env vars: they only change throughput.

A branch of a run (same commit, different hyperparameters) is a new data root with its own
run.toml: a new `run_id`, `parent_run_id` set to the parent and `fork_sample` to the samples
trained at the fork point, starting from a copy of the parent's net and
`hidden/trainer_state.pt`. Branches never share replay files.

Provenance: `hidden/provenance/datagen.jsonl` (self-play: generator, run, config, commit,
net for each replay file) and `hidden/provenance/trainer.jsonl` (trainer: each received
file) join on `bin_sha256`.

**observe.toml is how the run is watched** and can change any time:

- `[wandb]`: on by default (`mode = "online"`, project `truezero`, your default entity,
  run name = `run_id`, group = `parent_run_id` or `run_id`). Log in once with
  `uv run wandb login`; without a login the trainer logs offline rather than prompting
  (`wandb sync hidden/wandb/offline-run-*` uploads later). `mode = "off"` disables it. Metrics
  are plotted against samples trained: losses, policy accuracy (full searches only),
  policy entropy, LR, buffer age, gate results, per-file game length, draw rate and the
  first-move distribution, and (with `[tail]`) nodes/s, evals/s and executor timings
  parsed from the Rust logs (`.\server.exe | Tee-Object logs\server.log`).
- `[ladder]`: every N accepted nets the trainer runs `ladder` (build it with
  `cargo build --release --bin ladder`) in the background: fixed-node matches against
  frozen anchors (a copied net such as `anchors/tz_18.pt`, `random`, or
  `stockfish:<path>:<skill>`), giving Elo against things that never change, plus the
  bundled mate-in-1/2 suite (`src/python/eval/mate_suite.txt`) as % solved. Results go
  to W&B (`ladder/*`) and `hidden/ladder.jsonl`. `uv run python src/python/ladder.py
  nets/tz_40.pt` evaluates nets by hand.

## Running the Engine in UCI

To use TrueZero through UCI, run the following from `src/rust`:

```
cd src/rust
cargo run --bin ucimain --release -- nets/tz_42.pt
```

The optional argument is the network file, resolved relative to the project root (or `TZ_DATA_ROOT` if set); an absolute path also works. Without it, the engine uses `TZ_MODEL_PATH` if set, otherwise `nets/tz_0.pt`. The compiled binaries (in `src/rust/target/release/`) can also be launched directly from any directory.

## Internal Engine testing (non-UCI compliant)
- `getdecode.rs` - used for obtaining the encoded NN inputs.
- `getmove.rs` - used for obtaining a single tree search.
- `getgame.rs` - used for obtaining a game.
- `getinferencetime.rs` - used for benchmarking inference times and batching effectiveness through calculating the nodes/s.

## External Engine testing (UCI compliant)
- `uci.rs` - contains code for UCI implementation. Code modified from JW's [monty](https://github.com/official-monty/monty) engine
- `ucimain.rs` - used for running games using UCI.

## Source code for the Engine
- `boardmanager.rs` - a wrapper for the cozy-chess library. Manages and handles draw conditions, such as fifty-move repetition, threefold repetition and must-draw scenarios.
- `cache.rs` - contains abstractions for the cache key.
- `dataformat.rs` - contains necessary abstractions for `fileformat.rs`.
- `decoder.rs` - used to decode and encode inputs for the Engine. Also handles the creation of child nodes. This is where NN inference happens.
- `dirichlet.rs` - Dirichlet noise generator.
- `elo.rs` - contains code for elo calculation used in `enginetest.rs`.
- `fileformat.rs` - contains the code for saving data files for training the Engine.
- `mcts_trainer.rs` - used for MCTS tree search. Initialises the NN and manages the entire tree search. Adds Dirichlet noise to search results.
- `mcts.rs` - abstraction for non-trainer MCTS search. Supports UCI search options, such as `go infinite`.
- `mvs.rs` - a large array that contains all possible moves in chess. Used for indexing and storing (legal) move order. Statically loads and stored during programme execution.
- `message_types.rs` - contains the message protocols for processes (such as the Generator and the training loop) to communicate with the server and vice vera.
- `selfplay.rs` - facilitates selfplay. This is where search is initialised. Contains temperature management.


### Data Generation and Training components

#### Rust binaries
- `main.rs` - runs multi-threaded data generation code, where each thread runs an independent game. It needs to be connected to `server.rs` via TCP in order to get the latest Neural Net. It also sends key statistics for live telemetry.
- `server.rs` - a TCP server that co-ordinates Rust data generation and Python training. It sends each connected instance a unique identifier, broadcasts key information to different processes, which include statistics, Neural Network information and training settings.

#### Python code

- `client.py` - runs training and manages neural network training. Connects to `server.rs` via TCP to receive file paths for neural network training.

### Utility and visualisation code

- `exec_plotter.py` - debugging code that shows the thread schedule. Useful for debugging async tasks (such as data generation code for the Engine)
- `experiment.py` - code that can convert data generated from `client.py` into TensorBoard-readable format. This is an alternative experiment visualiser. Additionally, the code supports remote visualisation if SSH port forwarding is enabled on your device.
- `gui.py` - GUI code for `scheduler.py`. Code is currently work in progress. Running this file directly launches a demo version without server backend.
- `onnx_exporter.py` - contains code to convert `.pt` model weights to `.onnx`
- `scheduler.py` - code for TrueScheduler, an experiment scheduler for scheduling experiments and monitoring server controls. This GUI also supports remote SSH logins to schedule experiments on external devices. 
- `visualiser.py` - visualises training data and monitoring key performance indicators logged in `client.py`, where code from `lib/plotter.py`. For more details on the `lib` folder see [here](https://github.com/andreaslam/TrueZero?tab=readme-ov-file#credits-and-acknowledgements).
- `visualisenet.py` - code that allows visualisation of neural network activations. Generates a `.gif` file for an animation of a policy-only game.

## Libraries/technologies used 
This Python and Rust Engine uses the following:
### Python 

- **Matplotlib** - used for plotting and creating animations and visualisations in `visualisenet.py`. 
- **NumPy** - used for processing data (chess board representation after one-hot encoding, handling final outcome and final game result)
- **PyTorch** - used for creating and training the Neural Network. Also used for visualising experiments through its TensorBoard API.

### Rust

- **cozy-chess** - chess move generation library. There is a [simple wrapper](https://github.com/andreaslam/TrueZero/blob/main/src/boardmanager.rs) of this library that TrueZero uses that covers draws, repetitions and serves as an interface between cozy-chess and the rest of the code.
- **flume** - multi-sender, multi-producer channels used to send data between channels for data generation.
- **tch-rs** - Rust wrapper of libtorch. Used for Neural Network inference.
- **crossbeam** - enables multithreading data generation.
- **serde** - serialises messages to send across TCP server and serialises finished data files to binary.


## Credits and Acknowledgements

I would like to extend my heartfelt thanks to **[Karel Peeters](https://github.com/KarelPeeters)** for his persistent help and guidance. Without him this project would not been possible. 

Portions/entire files of code from [KZero](https://github.com/KarelPeeters/kZero) are being used in this Repository with express permission from Karel, which include:

- `src/dirichlet.rs`
- `src/fileformat.rs`
- `src/dataformat.rs`
- the `lib` folder used for reading KZero's custom data format and training

The [UCI code](https://github.com/andreaslam/TrueZero/blob/main/src/uci.rs) is modified from JW's [monty](https://github.com/jw1912/monty) engine.
