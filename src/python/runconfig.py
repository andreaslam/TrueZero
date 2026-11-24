"""Settings and runs for the trainer: the Python side of ``src/rust/runconfig.rs`` (same
files, same rules; keep the two in step).

- ``truezero.example.toml`` (repository root, committed): every setting and its default.
- ``truezero.toml`` (repository root, not committed): the user's settings, laid over the
  example. Only changed keys are needed; an unknown key or a wrongly typed value is an
  error. Created from the example on first use.
- ``<runs_dir>/current_run``: the current run's id, written only by the server.
- ``<runs_dir>/<run id>/``: the run's folder (``nets/``, ``games/``, ``hidden/``, ...) with
  ``run.json`` and ``config.toml``.
- ``<run>/hidden/run_log.jsonl``: one line per process start; a start with a changed
  truezero.toml keeps a copy in ``<run>/hidden/config_history/``. Nothing is refused.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

import paths

REPO_ROOT = paths.REPO_ROOT
EXAMPLE_FILE = "truezero.example.toml"
CONFIG_FILE = "truezero.toml"
CURRENT_RUN_FILE = "current_run"
EXAMPLE_PATH = REPO_ROOT / EXAMPLE_FILE

# Same as runconfig.rs RESTART_EXIT_CODE: "start me again" to the launcher.
RESTART_EXIT_CODE = 75
# Same as runconfig.rs RESTART_SENTINEL: printed just before exiting for a restart.
RESTART_SENTINEL = "exiting so the launcher restarts"

# Environment variables that used to configure TrueZero; ignored with a warning.
RETIRED_ENV_VARS = {
    "TZ_LR_SCHEDULE": "[trainer] lr_schedule",
    "TZ_LR_WARMUP_SAMPLES": "[trainer] lr_warmup_samples",
    "TZ_NUM_GENERATORS": "[selfplay] num_generators",
    "TZ_VIRTUAL_LEAVES": "[selfplay] virtual_leaves",
    "TZ_PCR_FULL_PROB": "[selfplay] pcr_full_prob",
    "TZ_PCR_FAST_VISITS": "[selfplay] pcr_fast_visits",
    "TZ_PCR_EARLY_STOP": "[selfplay] pcr_early_stop",
    "TZ_NUM_EXECUTORS": "[selfplay] num_executors",
    "TZ_BATCH_TIMEOUT_US": "[selfplay] batch_timeout_us",
    "TZ_SERVER_PORT": "[server] port",
}


class ConfigError(Exception):
    """truezero.toml cannot be used; the message says why."""


def _kind(value):
    for kind, name in (
        (bool, "true or false"),
        (int, "an integer"),
        (float, "a number"),
        (str, "a string"),
        (list, "a list"),
        (dict, "a table"),
    ):
        if isinstance(value, kind):
            return kind, name
    return type(value), type(value).__name__


def merge(example, user):
    """The example's tables with the user's values laid over them. Raises ValueError naming
    every unknown key or wrongly typed value (an integer is accepted for a number)."""
    merged = copy.deepcopy(example)
    problems = []
    for table, values in user.items():
        base = merged.get(table)
        if not isinstance(base, dict) or not isinstance(values, dict):
            problems.append(f"unknown table [{table}]")
            continue
        for key, value in values.items():
            if key not in base:
                problems.append(f"unknown key [{table}] {key}")
                continue
            want, want_name = _kind(base[key])
            got, got_name = _kind(value)
            if want is float and got is int:
                value = float(value)
            elif want is not got:
                problems.append(f"[{table}] {key} must be {want_name}, got {got_name}")
                continue
            base[key] = value
    if problems:
        raise ValueError("; ".join(problems))
    return merged


def _range(problems, table, key, value, ok, msg):
    if not ok(value):
        problems.append(f"[{table}] {key} {msg}")


def validate(cfg):
    """Raise ValueError naming every out-of-range value in the effective settings (types
    were checked by `merge`). Same checks as Config::validate in Rust, plus [trainer]."""
    p = []
    name = cfg["run"]["name"]
    if not name.strip() or any(c in name for c in "/\\:"):
        p.append("[run] name must be non-empty and contain no / \\ or :")
    if cfg["net"]["arch"] not in ("TrueNet", "TrueNetXS"):
        p.append('[net] arch must be "TrueNet" or "TrueNetXS"')
    for key in ("blocks", "channels"):
        _range(p, "net", key, cfg["net"][key], lambda v: v > 0, "must be > 0")
    s = cfg["selfplay"]
    for key in ("num_generators", "virtual_leaves", "num_executors", "batch_size", "max_nodes"):
        _range(p, "selfplay", key, s[key], lambda v: v > 0, "must be > 0")
    _range(p, "selfplay", "pcr_full_prob", s["pcr_full_prob"], lambda v: 0 <= v <= 1,
           "must be in [0, 1]")
    if s["pcr_full_prob"] < 1 and s["pcr_fast_visits"] < 2:
        p.append("[selfplay] pcr_fast_visits must be at least 2")
    t = cfg["trainer"]
    for key in ("buffer_size", "min_buffer_to_train", "min_training_steps",
                "startup_training_steps", "epochs_per_gen", "moves_left_delta", "clip_norm"):
        _range(p, "trainer", key, t[key], lambda v: v > 0, "must be > 0")
    bs = t["batch_size"]
    if not (bs > 0 and bs & (bs - 1) == 0):
        p.append("[trainer] batch_size must be a power of two")
    for key in ("recent_batch_fraction", "test_fraction"):
        _range(p, "trainer", key, t[key], lambda v: 0 <= v <= 1, "must be in [0, 1]")
    for key in ("weight_decay", "value_weight", "wdl_weight", "moves_left_weight", "policy_weight"):
        _range(p, "trainer", key, t[key], lambda v: v >= 0, "must be >= 0")
    if cfg["wandb"]["mode"] not in ("online", "offline", "off"):
        p.append('[wandb] mode must be "online", "offline" or "off"')
    if not cfg["launcher"]["python"]:
        p.append("[launcher] python must name a program")
    if p:
        raise ValueError("; ".join(p))


def config_from_texts(example_text, user_text):
    """The checked effective settings from the two files' texts."""
    if tomllib is None:
        raise ConfigError("reading truezero.toml needs Python 3.11+ (tomllib)")
    try:
        cfg = merge(tomllib.loads(example_text), tomllib.loads(user_text))
        validate(cfg)
    except (ValueError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{CONFIG_FILE}: {e}") from e
    return cfg


def ensure_user_config(root=REPO_ROOT):
    """Create truezero.toml from the example if it does not exist yet."""
    path = Path(root) / CONFIG_FILE
    if path.exists():
        return
    try:
        with open(path, "xb") as f:
            f.write((Path(root) / EXAMPLE_FILE).read_bytes())
        print(f"[config] Created {path} from {EXAMPLE_FILE}; edit it to change any setting.")
    except FileExistsError:
        pass


@dataclass
class LoadedConfig:
    config: dict
    user_bytes: bytes
    config_sha256: str


def load_config(root=REPO_ROOT):
    """Load truezero.toml (creating it if needed) over the example in `root`."""
    root = Path(root)
    try:
        ensure_user_config(root)
        example = (root / EXAMPLE_FILE).read_text(encoding="utf-8")
        user_bytes = (root / CONFIG_FILE).read_bytes()
    except OSError as e:
        raise ConfigError(str(e)) from e
    cfg = config_from_texts(example, user_bytes.decode("utf-8"))
    return LoadedConfig(cfg, user_bytes, hashlib.sha256(user_bytes).hexdigest())


def runs_dir(cfg, root=REPO_ROOT):
    return Path(root) / cfg["paths"]["runs_dir"]


def repo_path(rel, root=REPO_ROOT):
    """A path from the config that is relative to the repository root."""
    return str(Path(root) / rel)


def current_run_id(runs):
    try:
        text = (Path(runs) / CURRENT_RUN_FILE).read_text().strip()
    except OSError:
        return None
    return text or None


def retired_env_var_warning(environ=os.environ):
    """A warning naming every retired variable that is set, or None. They are ignored."""
    found = [
        f"{n} (now truezero.toml {k})" for n, k in RETIRED_ENV_VARS.items() if n in environ
    ]
    if not found:
        return None
    return (
        "IGNORING environment variables that no longer configure TrueZero: "
        + ", ".join(found)
        + ". truezero.toml decides; remove them (PowerShell: Remove-Item Env:NAME) to "
        "silence this."
    )


def git_state(repo=REPO_ROOT):
    """(commit, dirty) of the working tree the trainer runs from. Untracked files do not
    count as dirty. Without a repository, TZ_BUILD_GIT_SHA (set in the Docker image) or
    ("unknown", True)."""

    def git(*args):
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode != 0:
            raise OSError(out.stderr.strip())
        return out.stdout.strip()

    try:
        sha = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no") != ""
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        injected = os.environ.get("TZ_BUILD_GIT_SHA", "")
        if injected and injected != "unknown":
            return injected, False
        return "unknown", True


@dataclass
class StartIdentity:
    run_id: str
    run_name: str
    config_sha256: str
    git_sha: str
    git_dirty: bool
    process: str


def _append_jsonl(path, entry):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"[run][Warning] could not append to {path}: {e!r}")


def record_start(run_dir, ident, user_bytes):
    """Log this start in hidden/run_log.jsonl; keep a copy of truezero.toml if it changed
    since the run was created. Returns the warnings. Never refuses (same as Rust)."""
    run_dir = Path(run_dir)
    hidden = run_dir / "hidden"
    notes = []
    if ident.git_dirty:
        notes.append("tracked files have uncommitted changes (logged as git_dirty)")
    try:
        record = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        record = None
    if record is not None:
        if record.get("git_sha") != ident.git_sha:
            notes.append(
                f"the run was created at commit {record.get('git_sha')} and this program is "
                f"at {ident.git_sha} (logged)"
            )
        if record.get("config_sha256") != ident.config_sha256:
            copy_path = hidden / "config_history" / f"{ident.config_sha256[:12]}.toml"
            if not copy_path.exists():
                try:
                    copy_path.parent.mkdir(parents=True, exist_ok=True)
                    copy_path.write_bytes(user_bytes)
                except OSError as e:
                    print(f"[run][Warning] could not save {copy_path}: {e!r}")
            notes.append(
                f"truezero.toml changed since the run started (copy kept in {copy_path})"
            )
    _append_jsonl(
        str(hidden / "run_log.jsonl"),
        {
            "time": time.time(),
            "process": ident.process,
            "pid": os.getpid(),
            "run_id": ident.run_id,
            "git_sha": ident.git_sha,
            "git_dirty": ident.git_dirty,
            "config_sha256": ident.config_sha256,
        },
    )
    return notes


def start(process):
    """Start-up for the trainer: load the settings, wait for the server to have created a
    run, fix this process's data root to it and record the start. Returns
    (effective settings, StartIdentity, run folder). Exits with a message on a bad
    truezero.toml."""
    if (warning := retired_env_var_warning()) is not None:
        print(f"[run][Warning] {warning}", file=sys.stderr)
    try:
        loaded = load_config()
    except ConfigError as e:
        print(f"[config] {e}", file=sys.stderr)
        sys.exit(2)
    cfg = loaded.config
    pinned = os.environ.get("TZ_DATA_ROOT")
    if pinned:
        run_dir = Path(pinned)
    else:
        runs = runs_dir(cfg)
        waited = False
        while (run_id := current_run_id(runs)) is None:
            if not waited:
                print(f"[run] No run yet in {runs}; waiting for the server to create one.")
                waited = True
            time.sleep(1)
        run_dir = runs / run_id
    paths.set_root(run_dir)
    git_sha, dirty = git_state()
    ident = StartIdentity(
        run_dir.name, cfg["run"]["name"], loaded.config_sha256, git_sha, dirty, process
    )
    notes = record_start(run_dir, ident, loaded.user_bytes)
    print(
        f"[run] run {ident.run_id} in {run_dir}, settings sha256 {ident.config_sha256[:12]}, "
        f"commit {git_sha}" + (" (dirty)" if dirty else "")
    )
    for n in notes:
        print(f"[run][Warning] {n}")
    print(f"[run] effective [trainer]: {cfg['trainer']}")
    return cfg, ident, run_dir


# ---------------------------------------------------------------------------
# Replay-file provenance
#
# A sidecar to the KZero-format files, which are not changed. Self-play appends one line
# per file it sends to hidden/provenance/datagen.jsonl (generator_id, run, config, commit,
# net, bin_sha256); the trainer appends one line per file it receives to
# hidden/provenance/trainer.jsonl (saved path, bin_sha256). Join the two on bin_sha256 to
# trace any replay file to the generator, run, config, commit and net that produced it.
# ---------------------------------------------------------------------------


def find_datagen_record(bin_sha256, path=None):
    """The self-play line for a file, or None (e.g. self-play runs on another machine)."""
    path = path or paths.data_path("hidden/provenance/datagen.jsonl")
    try:
        with open(path) as f:
            for line in f:
                if bin_sha256 in line:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("bin_sha256") == bin_sha256:
                        return rec
    except OSError:
        pass
    return None


def record_received_file(saved_path, bin_bytes, ident, samples_trained=None, hidden_dir=None):
    """Log a received replay file and check it came from this run. Returns the matching
    self-play record (or None). A file from another run or config is reported, not dropped:
    deciding to discard data is the user's call."""
    hidden_dir = hidden_dir or paths.data_path("hidden")
    sha = hashlib.sha256(bin_bytes).hexdigest()
    rec = find_datagen_record(sha, os.path.join(hidden_dir, "provenance", "datagen.jsonl"))
    mismatch = []
    if rec is not None and ident is not None:
        for key, ours in (
            ("run_id", ident.run_id),
            ("config_sha256", ident.config_sha256),
            ("git_sha", ident.git_sha),
        ):
            if rec.get(key) != ours:
                mismatch.append(f"{key} {rec.get(key)!r} != {ours!r}")
    if mismatch:
        print(
            f"[run][Warning] replay file {saved_path} came from different self-play: "
            + "; ".join(mismatch)
        )
    _append_jsonl(
        os.path.join(hidden_dir, "provenance", "trainer.jsonl"),
        {
            "time": time.time(),
            "path": str(saved_path),
            "bin_sha256": sha,
            "generator_id": None if rec is None else rec.get("generator_id"),
            "net_sha256": None if rec is None else rec.get("net_sha256"),
            "samples_trained": samples_trained,
            "mismatch": mismatch,
        },
    )
    return rec
