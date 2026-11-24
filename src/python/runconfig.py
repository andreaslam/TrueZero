"""Run identity for the trainer: the frozen run.toml, the run lock and the start log.

Same files and the same rules as the Rust side (`src/rust/runconfig.rs`); keep the two in
step. See `run.example.toml` for what a run is and how the lock works.

- ``<data root>/run.toml``: the run's hyperparameters.
- ``<data root>/hidden/run_lock.json``: created by the first process to start; pins the
  run_id, the SHA-256 of run.toml's bytes and the git commit.
- ``<data root>/hidden/run_log.jsonl``: one line per process start.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

from paths import data_path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "run.example.toml"

# Every key is required and nothing else is allowed, so a typo cannot fall back to a
# default. Values are (type, check) with check returning an error message or None.
_POS = (lambda v: None if v > 0 else "must be > 0")
_NONNEG = (lambda v: None if v >= 0 else "must be >= 0")
_PROB = (lambda v: None if 0 <= v <= 1 else "must be in [0, 1]")
_ANY = (lambda v: None)

SCHEMA = {
    "run": {
        "run_id": (str, lambda v: None if v.strip() else "must not be empty"),
        "parent_run_id": (str, _ANY),
        "fork_sample": (int, _NONNEG),
        "seed": (int, _NONNEG),
    },
    "selfplay": {
        "num_generators": (int, _POS),
        "virtual_leaves": (int, _POS),
        "max_nodes": (int, _POS),
        "root_fpu": (float, _ANY),
        "children_fpu": (float, _ANY),
        "root_c_puct": (float, _ANY),
        "children_c_puct": (float, _ANY),
        "root_pst": (float, _ANY),
        "children_pst": (float, _ANY),
        "dirichlet_alpha": (float, _ANY),
        "dirichlet_eps": (float, _PROB),
        "moves_left_weight": (float, _ANY),
        "moves_left_clip": (float, _ANY),
        "moves_left_sharpness": (float, _ANY),
        "pcr_full_prob": (float, _PROB),
        "pcr_fast_visits": (int, _NONNEG),
        "pcr_early_stop": (bool, _ANY),
    },
    "trainer": {
        "buffer_size": (int, _POS),
        "batch_size": (
            int,
            lambda v: None if v > 0 and v & (v - 1) == 0 else "must be a power of two",
        ),
        "min_buffer_to_train": (int, _POS),
        "epochs_per_gen": (float, _POS),
        "recent_batch_fraction": (float, _PROB),
        "min_training_steps": (int, _POS),
        "startup_training_steps": (int, _POS),
        "test_fraction": (float, _PROB),
        "lr_schedule": (str, _ANY),
        "lr_warmup_samples": (str, _ANY),
        "weight_decay": (float, _NONNEG),
        "value_weight": (float, _NONNEG),
        "wdl_weight": (float, _NONNEG),
        "moves_left_weight": (float, _NONNEG),
        "moves_left_delta": (float, _POS),
        "policy_weight": (float, _NONNEG),
        "clip_norm": (float, _POS),
    },
}

# Environment variables that used to set run hyperparameters. They have no effect any more;
# a start that finds one warns, so a value left in a PowerShell session is noticed but can
# neither change a run nor block a start.
RETIRED_ENV_VARS = {
    "TZ_LR_SCHEDULE": "[trainer] lr_schedule",
    "TZ_LR_WARMUP_SAMPLES": "[trainer] lr_warmup_samples",
    "TZ_NUM_GENERATORS": "[selfplay] num_generators",
    "TZ_VIRTUAL_LEAVES": "[selfplay] virtual_leaves",
    "TZ_PCR_FULL_PROB": "[selfplay] pcr_full_prob",
    "TZ_PCR_FAST_VISITS": "[selfplay] pcr_fast_visits",
    "TZ_PCR_EARLY_STOP": "[selfplay] pcr_early_stop",
}


class RunRefused(Exception):
    """The run must not start; the message says why and how to override."""


def _type_ok(value, kind):
    if kind is float:
        # TOML writes 1 and 1.0 differently; both are fine for a float setting.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, kind)


def validate(cfg):
    """Raise ValueError naming every problem in a parsed run.toml."""
    problems = []
    for name in cfg.keys() - SCHEMA.keys():
        problems.append(f"unknown table [{name}]")
    for table, keys in SCHEMA.items():
        section = cfg.get(table)
        if not isinstance(section, dict):
            problems.append(f"missing table [{table}]")
            continue
        for key in section.keys() - keys.keys():
            problems.append(f"unknown key [{table}] {key}")
        for key, (kind, check) in keys.items():
            if key not in section:
                problems.append(f"missing key [{table}] {key}")
            elif not _type_ok(section[key], kind):
                problems.append(f"[{table}] {key} must be a {kind.__name__}")
            elif (msg := check(section[key])) is not None:
                problems.append(f"[{table}] {key} {msg}")
    run = cfg.get("run", {})
    if isinstance(run, dict) and ("parent_run_id" in run and "fork_sample" in run):
        if (run["parent_run_id"] == "") != (run["fork_sample"] == 0):
            problems.append("[run] parent_run_id and fork_sample must be set together")
    if problems:
        raise ValueError("; ".join(problems))


def load_run_config(path):
    """Parse and validate run.toml. Returns (config dict, SHA-256 of the file's bytes)."""
    if tomllib is None:
        raise RunRefused("reading run.toml needs Python 3.11+ (tomllib)")
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise RunRefused(
            f"cannot read {path}: {e}. Copy {EXAMPLE_PATH.name} from the repository root "
            "there and edit it before the first start."
        ) from e
    try:
        cfg = tomllib.loads(raw.decode("utf-8"))
        validate(cfg)
    except (ValueError, UnicodeDecodeError) as e:
        raise RunRefused(f"{path}: {e}") from e
    return cfg, hashlib.sha256(raw).hexdigest()


def retired_env_var_warning(environ=os.environ):
    """A warning naming every retired variable that is set, or None. They are ignored."""
    found = [f"{n} (now run.toml {k})" for n, k in RETIRED_ENV_VARS.items() if n in environ]
    if not found:
        return None
    return (
        "IGNORING environment variables that no longer configure a run: "
        + ", ".join(found)
        + ". run.toml decides; remove them (PowerShell: Remove-Item Env:NAME) to silence this."
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
        # No repository (the Docker image): the commit the image was built from.
        injected = os.environ.get("TZ_BUILD_GIT_SHA", "")
        if injected and injected != "unknown":
            return injected, False
        return "unknown", True


@dataclass
class StartIdentity:
    run: dict
    config_sha256: str
    git_sha: str
    git_dirty: bool
    process: str
    overrides_used: list = field(default_factory=list)


def _append_jsonl(path, entry):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"[run][Warning] could not append to {path}: {e!r}")


def _log_start(hidden_dir, ident, used):
    _append_jsonl(
        os.path.join(hidden_dir, "run_log.jsonl"),
        {
            "time": time.time(),
            "process": ident.process,
            "pid": os.getpid(),
            "run_id": ident.run["run_id"],
            "git_sha": ident.git_sha,
            "git_dirty": ident.git_dirty,
            "config_sha256": ident.config_sha256,
            "overrides_used": used,
        },
    )


def check_run_lock(hidden_dir, ident, allow_config_change=False, allow_code_change=False):
    """Check `ident` against the run lock, creating it on the first start, and log the
    start. Returns the overrides that were needed; raises RunRefused otherwise.

    - a different run_id in the same data root is always refused;
    - a different commit, uncommitted changes or an unknown commit need allow_code_change;
    - a different run.toml needs allow_config_change.
    """
    os.makedirs(hidden_dir, exist_ok=True)
    lock_path = os.path.join(hidden_dir, "run_lock.json")
    used = []
    code_problems = []
    if ident.git_sha == "unknown":
        code_problems.append("the git commit is unknown")
    if ident.git_dirty:
        code_problems.append("tracked files have uncommitted changes")

    if not os.path.exists(lock_path):
        if code_problems:
            if not allow_code_change:
                raise RunRefused(
                    f"refusing to start run {ident.run['run_id']}: "
                    + "; ".join(code_problems)
                    + ". Commit first, or pass --allow-code-change."
                )
            used.append("allow-code-change: " + "; ".join(code_problems))
        run = ident.run
        lock = {
            "run_id": run["run_id"],
            "parent_run_id": run["parent_run_id"],
            "fork_sample": run["fork_sample"],
            "seed": run["seed"],
            "git_sha": ident.git_sha,
            "config_sha256": ident.config_sha256,
            "created_at": time.time(),
            "created_by": ident.process,
        }
        try:
            with open(lock_path, "x") as f:
                json.dump(lock, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            print(
                f"[run] Created run lock {lock_path} for run {run['run_id']} "
                f"at commit {ident.git_sha}"
            )
            _log_start(hidden_dir, ident, used)
            return used
        except FileExistsError:
            used = []  # someone else created it first: check against theirs

    with open(lock_path) as f:
        lock = json.load(f)

    if lock["run_id"] != ident.run["run_id"]:
        raise RunRefused(
            f"run.toml says run_id {ident.run['run_id']!r} but this data root belongs to "
            f"run {lock['run_id']!r} ({lock_path}). A new run or a branch needs its own "
            "data root."
        )
    if lock["git_sha"] != ident.git_sha:
        code_problems.insert(
            0, f"at commit {ident.git_sha} but the run is pinned to {lock['git_sha']}"
        )
    if code_problems:
        if not allow_code_change:
            raise RunRefused(
                f"refusing to start run {ident.run['run_id']}: "
                + "; ".join(code_problems)
                + ". Check out the pinned commit, or pass --allow-code-change."
            )
        used.append("allow-code-change: " + "; ".join(code_problems))
    if lock["config_sha256"] != ident.config_sha256:
        problem = (
            f"run.toml (sha256 {ident.config_sha256[:12]}) differs from the locked one "
            f"(sha256 {lock['config_sha256'][:12]})"
        )
        if not allow_config_change:
            raise RunRefused(
                f"refusing to start run {ident.run['run_id']}: {problem}. Restore it, start "
                "a new run in its own data root, or pass --allow-config-change."
            )
        used.append("allow-config-change: " + problem)
    _log_start(hidden_dir, ident, used)
    return used


def start_run(process, allow_config_change=False, allow_code_change=False, path=None):
    """Load run.toml, refuse retired env vars, check the run lock and print the effective
    config. Returns (config, StartIdentity). Exits with a message on refusal."""
    try:
        if (warning := retired_env_var_warning()) is not None:
            print(f"[run][Warning] {warning}", file=sys.stderr)
        path = path or data_path("run.toml")
        cfg, sha = load_run_config(path)
        git_sha, dirty = git_state()
        ident = StartIdentity(cfg["run"], sha, git_sha, dirty, process)
        ident.overrides_used = check_run_lock(
            data_path("hidden"), ident, allow_config_change, allow_code_change
        )
    except RunRefused as e:
        print(f"[run] {e}", file=sys.stderr)
        sys.exit(2)
    run = cfg["run"]
    print(
        f"[run] run {run['run_id']} (parent {run['parent_run_id']!r}, fork_sample "
        f"{run['fork_sample']}), config {path} sha256 {sha}, commit {git_sha}"
        + (" (dirty)" if dirty else "")
    )
    for o in ident.overrides_used:
        print(f"[run][Warning] override used: {o}")
    print(f"[run] effective [trainer]: {cfg['trainer']}")
    return cfg, ident


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
    path = path or data_path("hidden/provenance/datagen.jsonl")
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
    hidden_dir = hidden_dir or data_path("hidden")
    sha = hashlib.sha256(bin_bytes).hexdigest()
    rec = find_datagen_record(sha, os.path.join(hidden_dir, "provenance", "datagen.jsonl"))
    mismatch = []
    if rec is not None and ident is not None:
        for key, ours in (
            ("run_id", ident.run["run_id"]),
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
