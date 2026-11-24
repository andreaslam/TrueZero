"""Path helpers anchoring data folders (nets/, games/, hidden/, ...) to the data root.

The data root is the current run's folder, ``<runs_dir>/<run id>/`` (see
``runconfig.py``); ``$TZ_DATA_ROOT`` pins it instead (Docker, tests). Before any run exists
it is the repository root. The trainer fixes it with ``set_root`` once it knows its run.

Use ``data_path("nets/tz_0.pt")`` anywhere so scripts work regardless of the current
working directory.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_root = None


def _default_root():
    pinned = os.environ.get("TZ_DATA_ROOT")
    if pinned:
        return Path(pinned)
    try:
        import runconfig

        cfg = runconfig.load_config().config
        runs = runconfig.runs_dir(cfg)
    except Exception:
        runs = REPO_ROOT / "runs"
    try:
        run_id = (runs / "current_run").read_text().strip()
    except OSError:
        run_id = ""
    return runs / run_id if run_id else REPO_ROOT


def root() -> Path:
    """The data root, fixed at first use (or by `set_root`)."""
    global _root
    if _root is None:
        _root = _default_root()
    return _root


def set_root(path) -> None:
    global _root
    _root = Path(path)


def data_path(rel: str) -> str:
    """Return an absolute path for ``rel`` relative to the data root."""
    return str(root() / rel)


def data_dir(rel: str) -> str:
    """Return an absolute path for a directory relative to the data root,
    creating it if it does not exist."""
    p = root() / rel
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
