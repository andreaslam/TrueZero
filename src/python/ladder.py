"""Fixed anchor ladder: run the Rust `ladder` binary on a net and log the results.

The trainer calls `LadderRunner.maybe_run` whenever a candidate is accepted; every
`every_accepted_nets` generations (observe.toml [ladder]) it evaluates the new net in a
background thread against each anchor and on the tactics suite. Results are logged to W&B
(ladder/*) at the samples_trained of the net, and appended to hidden/ladder.jsonl.

By hand, e.g. to back-fill old nets:

    uv run python src/python/ladder.py nets/tz_40.pt nets/tz_80.pt
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from paths import data_path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = REPO_ROOT / "src" / "python" / "eval" / "mate_suite.txt"
RESULT_PREFIX = "LADDER_RESULT "


def default_binary():
    exe = "ladder.exe" if os.name == "nt" else "ladder"
    return str(REPO_ROOT / "src" / "rust" / "target" / "release" / exe)


def anchor_name(spec):
    """A short W&B key for an anchor spec."""
    if spec == "random":
        return "random"
    if spec.startswith("stockfish:"):
        return f"stockfish_skill{spec.rsplit(':', 1)[1]}"
    return Path(spec).stem


def resolve(path):
    return path if os.path.isabs(path) else data_path(path)


def parse_result(stdout):
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return None


def result_rows(results):
    """W&B metrics from a list of ladder results."""
    row = {}
    for r in results:
        if r["kind"] == "match":
            k = f"ladder/{anchor_name(r['anchor'])}"
            for key in ("elo", "elo_low", "elo_high", "score"):
                row[f"{k}/{key}"] = r[key]
            row[f"{k}/games"] = r["games"]
        elif r["kind"] == "tactics":
            row["ladder/tactics/solved_fraction"] = r["solved_fraction"]
            for cat, frac in r["by_category"].items():
                row[f"ladder/tactics/{cat}"] = frac
    return row


def ladder_commands(cfg, net_path):
    """The ladder invocations for one net, from observe.toml [ladder]."""
    binary = cfg["binary"] or default_binary()
    cmds = []
    for spec in cfg["anchors"]:
        anchor = spec
        if spec != "random" and not spec.startswith("stockfish:"):
            anchor = resolve(spec)
        cmd = [
            binary,
            "match",
            "--net",
            net_path,
            "--anchor",
            anchor,
            "--games",
            str(cfg["games"]),
            "--nodes",
            str(cfg["nodes"]),
        ]
        if cfg["openings"]:
            cmd += ["--openings", resolve(cfg["openings"])]
        cmds.append((spec, cmd))
    suite = resolve(cfg["tactics"]) if cfg["tactics"] else str(DEFAULT_SUITE)
    cmds.append(
        (
            "tactics",
            [
                binary,
                "tactics",
                "--net",
                net_path,
                "--suite",
                suite,
                "--nodes",
                str(cfg["tactics_nodes"]),
            ],
        )
    )
    return cmds


def run_ladder(cfg, net_path, log=print):
    """Run every rung for `net_path`; returns the parsed results (failed rungs skipped)."""
    results = []
    for spec, cmd in ladder_commands(cfg, net_path):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
        except (OSError, subprocess.SubprocessError) as e:
            log(f"[ladder][Warning] {spec}: {e!r}")
            continue
        result = parse_result(out.stdout)
        if out.returncode != 0 or result is None:
            tail = (out.stderr or out.stdout).strip().splitlines()[-3:]
            log(f"[ladder][Warning] {spec} failed (exit {out.returncode}): {tail}")
            continue
        results.append(result)
    return results


class LadderRunner:
    """Runs the ladder in the background for accepted nets; at most one at a time."""

    def __init__(self, cfg, telemetry=None, jsonl_path=None):
        self.cfg = cfg
        self.telemetry = telemetry
        self.jsonl_path = jsonl_path or data_path("hidden/ladder.jsonl")
        self._thread = None

    def due(self, accepted_gen):
        every = int(self.cfg["every_accepted_nets"])
        return bool(self.cfg["enabled"]) and every > 0 and accepted_gen % every == 0

    def maybe_run(self, net_path, accepted_gen, samples_trained):
        if not self.due(accepted_gen):
            return False
        if self._thread is not None and self._thread.is_alive():
            print(f"[ladder] still evaluating the previous net; skipping gen {accepted_gen}")
            return False
        self._thread = threading.Thread(
            target=self._run,
            args=(net_path, accepted_gen, samples_trained),
            daemon=True,
            name="ladder",
        )
        self._thread.start()
        print(f"[ladder] evaluating gen {accepted_gen} ({net_path}) in the background")
        return True

    def _run(self, net_path, accepted_gen, samples_trained):
        started = time.time()
        results = run_ladder(self.cfg, net_path)
        entry = {
            "time": time.time(),
            "seconds": time.time() - started,
            "net": net_path,
            "gen": accepted_gen,
            "samples_trained": samples_trained,
            "results": results,
        }
        try:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            print(f"[ladder][Warning] could not append to {self.jsonl_path}: {e!r}")
        row = result_rows(results)
        if row and self.telemetry is not None:
            row["ladder/gen"] = accepted_gen
            self.telemetry.log(row, samples_trained=samples_trained)
        print(f"[ladder] gen {accepted_gen}: {row}")

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)


def main(argv=None):
    from telemetry import load_observe_config

    p = argparse.ArgumentParser(description="Run the anchor ladder on nets (no W&B).")
    p.add_argument("nets", nargs="+")
    args = p.parse_args(argv)
    cfg = load_observe_config()["ladder"]
    for net in args.nets:
        results = run_ladder(cfg, resolve(net))
        print(json.dumps({"net": net, **result_rows(results)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
