"""The anchor ladder's Python side: command lines, result parsing, scheduling, and the
bundled mate suite's correctness (proved again with python-chess).

Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import chess

import ladder
from eval.make_mate_suite import forces_mate_in_two, mates_in_one
import runconfig

LADDER_DEFAULTS = runconfig.config_from_texts(runconfig.EXAMPLE_PATH.read_text(), "")["ladder"]

MATCH = {
    "kind": "match",
    "anchor": "stockfish:C:\\sf\\stockfish.exe:3",
    "games": 100,
    "wins": 60,
    "draws": 20,
    "losses": 20,
    "score": 0.7,
    "elo": 147.2,
    "elo_low": 80.0,
    "elo_high": 220.0,
}
TACTICS = {
    "kind": "tactics",
    "solved_fraction": 0.86,
    "by_category": {"mate1": 0.96, "mate2": 0.76},
}


def cfg(**kw):
    c = dict(LADDER_DEFAULTS)
    c.update(kw)
    return c


class LadderTests(unittest.TestCase):
    def test_result_parsing_and_rows(self):
        out = "noise\nLADDER_RESULT " + json.dumps(MATCH) + "\n"
        self.assertEqual(ladder.parse_result(out), MATCH)
        self.assertIsNone(ladder.parse_result("panic"))
        row = ladder.result_rows([MATCH, TACTICS])
        self.assertEqual(row["ladder/stockfish_skill3/elo"], 147.2)
        self.assertEqual(row["ladder/tactics/mate2"], 0.76)
        self.assertEqual(row["ladder/tactics/solved_fraction"], 0.86)

    def test_anchor_names(self):
        self.assertEqual(ladder.anchor_name("random"), "random")
        self.assertEqual(ladder.anchor_name("anchors/tz_18.pt"), "tz_18")
        self.assertEqual(ladder.anchor_name("stockfish:C:\\sf\\sf.exe:0"), "stockfish_skill0")

    def test_commands(self):
        cmds = ladder.ladder_commands(
            cfg(anchors=["random", "anchors/tz_18.pt"], binary="lad"), "/n/tz_40.pt"
        )
        self.assertEqual([c[0] for c in cmds], ["random", "anchors/tz_18.pt", "tactics"])
        self.assertTrue(os.path.isabs(cmds[1][1][5]))  # anchor net resolved to the repository root
        self.assertIn(str(ladder.DEFAULT_SUITE), cmds[2][1])

    def test_due_every_n_accepted_and_one_at_a_time(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        r = ladder.LadderRunner(
            cfg(enabled=True, every_accepted_nets=5), jsonl_path=os.path.join(d, "l.jsonl")
        )
        self.assertFalse(r.due(4))
        self.assertTrue(r.due(10))
        self.assertFalse(ladder.LadderRunner(cfg(enabled=False)).due(10))
        logged = []

        class T:
            def log(self, row, samples_trained=None):
                logged.append((row, samples_trained))

        r.telemetry = T()
        with mock.patch.object(ladder, "run_ladder", return_value=[MATCH, TACTICS]):
            self.assertTrue(r.maybe_run("tz_10.pt", 10, 123456))
            r.join()
        row, samples = logged[0]
        self.assertEqual(samples, 123456)
        self.assertEqual(row["ladder/gen"], 10)
        with open(os.path.join(d, "l.jsonl")) as f:
            self.assertEqual(json.loads(f.readline())["gen"], 10)


class MateSuiteTests(unittest.TestCase):
    def test_every_listed_solution_is_proven(self):
        n = {"mate1": 0, "mate2": 0}
        with open(ladder.DEFAULT_SUITE) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                fen, sols, ident = [x.strip() for x in line.split("|")]
                board = chess.Board(fen)
                kind = ident.split("-")[0]
                n[kind] += 1
                moves = [chess.Move.from_uci(s) for s in sols.split()]
                if kind == "mate1":
                    self.assertEqual(set(mates_in_one(board)), set(moves), ident)
                else:
                    self.assertEqual(mates_in_one(board), [], ident)  # exactly mate in 2
                    for mv in moves:
                        self.assertTrue(forces_mate_in_two(board, mv), ident)
        self.assertEqual(n, {"mate1": 50, "mate2": 50})


if __name__ == "__main__":
    unittest.main()
