# Invariants

What TrueZero must keep true, and the test that checks each one. Python's trainer
(`src/python/lib`, inherited from [kZero](https://github.com/KarelPeeters/kZero)) is the
source of truth for the data format: the Rust side is adapted to it, never the reverse.

If you change any code named here, run both suites:

```sh
# Rust (from src/rust; see "Rust test setup" below for libtorch)
cargo test --lib
# Python (from src/python; needs torch, chess, PyQt5, darkdetect, pyqtgraph)
PYTHONPATH=. python -m unittest discover -s ../../tests
```

Rust tests in `src/rust/invariant_tests.rs` use a **stub network**: a pure function of the input
tensor with a uniform policy (unless a test needs to tell moves apart) and either a constant
or a material-count value. No model file is needed, and each expected value is recomputed
from the position itself.

## Conventions

- **Side to move (POV).** The network sees the board from the side to move: "us" planes
  first, ranks flipped for Black. Its outputs `[v_logit, w, d, l, moves_left]` are from the
  side to move's point of view.
- **Absolute.** The search tree and cache store values from White's point of view.
  `ZeroValuesPov::to_absolute(stm)` and `ZeroValuesAbs::to_relative(player)` convert.
- **Records.** z (final), q (search) and v (net) in replay files are from the point of view
  of the side to move in that record.

## Search (Rust)

| Id | Invariant | Test |
|----|-----------|------|
| S1 | Every expanded node stores the net output for its own position, converted to absolute: `net_evaluation.to_relative(stm)` equals tanh / softmax of the stub's output for that position. | `invariant_tests::s1_tree_stores_the_side_to_move_net_output_as_absolute` |
| S2 | A terminal node is scored from the result alone, in absolute terms: White win (+1, [1,0,0]), Black win (−1, [0,0,1]), draw (0, [0,1,0]), moves left 0. The net is never asked. | `invariant_tests::s2_terminal_nodes_are_scored_by_the_winner`, `tests::mated_root_is_loss_for_{white,black}` |
| S3 | With a material-only evaluator and a uniform prior, search wins a hanging queen for either colour and reports root Q > 0 for the side to move. With a value-0 uniform stub it finds mate in one for either colour. | `invariant_tests::s3_search_wins_a_hanging_queen_for_either_colour`, `tests::mcts_finds_mate_in_one_{white,black}` |
| B1 | Backup adds absolute values with no sign change: `visits = own + Σ child visits` (own ≥ 1), `total = own·net + Σ child totals` for value and WDL, and moves left gains 1 per ply of depth. | `invariant_tests::b1_backup_conserves_visits_and_values`, `tests::test_backup_adds_no_extra_move_at_evaluated_node` |
| P1 | Child *i* of a node holds the prior of its own move: softmax over legal moves of `logits[policy_index(move)]`; `p^children_pst` renormalised below the root; `move_idx` lists the indices in child order. | `invariant_tests::p1_child_priors_follow_their_own_policy_index` |
| C1 | Root FPU and c_puct apply to the root's children; the children's settings apply deeper. | `tests::test_root_settings_apply_to_root_children`, `tests::test_puct_with_nonzero_fpu` |
| E1 | Early stop (`get_move_with_early_stop`) returns the move a full search plays, never uses more visits, and stops early when one move is clearly best. | `invariant_tests::e1_early_stop_plays_the_full_search_move` |
| E2 | `Tree::root_move_decided` is exact: once it holds, the most-visited root child is the unique most-visited child at the end of the search. | `invariant_tests::e2_root_move_decided_never_changes_the_final_move` |

## Playout cap randomisation (Rust self-play)

| Id | Invariant | Test |
|----|-----------|------|
| PCR1 | A fast search caps visits at `min(fast_visits, max_nodes)` and has no root noise; a full search uses the settings unchanged. | `invariant_tests::pcr1_move_settings_for_full_and_fast_searches` |
| PCR2 | A record with `is_full_search` used the full node count; every other record used at most the fast cap (exactly the cap while moves are sampled, since early stop applies only to greedy moves). p = 0 gives only fast searches, p = 1 only full. The flag and `zero_visits` reach the file. | `invariant_tests::pcr2_selfplay_records_full_and_fast_searches` |

Fast-search records keep their search visit distribution as the policy target: the inherited
trainer does not read `is_full_search`, and leaving positions out would break the moves-left
indexing (R7).

## Self-play records (Rust writer)

Checked on real `DataGen::play_game` games with the stub, written by `BinaryOutput` and read
back byte by byte: `invariant_tests::r_selfplay_records_match_search_inputs_and_results`.

| Id | Invariant |
|----|-----------|
| R1 | The stored input is exactly the network input (`convert_board`), with the raw halfmove clock. |
| R2 | Consecutive records are linked by the played move; `played_mv` is that move's policy index. |
| R3 | The policy target is `(policy_index(move), child visits / Σ child visits)` for every legal move. |
| R4 | v is the net's side-to-move output: with a stub that says "side to move is better", v > 0 and w > l for both colours. |
| R5 | z is the game result from the side to move's point of view: +1 for the winner, −1 for the loser, 0 for a draw or a move-limit game. |
| R6 | q, v and z satisfy `value = w − l`, so values and WDL flip together. |
| R7 | Moves left is `game_length + 1 − pos_index`, and 0 at the appended final position (KZero; the sampler finds the final position at `pi + final_moves_left − 1`). |

## KZero file format

| Id | Invariant | Test |
|----|-----------|------|
| K1 | TrueZero's writer produces the same `.bin`, `.off` and `.json` as kZero's own `BinaryOutput` + `ChessStdMapper` for the same games (castling as e1g1/e1c1, en passant only when capturable, castling scalars kingside then queenside, raw halfmove, Black rank-flipped). | `fileformat::tests::writes_the_same_records_as_kzero` against `tests/fixtures/kzero/` |
| K2 | The inherited checker, sampler and trainer accept that file and train on it. | `KZeroFileTests` in `tests/test_kzero_interop.py` |
| K3 | `decoder.py` (python-chess) encodes inputs and policy indices exactly like the file. | `PythonEncoderTests.test_inputs_and_policy_indices_match_kzero` |
| K4 | The halfmove clock is scaled by 1/100 inside `TrueNet`, not in the data; scripted and eager nets agree. | `test_replay_contract.py`: `test_stored_input_is_the_network_input`, `test_network_scales_the_raw_halfmove_clock` |

R1 holds even if both the writer and `convert_board` scaled the halfmove clock; K1 is what
pins the raw value.

## Trainer targets (Python)

`TrainerTargetTests.test_batch_targets_match_the_games` builds the trainer's `PositionBatch`
from the fixture and compares it with python-chess replays of the same games.

| Id | Invariant |
|----|-----------|
| T1 | `batch.input_full` is the encoder's network input for the position. |
| T2 | `v_final`, `wdl_final` are the result from the side to move's point of view (checkmate decides the winner; anything else is a draw). |
| T3 | `moves_left_final` counts down to 0 at the appended final position. |
| T4 | `played_mv` is the policy index of the move actually played; the final position has `played_mv = −1` and no policy target. |
| T5 | The policy target sums to 1 over exactly the legal moves; padding is −1. |

## Match play and UCI

| Id | Invariant | Test |
|----|-----------|------|
| M1 | The winner of a finished game is the side **not** to move at checkmate, regardless of ply count or which colour started (`BoardStack::winner`). | `invariant_tests::m1_winner_is_the_side_not_to_move_at_checkmate` |
| M2 | In `trainenginetest`, engine 0 (the candidate) plays White iff `engine0_white`, chosen from the side to move (`boardmanager::engine_index`). | `invariant_tests::m2_engine_is_chosen_by_side_to_move` |
| U1 | UCI `go nodes N` is honoured; with a clock too, the stricter limit applies; with neither, 1600 visits. | `invariant_tests::u1_go_nodes_sets_the_visit_limit` |

`enginetest` attributes results by the engine that made the last move, which is correct
without `winner()`; it is not unit-tested.

## Known gaps

- The Rust crate still links libtorch: `executor::Packet` carries `tch::Tensor`, so the stub
  builds tensors even though no model is loaded. Making the tests torch-free means
  feature-gating `tch` behind the evaluation boundary.
- FPU is absolute (children 0.5 in self-play); kZero uses relative FPU (parent Q + 0). Not a
  correctness bug, but it changes how deep the search goes.
- `decoder.py`'s visualisation path is not covered beyond K3.

## Rust test setup

`tch` downloads libtorch at build time. If that is blocked, use a PyPI torch wheel:

```sh
uv venv /tmp/tzvenv -p 3.12 && VIRTUAL_ENV=/tmp/tzvenv uv pip install torch==2.4.0
PATH=/tmp/tzvenv/bin:$PATH LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
  LD_LIBRARY_PATH=/tmp/tzvenv/lib/python3.12/site-packages/torch/lib TZ_DEVICE=cpu \
  cargo test --lib
```

Do not point `LIBTORCH` at the wheel directly: the wheel uses the pre-C++11 ABI.
