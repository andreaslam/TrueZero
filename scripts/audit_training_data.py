#!/usr/bin/env python3
"""Audit TrueZero chess training data without using the project data loaders.

Requires python-chess for independent legal-move generation. The dataset itself
is parsed directly from its JSON, offset, and binary files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import NoReturn

try:
    import chess
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: chess. Run `uv sync` at the project root, "
        "then `uv run python scripts/audit_training_data.py ...`."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
RUST_MOVE_TABLE = ROOT / "src/rust/mvs.rs"
SCALAR_NAMES = [
    "game_id",
    "pos_index",
    "game_length",
    "zero_visits",
    "is_full_search",
    "is_final_position",
    "is_terminal",
    "hit_move_limit",
    "available_mv_count",
    "played_mv",
    "kdl_policy",
    "final_v",
    "final_wdl_w",
    "final_wdl_d",
    "final_wdl_l",
    "final_moves_left",
    "zero_v",
    "zero_wdl_w",
    "zero_wdl_d",
    "zero_wdl_l",
    "zero_moves_left",
    "net_v",
    "net_wdl_w",
    "net_wdl_d",
    "net_wdl_l",
    "net_moves_left",
]
INTEGER_FIELDS = {
    "game_id",
    "pos_index",
    "game_length",
    "zero_visits",
    "is_full_search",
    "is_final_position",
    "is_terminal",
    "hit_move_limit",
    "available_mv_count",
    "played_mv",
}
PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
RESULT_WDL = {
    "win": (1.0, 0.0, 0.0),
    "draw": (0.0, 1.0, 0.0),
    "loss": (0.0, 0.0, 1.0),
}


class Audit:
    def __init__(self, max_examples: int):
        self.failures = Counter()
        self.examples = {}
        self.max_examples = max_examples
        self.checks = Counter()

    def check(self, category: str):
        self.checks[category] += 1

    def fail(self, category: str, message: str):
        self.failures[category] += 1
        examples = self.examples.setdefault(category, [])
        if len(examples) < self.max_examples:
            examples.append(message)

    def summary(self):
        print("\nAudit summary:")
        for category in (
            "Data integrity",
            "Value targets",
            "Policy targets",
            "Board transformations",
            "Game sequencing",
        ):
            violations = self.failures[category]
            status = "FAIL" if violations else "PASS"
            print(
                f"  {category}: {status} "
                f"({violations} violations; {self.checks[category]} checks)"
            )
            for example in self.examples.get(category, []):
                print(f"    - {example}")


def fail_fast(message: str) -> NoReturn:
    raise SystemExit(f"Cannot audit dataset: {message}")


def dataset_stem(path: Path) -> Path:
    if path.suffix in {".bin", ".off", ".json"}:
        return path.with_suffix("")
    return path


def load_metadata(stem: Path):
    meta_path = stem.with_suffix(".json")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail_fast(f"cannot read {meta_path}: {exc}")


def read_move_table(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        fail_fast(f"cannot read policy move table {path}: {exc}")
    marker = "let contents: [&'static str; 1880] = ["
    start = text.find(marker)
    if start < 0:
        fail_fast(f"could not find the 1880-entry move list in {path}")
    end = text.find("];", start)
    if end < 0:
        fail_fast(f"unterminated move list in {path}")
    strings = re.findall(r'"([a-h][1-8][a-h][1-8][qrbn]?)"', text[start:end])
    if len(strings) != 1880:
        fail_fast(f"expected 1880 move strings, found {len(strings)}")
    moves = [chess.Move.from_uci(move) for move in strings]
    if len(set(moves)) != len(moves):
        fail_fast("policy move table contains duplicate moves")
    return moves, {move: index for index, move in enumerate(moves)}


def read_offsets(path: Path, expected_count: int):
    raw = path.read_bytes()
    if len(raw) % 8:
        fail_fast(f"{path} has {len(raw)} bytes, not a multiple of 8")
    values = [value[0] for value in struct.iter_unpack("<Q", raw)]
    if len(values) != expected_count:
        fail_fast(f"{path} has {len(values)} offsets, expected {expected_count}")
    return values


def parse_position(data: bytes, start: int, end: int, meta: dict, file_pi: int):
    names = meta["scalar_names"]
    scalar_bytes = 4 * len(names)
    bool_count = math.prod(meta["input_bool_shape"])
    bool_bytes = (bool_count + 7) // 8
    scalar_count = int(meta["input_scalar_count"])
    fixed_bytes = scalar_bytes + bool_bytes + 4 * scalar_count
    if start < 0 or end < start or end > len(data) or end - start < fixed_bytes:
        raise ValueError(f"invalid record boundary [{start}, {end})")

    scalars = struct.unpack_from(f"<{len(names)}f", data, start)
    fields = dict(zip(names, scalars))
    count_float = fields["available_mv_count"]
    if not math.isfinite(count_float) or count_float < 0 or not count_float.is_integer():
        raise ValueError(f"invalid available_mv_count {count_float!r}")
    move_count = int(count_float)
    expected_end = start + fixed_bytes + 8 * move_count
    if end != expected_end:
        raise ValueError(
            f"record length {end - start} does not match expected "
            f"{expected_end - start} bytes (policy count {move_count})"
        )

    cursor = start + scalar_bytes
    packed = data[cursor : cursor + bool_bytes]
    bits = [
        (byte >> bit) & 1
        for byte in packed
        for bit in range(8)
    ][:bool_count]
    cursor += bool_bytes
    input_scalars = struct.unpack_from(f"<{scalar_count}f", data, cursor)
    cursor += 4 * scalar_count
    policy_indices = (
        struct.unpack_from(f"<{move_count}I", data, cursor) if move_count else ()
    )
    cursor += 4 * move_count
    policy_values = (
        struct.unpack_from(f"<{move_count}f", data, cursor) if move_count else ()
    )
    return {
        "file_pi": file_pi,
        "fields": fields,
        "bits": bits,
        "input_scalars": input_scalars,
        "policy_indices": policy_indices,
        "policy_values": policy_values,
        "start": start,
        "end": end,
    }


def flip_rank_square(square: int) -> int:
    return square ^ 56


def flip_180_square(square: int) -> int:
    return 63 - square


def transform_move(move: chess.Move, transform):
    return chess.Move(
        transform(move.from_square),
        transform(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


def canonical_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    return transform_move(move, flip_rank_square) if turn == chess.BLACK else move


def cozy_policy_move(board: chess.Board, move: chess.Move) -> chess.Move:
    if not board.is_castling(move):
        return move
    rank = 0 if board.turn == chess.WHITE else 7
    rook_file = 7 if board.is_kingside_castling(move) else 0
    return chess.Move(move.from_square, chess.square(rook_file, rank))


def decode_policy_move(board: chess.Board, move: chess.Move) -> chess.Move:
    actual = transform_move(move, flip_rank_square) if board.turn == chess.BLACK else move
    rook = board.piece_at(actual.to_square)
    king = board.piece_at(actual.from_square)
    home_rank = 0 if board.turn == chess.WHITE else 7
    if (
        king == chess.Piece(chess.KING, board.turn)
        and rook == chess.Piece(chess.ROOK, board.turn)
        and chess.square_rank(actual.from_square) == home_rank
        and chess.square_rank(actual.to_square) == home_rank
        and chess.square_file(actual.to_square) in (0, 7)
        and board.castling_rights & chess.BB_SQUARES[actual.to_square]
    ):
        king_file = 2 if chess.square_file(actual.to_square) == 0 else 6
        return chess.Move(actual.from_square, chess.square(king_file, home_rank))
    return actual


def expected_wdl(winner, perspective):
    if winner is None:
        return RESULT_WDL["draw"]
    return RESULT_WDL["win"] if winner == perspective else RESULT_WDL["loss"]


def piece_bits(board: chess.Board, turn: chess.Color, transform):
    result = [0] * (13 * 64)
    channel = 0
    for color in (turn, not turn):
        for piece_type in PIECE_TYPES:
            for square in board.pieces(piece_type, color):
                mapped = transform(square) if turn == chess.BLACK else square
                result[channel * 64 + mapped] = 1
            channel += 1

    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        if transform is flip_rank_square:
            victim = (
                board.ep_square - 8
                if turn == chess.WHITE
                else board.ep_square + 8
            )
            if turn == chess.BLACK:
                victim = flip_rank_square(victim)
            result[12 * 64 + victim] = 1
        elif turn == chess.WHITE:
            result[12 * 64 + chess.square(ep_file, 3)] = 1
        else:
            result[12 * 64 + 63 - chess.square(ep_file, 4)] = 1
    return result


def expected_input_scalars(board: chess.Board):
    turn = board.turn
    opponent = not turn
    return (
        float(turn == chess.WHITE),
        float(turn == chess.BLACK),
        float(board.has_queenside_castling_rights(turn)),
        float(board.has_kingside_castling_rights(turn)),
        float(board.has_queenside_castling_rights(opponent)),
        float(board.has_kingside_castling_rights(opponent)),
        float(repetition_count(board)),
        float(board.halfmove_clock),
    )


def repetition_count(board: chess.Board) -> int:
    key = board._transposition_key()
    probe = board.copy(stack=True)
    count = 0
    while probe.move_stack:
        probe.pop()
        if probe._transposition_key() == key:
            count += 1
    return count


def decode_stored_board(record, transform):
    """Reconstruct a board from a record using the selected inverse transform."""
    values = record["input_scalars"]
    white_to_move = values[0] == 1.0
    if values[0] + values[1] != 1.0:
        raise ValueError(f"invalid side-to-move scalar pair {values[:2]}")
    turn = chess.WHITE if white_to_move else chess.BLACK
    board = chess.Board(None)
    board.turn = turn
    bits = record["bits"]
    channel = 0
    for color in (turn, not turn):
        for piece_type in PIECE_TYPES:
            for canonical_square in range(64):
                if bits[channel * 64 + canonical_square]:
                    square = (
                        transform(canonical_square) if turn == chess.BLACK else canonical_square
                    )
                    board.set_piece_at(square, chess.Piece(piece_type, color))
            channel += 1

    for color, base in ((turn, 2), (not turn, 4)):
        rank = 0 if color == chess.WHITE else 7
        if values[base] == 1.0:
            board.castling_rights |= chess.BB_SQUARES[chess.square(0, rank)]
        if values[base + 1] == 1.0:
            board.castling_rights |= chess.BB_SQUARES[chess.square(7, rank)]

    ep_squares = [index for index, bit in enumerate(bits[12 * 64 : 13 * 64]) if bit]
    if ep_squares:
        if len(ep_squares) != 1:
            raise ValueError(f"expected at most one en-passant square, got {len(ep_squares)}")
        square = ep_squares[0]
        if turn == chess.BLACK:
            square = transform(square)
        file = square % 8
        rank = square // 8
        if turn == chess.WHITE and rank == 4:
            board.ep_square = chess.square(file, 5)
        elif turn == chess.BLACK and rank == 3:
            board.ep_square = chess.square(file, 2)
    board.halfmove_clock = int(values[7])
    board.fullmove_number = 1
    return board


def rust_draw_condition(board: chess.Board) -> bool:
    pieces = list(board.piece_map().values())
    bishops = sum(piece.piece_type == chess.BISHOP for piece in pieces)
    knights = sum(piece.piece_type == chess.KNIGHT for piece in pieces)
    return (
        len(pieces) == 2
        or (len(pieces) <= 3 and (bishops == 1 or knights == 1))
        or board.halfmove_clock == 100
        or board.is_repetition(3)
        or board.is_stalemate()
    )


def actual_winner(board: chess.Board):
    if board.is_checkmate():
        return not board.turn
    if rust_draw_condition(board):
        return None
    return "ongoing"


def close_enough(actual: float, expected: float, tolerance: float = 1e-4):
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def compare_bits(actual, expected):
    return [index for index, (left, right) in enumerate(zip(actual, expected)) if left != right]


def transform_bits(bits, transform):
    return [
        bits[channel * 64 + transform(square)]
        for channel in range(13)
        for square in range(64)
    ]


def audit(stem: Path, max_examples: int):
    stem = dataset_stem(stem).absolute()
    meta = load_metadata(stem)
    bin_path, off_path = stem.with_suffix(".bin"), stem.with_suffix(".off")
    try:
        data = bin_path.read_bytes()
        bin_hash = hashlib.sha256(data).hexdigest()
    except OSError as exc:
        fail_fast(f"cannot read {bin_path}: {exc}")
    if not isinstance(meta.get("scalar_names"), list):
        fail_fast("metadata scalar_names is missing or not a list")
    if not all(key in meta for key in ("position_count", "game_count")):
        fail_fast("metadata lacks position_count or game_count")

    position_count = int(meta["position_count"])
    game_count = int(meta["game_count"])
    offsets_per_position = position_count + (
        game_count if meta.get("includes_game_start_indices", False) else 0
    )
    offsets = read_offsets(off_path, offsets_per_position)
    pos_offsets = offsets[:position_count]
    game_starts = (
        offsets[position_count:] if meta.get("includes_game_start_indices", False) else []
    )
    moves, move_to_index = read_move_table(RUST_MOVE_TABLE)

    report = Audit(max_examples)
    required_keys = {
        "game",
        "input_bool_shape",
        "input_scalar_count",
        "policy_shape",
        "position_count",
        "game_count",
        "min_game_length",
        "max_game_length",
        "scalar_names",
    }
    report.check("Data integrity")
    missing = sorted(required_keys - set(meta))
    if missing:
        report.fail("Data integrity", f"metadata missing keys: {missing}")
    if meta.get("game") != "chess":
        report.fail("Data integrity", f"expected chess dataset, got {meta.get('game')!r}")
    if meta.get("input_bool_shape") != [13, 8, 8]:
        report.fail("Data integrity", f"unexpected input_bool_shape {meta.get('input_bool_shape')}")
    if meta.get("input_scalar_count") != 8:
        report.fail("Data integrity", f"unexpected input_scalar_count {meta.get('input_scalar_count')}")
    if meta.get("policy_shape") != [1880]:
        report.fail("Data integrity", f"unexpected policy_shape {meta.get('policy_shape')}")
    if meta.get("includes_terminal_positions") is not True:
        report.fail("Data integrity", "dataset does not declare appended terminal positions")
    if meta["scalar_names"] != SCALAR_NAMES:
        report.fail("Data integrity", "scalar_names/order differs from the audited 26-field schema")
    if meta.get("includes_game_start_indices") and len(game_starts) != game_count:
        report.fail("Data integrity", f"found {len(game_starts)} game starts for {game_count} games")
    if not pos_offsets or pos_offsets[0] != 0:
        report.fail("Data integrity", f"first position offset is {pos_offsets[0] if pos_offsets else 'missing'}, expected 0")
    for index in range(1, len(pos_offsets)):
        if pos_offsets[index] <= pos_offsets[index - 1]:
            report.fail("Data integrity", f"position offsets not strictly increasing at index {index}")
    if pos_offsets and pos_offsets[-1] >= len(data):
        report.fail("Data integrity", f"final start offset {pos_offsets[-1]} is outside {len(data)}-byte .bin")
    if game_starts and (game_starts[0] != 0 or any(a >= b for a, b in zip(game_starts, game_starts[1:]))):
        report.fail("Data integrity", "game start indices are not strictly increasing from zero")
    if game_starts and game_starts[-1] >= position_count:
        report.fail("Data integrity", f"last game start {game_starts[-1]} is outside {position_count} positions")

    report.check("Board transformations")
    for square in chess.SQUARES:
        if (
            flip_180_square(flip_180_square(square)) != square
            or flip_rank_square(flip_rank_square(square)) != square
        ):
            report.fail("Board transformations", f"square transform is not involutive at square {square}")
    for index, move in enumerate(moves):
        if transform_move(
            transform_move(move, flip_rank_square), flip_rank_square
        ) != move:
            report.fail("Board transformations", f"rank-flip move transform is not involutive at index {index}")

    records = []
    allowed_nan_counts = Counter()
    invalid_integer_field = False
    for pi, start in enumerate(pos_offsets):
        end = pos_offsets[pi + 1] if pi + 1 < position_count else len(data)
        try:
            record = parse_position(data, start, end, meta, pi)
        except (ValueError, KeyError, struct.error) as exc:
            report.fail("Data integrity", f"position {pi}: {exc}")
            continue
        records.append(record)
        report.check("Data integrity")
        for name in INTEGER_FIELDS:
            value = record["fields"].get(name, math.nan)
            if not math.isfinite(value) or not value.is_integer():
                report.fail("Data integrity", f"position {pi}: {name}={value!r} is not a finite integer")
                invalid_integer_field = True
            elif name.startswith("is_") or name == "hit_move_limit":
                if value not in (0.0, 1.0):
                    report.fail("Data integrity", f"position {pi}: {name}={value} is not a boolean flag")
                    invalid_integer_field = True
        for name, value in record["fields"].items():
            if math.isinf(value):
                report.fail("Data integrity", f"position {pi}: {name} is infinite")
            elif math.isnan(value):
                expected_nan = name == "kdl_policy" or (
                    bool(record["fields"]["is_final_position"])
                    and name in {
                        "zero_v",
                        "zero_wdl_w",
                        "zero_wdl_d",
                        "zero_wdl_l",
                        "zero_moves_left",
                        "net_v",
                        "net_wdl_w",
                        "net_wdl_d",
                        "net_wdl_l",
                        "net_moves_left",
                    }
                )
                if not expected_nan:
                    report.fail("Data integrity", f"position {pi}: unexpected NaN in {name}")
                else:
                    allowed_nan_counts[name] += 1
        if any(not math.isfinite(value) for value in record["input_scalars"]):
            report.fail("Data integrity", f"position {pi}: non-finite input scalar")
        if any(index >= 1880 for index in record["policy_indices"]):
            bad = next(index for index in record["policy_indices"] if index >= 1880)
            report.fail("Data integrity", f"position {pi}: policy index {bad} outside [0,1880)")
        if any(not math.isfinite(value) for value in record["policy_values"]):
            report.fail("Data integrity", f"position {pi}: non-finite policy value")
        if any(value < 0.0 for value in record["policy_values"]):
            report.fail("Data integrity", f"position {pi}: negative policy probability")
        if record["policy_values"] and not close_enough(sum(record["policy_values"]), 1.0, 1e-3):
            report.fail("Data integrity", f"position {pi}: policy mass {sum(record['policy_values']):.7g}, expected 1")
        if len(set(record["policy_indices"])) != len(record["policy_indices"]):
            report.fail("Data integrity", f"position {pi}: duplicate policy indices")

    report.check("Data integrity")
    if len(records) != position_count:
        report.fail("Data integrity", f"parsed {len(records)}/{position_count} position records")
    if len(data) != (records[-1]["end"] if records else 0):
        report.fail("Data integrity", f"last record ends at {records[-1]['end'] if records else 0}, .bin has {len(data)} bytes")

    if len(records) != position_count or not records:
        report.summary()
        return 1
    if invalid_integer_field:
        report.summary()
        return 1
    for transform in (flip_180_square, flip_rank_square):
        if transform_bits(transform_bits(records[0]["bits"], transform), transform) != records[0]["bits"]:
            report.fail("Board transformations", "flipping the first stored board twice did not restore it")
            break

    if meta.get("includes_game_start_indices"):
        expected_starts = [
            index for index, record in enumerate(records)
            if record["fields"]["pos_index"] == 0
        ]
        if game_starts != expected_starts:
            report.fail(
                "Data integrity",
                f"game-start table differs from pos_index==0 records "
                f"(stored head={game_starts[:5]}, expected head={expected_starts[:5]})",
            )

    boundaries = list(game_starts)
    if not boundaries:
        boundaries = [
            index for index, record in enumerate(records)
            if record["fields"]["pos_index"] == 0
        ]
    if len(boundaries) != game_count:
        report.fail(
            "Data integrity",
            f"could not establish {game_count} game boundaries; found {len(boundaries)} starts",
        )
        report.summary()
        return 1

    lengths = []
    result_counts = Counter()
    game_results = []
    hit_move_limit_count = 0
    total_policy_positions = 0
    policy_entry_count = 0
    played_move_count = 0
    python_board_mismatch_count = 0
    rust_board_mismatch_count = 0
    cross_transform_mismatch_count = 0
    python_white_mismatch_count = 0
    python_black_mismatch_count = 0
    python_piece_plane_mismatch_count = 0
    python_ep_plane_mismatch_count = 0
    python_policy_transform_mismatch_count = 0
    custom_start_game_count = 0

    for gi, start_pi in enumerate(boundaries):
        end_pi = boundaries[gi + 1] if gi + 1 < game_count else position_count
        game_records = records[start_pi:end_pi]
        if not game_records:
            report.fail("Data integrity", f"game {gi}: empty position span")
            continue
        root = game_records[0]
        length_value = root["fields"]["game_length"]
        if not math.isfinite(length_value) or not length_value.is_integer():
            report.fail("Data integrity", f"game {gi}, position {start_pi}: invalid game_length {length_value}")
            continue
        game_length = int(length_value)
        lengths.append(game_length)
        if len(game_records) != game_length + 1:
            report.fail("Data integrity", f"game {gi}: span has {len(game_records)} positions, expected {game_length + 1}")

        board = chess.Board()
        start_board = decode_stored_board(root, flip_rank_square)
        standard_start = (
            start_board.board_fen() == board.board_fen()
            and start_board.turn == board.turn
            and start_board.castling_rights == board.castling_rights
            and start_board.ep_square is None
            and start_board.halfmove_clock == 0
        )
        if not standard_start:
            custom_start_game_count += 1
            board = start_board

        moves_for_game = []
        game_bad = False
        for local_pi, record in enumerate(game_records):
            pi = start_pi + local_pi
            fields = record["fields"]
            report.check("Game sequencing")
            report.check("Value targets")
            report.check("Policy targets")
            report.check("Board transformations")

            expected_index = local_pi
            if int(fields["game_id"]) != gi:
                report.fail("Data integrity", f"game {gi}, position {pi}: game_id={fields['game_id']}, expected {gi}")
            if int(fields["pos_index"]) != expected_index:
                report.fail("Game sequencing", f"game {gi}, position {pi}: pos_index={fields['pos_index']}, expected {expected_index}")
            if int(fields["game_length"]) != game_length:
                report.fail("Game sequencing", f"game {gi}, position {pi}: game_length={fields['game_length']}, expected {game_length}")
            if bool(fields["is_final_position"]) != (local_pi == game_length):
                report.fail("Game sequencing", f"game {gi}, position {pi}: final flag={bool(fields['is_final_position'])}, expected {local_pi == game_length}")
            if not close_enough(fields["final_moves_left"], game_length - local_pi):
                report.fail("Value targets", f"game {gi}, position {pi}: final_moves_left={fields['final_moves_left']}, expected {game_length - local_pi}")

            expected_scalars = expected_input_scalars(board)
            for channel, (actual, expected) in enumerate(zip(record["input_scalars"], expected_scalars)):
                if not standard_start and local_pi == 0 and channel == 6:
                    continue
                if not close_enough(actual, expected):
                    report.fail(
                        "Game sequencing",
                        f"game {gi}, position {pi}: input scalar[{channel}]={actual}, expected {expected}",
                    )

            expected_python_bits = piece_bits(board, board.turn, flip_180_square)
            expected_rust_bits = piece_bits(board, board.turn, flip_rank_square)
            python_diff = compare_bits(record["bits"], expected_python_bits)
            rust_diff = compare_bits(record["bits"], expected_rust_bits)
            if python_diff:
                python_board_mismatch_count += 1
                if any(index // 64 < 12 for index in python_diff):
                    python_piece_plane_mismatch_count += 1
                if any(index // 64 == 12 for index in python_diff):
                    python_ep_plane_mismatch_count += 1
                if board.turn == chess.BLACK:
                    python_black_mismatch_count += 1
                else:
                    python_white_mismatch_count += 1
                if not any(
                    example.startswith(f"Python board encoding differs at game {gi}, position {pi}:")
                    for example in report.examples.get("Board transformations", [])
                ):
                    report.fail(
                        "Board transformations",
                        f"Python board encoding differs at game {gi}, position {pi}: "
                        f"side={'white' if board.turn == chess.WHITE else 'black'}, "
                        f"FEN={board.fen()}, {len(python_diff)} bit(s), plane "
                        f"{python_diff[0] // 64} square "
                        f"{chess.square_name(python_diff[0] % 64)} "
                        f"stored={record['bits'][python_diff[0]]} "
                        f"expected={expected_python_bits[python_diff[0]]}",
                    )
            if rust_diff:
                rust_board_mismatch_count += 1
                report.fail(
                    "Board transformations",
                    f"serialized board differs from Rust rank-flip encoding at game {gi}, position {pi}: "
                    f"{len(rust_diff)} bit(s), first flat plane index {rust_diff[0]}",
                )
            if board.turn == chess.BLACK and python_diff and not rust_diff:
                cross_transform_mismatch_count += 1

            final_v = fields["final_v"]
            final_wdl = (
                fields["final_wdl_w"],
                fields["final_wdl_d"],
                fields["final_wdl_l"],
            )
            if not all(math.isfinite(value) for value in (final_v, *final_wdl)):
                report.fail("Value targets", f"game {gi}, position {pi}: non-finite final target")
            else:
                if any(value not in (0.0, 1.0) for value in final_wdl) or not close_enough(sum(final_wdl), 1.0):
                    report.fail("Value targets", f"game {gi}, position {pi}: final WDL {final_wdl} is not one-hot")
                if not close_enough(final_v, final_wdl[0] - final_wdl[2]):
                    report.fail("Value targets", f"game {gi}, position {pi}: final_v={final_v}, W-L={final_wdl[0] - final_wdl[2]}")

            if local_pi == game_length:
                if record["policy_indices"] or record["policy_values"]:
                    report.fail("Policy targets", f"game {gi}, position {pi}: terminal record has policy entries")
                if int(fields["played_mv"]) != -1:
                    report.fail("Policy targets", f"game {gi}, position {pi}: terminal played_mv={fields['played_mv']}, expected -1")
                if bool(fields["is_terminal"]):
                    outcome = actual_winner(board)
                    if outcome == "ongoing":
                        report.fail("Game sequencing", f"game {gi}, position {pi}: marked terminal but replayed board is ongoing ({board.fen()})")
                else:
                    outcome = "ongoing"
                    hit_move_limit_count += 1
                    if not bool(fields["hit_move_limit"]):
                        report.fail("Game sequencing", f"game {gi}, position {pi}: non-terminal final board lacks hit_move_limit")
                if bool(fields["is_terminal"]) and bool(fields["hit_move_limit"]):
                    report.fail("Game sequencing", f"game {gi}, position {pi}: terminal and hit_move_limit are both true")

                if outcome == "ongoing":
                    expected_winner = None  # Rust labels a truncated game as a draw.
                    report.fail(
                        "Value targets",
                        f"game {gi}, position {pi}: unfinished game is assigned a draw target "
                        f"(v={fields['final_v']}, WDL="
                        f"{(fields['final_wdl_w'], fields['final_wdl_d'], fields['final_wdl_l'])})",
                    )
                else:
                    expected_winner = outcome
                result_counts[expected_winner] += 1
                game_results.append(expected_winner)
                for check_pi, check_record in enumerate(game_records):
                    check_fields = check_record["fields"]
                    replay_turn = board.turn if (game_length - check_pi) % 2 == 0 else not board.turn
                    value = check_fields["final_v"]
                    wdl = (
                        check_fields["final_wdl_w"],
                        check_fields["final_wdl_d"],
                        check_fields["final_wdl_l"],
                    )
                    check_expected_v = (
                        0.0 if expected_winner is None
                        else (1.0 if expected_winner == replay_turn else -1.0)
                    )
                    check_expected_wdl = expected_wdl(expected_winner, replay_turn)
                    if not close_enough(value, check_expected_v) or any(
                        not close_enough(actual, expected)
                        for actual, expected in zip(wdl, check_expected_wdl)
                    ):
                        report.fail(
                            "Value targets",
                            f"game {gi}, position {start_pi + check_pi}: "
                            f"stored (v={value}, WDL={wdl}), expected "
                            f"(v={check_expected_v}, WDL={check_expected_wdl})",
                        )

                break

            if bool(fields["is_terminal"]) or bool(fields["hit_move_limit"]):
                report.fail("Game sequencing", f"game {gi}, position {pi}: non-final position has terminal/move-limit flag")

            for label, prefix in (("zero", "zero"), ("net", "net")):
                value = fields[f"{prefix}_v"]
                wdl = tuple(
                    fields[f"{prefix}_wdl_{suffix}"]
                    for suffix in ("w", "d", "l")
                )
                if not math.isfinite(value) or not -1.0001 <= value <= 1.0001:
                    report.fail("Value targets", f"game {gi}, position {pi}: {label}_v={value} is non-finite or outside [-1,1]")
                if not all(math.isfinite(v) for v in wdl):
                    report.fail("Value targets", f"game {gi}, position {pi}: {label} WDL is non-finite before final record")
                elif any(v < -1e-5 for v in wdl) or not close_enough(sum(wdl), 1.0, 1e-3):
                    report.fail("Value targets", f"game {gi}, position {pi}: invalid {label} WDL {wdl}")

            if len(record["policy_indices"]) != len(record["policy_values"]):
                report.fail("Policy targets", f"game {gi}, position {pi}: policy index/value lengths differ")
            if len(set(record["policy_indices"])) != len(record["policy_indices"]):
                report.fail("Policy targets", f"game {gi}, position {pi}: duplicate legal policy indices")

            legal_moves = list(board.legal_moves)
            expected_policy_indices = {
                move_to_index[canonical_move(cozy_policy_move(board, move), board.turn)]
                for move in legal_moves
            }
            stored_policy_indices = set(record["policy_indices"])
            if board.turn == chess.BLACK:
                python_rotated_policy_indices = {
                    move_to_index[
                        transform_move(cozy_policy_move(board, move), flip_180_square)
                    ]
                    for move in legal_moves
                }
                if python_rotated_policy_indices != stored_policy_indices:
                    python_policy_transform_mismatch_count += 1
            if stored_policy_indices != expected_policy_indices:
                missing = sorted(expected_policy_indices - stored_policy_indices)
                extra = sorted(stored_policy_indices - expected_policy_indices)
                report.fail(
                    "Policy targets",
                    f"game {gi}, position {pi}: legal policy set mismatch; "
                    f"stored={len(stored_policy_indices)}, legal={len(expected_policy_indices)}, "
                    f"missing={missing[:4]}, extra={extra[:4]}",
                )
            else:
                for legal_move in legal_moves:
                    encoded_move = canonical_move(
                        cozy_policy_move(board, legal_move), board.turn
                    )
                    encoded_index = move_to_index.get(encoded_move)
                    if encoded_index is None or decode_policy_move(
                        board, moves[encoded_index]
                    ) != legal_move:
                        report.fail(
                            "Board transformations",
                            f"game {gi}, position {pi}: legal move "
                            f"{legal_move.uci()} failed policy transform round-trip",
                        )
                for position, policy_index in enumerate(record["policy_indices"]):
                    if policy_index == int(fields["played_mv"]):
                        if record["policy_values"][position] <= 0.0:
                            report.fail(
                                "Policy targets",
                                f"game {gi}, position {pi}: recorded played move has zero policy mass",
                            )
                        break
                else:
                    report.fail(
                        "Policy targets",
                        f"game {gi}, position {pi}: played_mv index {int(fields['played_mv'])} "
                        "is absent from the position's policy entries",
                    )
            total_policy_positions += 1
            policy_entry_count += len(record["policy_indices"])

            stored_move_index = fields["played_mv"]
            if not math.isfinite(stored_move_index) or not stored_move_index.is_integer():
                report.fail("Policy targets", f"game {gi}, position {pi}: invalid played_mv index {stored_move_index}")
                game_bad = True
            elif not 0 <= int(stored_move_index) < len(moves):
                report.fail("Policy targets", f"game {gi}, position {pi}: played_mv index {int(stored_move_index)} out of range")
                game_bad = True
            else:
                actual_move = decode_policy_move(
                    board, moves[int(stored_move_index)]
                )
                if actual_move not in legal_moves:
                    report.fail(
                        "Policy targets",
                        f"game {gi}, position {pi}: played_mv index {int(stored_move_index)} "
                        f"decodes to illegal {actual_move.uci()} in {board.fen()}",
                    )
                    game_bad = True
                else:
                    played_move_count += 1
                    moves_for_game.append(actual_move)
                    board.push(actual_move)

        if game_bad:
            continue
        if len(moves_for_game) != game_length:
            report.fail(
                "Game sequencing",
                f"game {gi}: replayed {len(moves_for_game)} moves, expected {game_length}",
            )

        if game_records:
            final = game_records[-1]["fields"]
            if bool(final["is_terminal"]) and final["hit_move_limit"]:
                report.fail("Game sequencing", f"game {gi}: terminal game also recorded move-limit flag")

    # Check the absolute result summary in metadata against per-game final targets.
    if "root_wdl" in meta and len(meta["root_wdl"]) == 3:
        white_wins = sum(result is chess.WHITE for result in game_results)
        draws = sum(result is None for result in game_results)
        black_wins = sum(result is chess.BLACK for result in game_results)
        expected_root = (white_wins, draws, black_wins)
        for index, (actual, expected) in enumerate(zip(meta["root_wdl"], expected_root)):
            report.check("Value targets")
            if not close_enough(float(actual), float(expected), 1e-3):
                report.fail("Value targets", f"metadata root_wdl[{index}]={actual}, derived count={expected}")
    if game_results and len(game_results) != game_count:
        report.fail("Data integrity", f"derived {len(game_results)} game results, expected {game_count}")
    if lengths:
        if min(lengths) != meta.get("min_game_length"):
            report.fail("Data integrity", f"min_game_length metadata={meta.get('min_game_length')}, observed={min(lengths)}")
        if max(lengths) != meta.get("max_game_length"):
            report.fail("Data integrity", f"max_game_length metadata={meta.get('max_game_length')}, observed={max(lengths)}")
    if "hit_move_limit" in meta and game_count:
        actual_fraction = hit_move_limit_count / game_count
        if not close_enough(float(meta["hit_move_limit"]), actual_fraction, 1e-5):
            report.fail("Data integrity", f"hit_move_limit metadata={meta['hit_move_limit']}, observed fraction={actual_fraction}")

    print(f"Dataset: {stem}")
    print(f"SHA-256 (.bin): {bin_hash}")
    print(
        f"Metadata: {game_count} games, {position_count} positions, "
        f"{len(data)} data bytes, {len(offsets)} offsets"
    )
    print(
        f"Game lengths: min={min(lengths) if lengths else 'n/a'}, "
        f"max={max(lengths) if lengths else 'n/a'}, "
        f"positions with policies={total_policy_positions}"
    )
    print(
        "Board encoding comparison: "
        f"Python 180-degree mismatch={python_board_mismatch_count} positions; "
        f"(White={python_white_mismatch_count}, Black={python_black_mismatch_count}); "
        f"Rust rank-flip mismatch={rust_board_mismatch_count}; "
        f"Black-only Python-vs-Rust mismatch={cross_transform_mismatch_count}"
    )
    print(
        "Python mismatch planes: piece planes="
        f"{python_piece_plane_mismatch_count} positions, en-passant plane="
        f"{python_ep_plane_mismatch_count} positions"
    )
    print(
        "Black policy rows differing under Python's 180-degree board transform "
        f"vs rank-flip policy transform={python_policy_transform_mismatch_count}"
    )
    print(
        f"Policy entries={policy_entry_count}; played moves decoded and replayed={played_move_count}; "
        f"expected NaN scalar cells={sum(allowed_nan_counts.values())}"
    )
    if custom_start_game_count:
        print(
            f"Non-standard starts seeded from serialized root positions="
            f"{custom_start_game_count}; initial history/repetition counts are unavailable"
        )
    print(
        "Results derived: "
        f"White wins={result_counts[chess.WHITE]}, draws/unfinished={result_counts[None]}, "
        f"Black wins={result_counts[chess.BLACK]}, move-limit games={hit_move_limit_count}"
    )
    report.summary()
    return 1 if any(report.failures.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="dataset stem or .bin/.off/.json path")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="maximum representative violations per category (default: 5)",
    )
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")
    return audit(args.dataset, args.max_examples)


if __name__ == "__main__":
    sys.exit(main())
