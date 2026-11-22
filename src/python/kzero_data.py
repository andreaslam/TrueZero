"""TrueZero's entry point to KZero replay data.

TrueZero writes and reads the KZero data format (`kz-selfplay/src/binary_output.rs` and
`lib/data`) unchanged. This module only adds what TrueZero needs on top of it:

- `open_data_file` opens a chess data file with the inherited `DataFile` and refuses files in
  TrueZero's pre-KZero-compatibility layout ("legacy", see below) with a clear error instead of
  letting them reach the sampler.
- `migrate_legacy_file` converts a legacy file into the KZero layout. It is explicit and
  writes a new file: nothing is converted implicitly at load time.

Legacy files were written by TrueZero before the fixes that restored KZero compatibility.
They differ from KZero's `ChessStdMapper` data in five ways, all reversible from the stored
bytes except `kdl_policy`:

1. `final_moves_left` was `game_length - pos_index` instead of `game_length + 1 - pos_index`.
2. Castling input scalars were ordered (queenside, kingside) instead of (kingside, queenside).
3. The en passant plane was set after every double push, not only when a pawn of the side to
   move stands next to the pushed pawn.
4. Castling moves used the e1h1 / e1a1 policy slots (cozy-chess "king takes rook") instead of
   KZero's e1g1 / e1c1.
5. `root_wdl` held absolute (White) game counts instead of the mean from the point of view of
   the first player to move.

`kdl_policy` was always NaN; the network policy it needs is not stored, so it stays NaN.
"""

import json
import os
from pathlib import Path

import numpy as np

from lib.data.file import DataFile
from lib.games import Game
from lib.mapping.mapping import CHESS_FLAT_TO_MOVE_INPUT

GAME_NAME = "chess"

SCALAR_NAMES = (
    "game_id pos_index game_length zero_visits is_full_search is_final_position is_terminal "
    "hit_move_limit available_mv_count played_mv kdl_policy "
    "final_v final_wdl_w final_wdl_d final_wdl_l final_moves_left "
    "zero_v zero_wdl_w zero_wdl_d zero_wdl_l zero_moves_left "
    "net_v net_wdl_w net_wdl_d net_wdl_l net_moves_left"
).split()
S = {name: i for i, name in enumerate(SCALAR_NAMES)}

BOOL_BYTES = 13 * 64 // 8
INPUT_SCALARS = 8
OWN_PAWN_PLANE = 0
OWN_KING_PLANE = 5
EN_PASSANT_PLANE = 12

E1, A1, C1, G1, H1 = 4, 0, 2, 6, 7


def _flat_index(from_sq: int, to_sq: int) -> int:
    matches = np.nonzero(
        (CHESS_FLAT_TO_MOVE_INPUT[:, 0] == from_sq)
        & (CHESS_FLAT_TO_MOVE_INPUT[:, 1] == to_sq)
        & (CHESS_FLAT_TO_MOVE_INPUT[:, 7] == 1)  # not a promotion
    )[0]
    assert len(matches) == 1, (from_sq, to_sq, matches)
    return int(matches[0])


# legacy castling slot -> KZero castling slot, in the side to move's point of view
LEGACY_CASTLING_INDEX = {
    _flat_index(E1, H1): _flat_index(E1, G1),
    _flat_index(E1, A1): _flat_index(E1, C1),
}


class LegacyDataError(ValueError):
    """The file uses TrueZero's pre-KZero-compatibility layout."""


def open_data_file(path: str) -> DataFile:
    """Open a chess data file, refusing legacy TrueZero files."""
    file = DataFile.open(Game.find(GAME_NAME), path)
    if detect_layout(file) == "legacy":
        file.close()
        raise LegacyDataError(
            f"{path} was written by TrueZero before KZero compatibility was restored. "
            f"Convert it with `python kzero_data.py migrate {path} <new path>`."
        )
    return file


def detect_layout(file: DataFile) -> str:
    """Return "kzero", "legacy" or "unknown" (no simulation has a move to tell them apart).

    KZero stores `final_moves_left = game_length + 1` for the first position of a game;
    legacy TrueZero stored `game_length`.
    """
    for sim in file.simulations:
        if sim.move_count == 0:
            continue
        first = file.positions[sim.start_file_pi]
        if first.final_moves_left == sim.move_count + 1:
            return "kzero"
        if first.final_moves_left == sim.move_count:
            return "legacy"
        raise ValueError(
            f"{file.info.bin_path}: unexpected final_moves_left {first.final_moves_left} "
            f"for a game of {sim.move_count} moves"
        )
    return "unknown"


def _bit(bools: bytearray, index: int) -> bool:
    return bool(bools[index // 8] >> (index % 8) & 1)


def _clear_bit(bools: bytearray, index: int):
    bools[index // 8] &= ~(1 << (index % 8)) & 0xFF


def _migrate_record(record: bytearray):
    scalars = np.frombuffer(record, dtype=np.float32, count=len(SCALAR_NAMES)).copy()
    pos = int(scalars[S["pos_index"]])
    length = int(scalars[S["game_length"]])
    is_final = bool(scalars[S["is_final_position"]])
    o = 4 * len(SCALAR_NAMES)
    bools = record[o : o + BOOL_BYTES]
    inputs = np.frombuffer(
        record, dtype=np.float32, count=INPUT_SCALARS, offset=o + BOOL_BYTES
    ).copy()
    n = int(scalars[S["available_mv_count"]])
    p = o + BOOL_BYTES + 4 * INPUT_SCALARS
    indices = np.frombuffer(record, dtype=np.int32, count=n, offset=p).copy()

    # 1. moves left
    expected = 0 if is_final else length - pos
    if scalars[S["final_moves_left"]] != expected:
        raise LegacyDataError(
            f"position {pos}/{length}: final_moves_left {scalars[S['final_moves_left']]} "
            f"does not follow the legacy layout (expected {expected})"
        )
    if not is_final:
        scalars[S["final_moves_left"]] = length + 1 - pos

    # 2. castling scalars: (queenside, kingside) -> (kingside, queenside), us then them
    inputs[[2, 3, 4, 5]] = inputs[[3, 2, 5, 4]]

    # 3. en passant only when a pawn of the side to move is next to the pushed pawn
    for sq in range(64):
        if _bit(bools, EN_PASSANT_PLANE * 64 + sq):
            file, rank = sq % 8, sq // 8
            neighbours = [rank * 8 + f for f in (file - 1, file + 1) if 0 <= f < 8]
            if not any(_bit(bools, OWN_PAWN_PLANE * 64 + s) for s in neighbours):
                _clear_bit(bools, EN_PASSANT_PLANE * 64 + sq)

    # 4. castling policy slots (only a king on e1 can castle, and e1h1/e1a1 is then castling)
    if _bit(bools, OWN_KING_PLANE * 64 + E1):
        assert not set(LEGACY_CASTLING_INDEX.values()) & set(indices.tolist())
        indices = np.array(
            [LEGACY_CASTLING_INDEX.get(int(i), int(i)) for i in indices], dtype=np.int32
        )
        played = int(scalars[S["played_mv"]])
        scalars[S["played_mv"]] = LEGACY_CASTLING_INDEX.get(played, played)

    record[0 : 4 * len(SCALAR_NAMES)] = scalars.tobytes()
    record[o : o + BOOL_BYTES] = bools
    record[o + BOOL_BYTES : p] = inputs.tobytes()
    record[p : p + 4 * n] = indices.tobytes()
    return scalars


def migrate_legacy_file(src: str, dst: str):
    """Write a KZero-layout copy of the legacy file `src` to `dst` (paths without extension)."""
    src, dst = Path(src), Path(dst)
    if any(dst.with_suffix(e).exists() for e in (".bin", ".off", ".json")):
        raise FileExistsError(dst)

    meta = json.loads(src.with_suffix(".json").read_text())
    assert meta["game"] == GAME_NAME and meta["scalar_names"] == SCALAR_NAMES
    assert meta["includes_terminal_positions"] and meta["includes_game_start_indices"]
    count = meta["position_count"]
    off = src.with_suffix(".off").read_bytes()
    offsets = np.frombuffer(off, dtype=np.uint64, count=count).astype(np.int64)
    starts = np.frombuffer(off, dtype=np.uint64, offset=8 * count).astype(np.int64)
    assert len(starts) == meta["game_count"]
    data = bytearray(src.with_suffix(".bin").read_bytes())
    bounds = list(offsets) + [len(data)]

    first_scalars = {}
    for pi in range(count):
        record = data[bounds[pi] : bounds[pi + 1]]
        scalars = _migrate_record(record)
        data[bounds[pi] : bounds[pi + 1]] = record
        first_scalars.setdefault(int(scalars[S["game_id"]]), scalars)

    # 5. root WDL: mean outcome from the first player to move's point of view
    wdl = np.mean(
        [
            s[[S["final_wdl_w"], S["final_wdl_d"], S["final_wdl_l"]]]
            for s in first_scalars.values()
        ],
        axis=0,
        dtype=np.float64,
    )
    # shortest repr of the f32, as serde_json writes it in KZero's files
    meta["root_wdl"] = [float(str(np.float32(x))) for x in wdl]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.with_suffix(".bin").write_bytes(bytes(data))
    dst.with_suffix(".off").write_bytes(off)
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 or sys.argv[1] != "migrate":
        sys.exit("usage: python kzero_data.py migrate <legacy path> <new path>")
    migrate_legacy_file(sys.argv[2], sys.argv[3])
    file = open_data_file(sys.argv[3])
    print(
        f"wrote {sys.argv[3]} ({len(file.positions)} positions, {detect_layout(file)} layout)"
    )
    file.close()
