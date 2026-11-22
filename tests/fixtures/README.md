# Replay-file fixtures

Both fixtures hold the same seven chess games (`kzero/fixture_games.txt`), with the same
synthetic search results: the zero policy weights each legal move by `1 + from_square % 7`,
the net policy is uniform, and the zero/net values are constants.

- `kzero/kzero_chess.{bin,off,json}`: written by kZero's own `BinaryOutput` with
  `ChessStdMapper` (github.com/KarelPeeters/kZero, commit 5bf3343, board-game 0.8.2,
  chess 3.2.0). Generator: `kzero/write_fixture.rs.txt`, placed in
  `rust/kz-selfplay/examples/write_fixture.rs` and run as
  `cargo run --release -p kz-selfplay --example write_fixture -- <out> fixture_games.txt`.
- `truezero_legacy/legacy_chess.{bin,off,json}`: the same games written by TrueZero's writer
  before KZero compatibility was restored (commit 1c83ad6), using the helper now in
  `src/rust/fileformat.rs` tests. Used to test `src/python/kzero_data.py` migration.

The games cover a decisive game from each side's first move, en passant with and without an
adjacent capturing pawn, castling on both sides, one-sided castling rights, under- and
queen promotion, repetitions and games cut off before the end.
