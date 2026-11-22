use bytemuck::cast_slice;
use cozy_chess::{Color, GameStatus};
use serde::Serialize;
use std::{
    cmp::{max, min},
    fs::File,
    io::{self, BufWriter, Seek, Write},
    path::{Path, PathBuf},
};

use crate::{
    boardmanager::BoardStack,
    dataformat::{Position, Simulation, ZeroValuesPov},
    decoder::{board_data, extract_policy, policy_index},
    mcts_trainer::Wdl,
    mvs::get_contents,
};

#[derive(Serialize)]
struct MetaData<'a> {
    game: &'a str,

    input_bool_shape: &'a [usize],
    input_scalar_count: usize,
    policy_shape: &'a [usize],

    game_count: usize,
    position_count: usize,
    includes_terminal_positions: bool,
    includes_game_start_indices: bool,

    max_game_length: i32,
    min_game_length: i32,
    root_wdl: [f32; 3],
    hit_move_limit: f32,

    scalar_names: &'static [&'static str],
}

#[derive(Debug)]
pub struct BinaryOutput {
    game: String,
    path: PathBuf,

    bin_write: BufWriter<File>,
    off_write: BufWriter<File>,
    json_tmp_write: BufWriter<File>,

    game_count: usize,
    position_count: usize,

    max_game_length: Option<i32>,
    min_game_length: Option<i32>,

    total_root_wdl: [u64; 3],
    hit_move_limit_count: u64,

    next_offset: u64,
    game_start_indices: Vec<u64>,

    finished: bool,
}

#[derive(Debug)]
struct Scalars {
    game_id: usize,
    pos_index: usize,
    game_length: usize,
    zero_visits: u64,
    is_full_search: bool,
    is_final_position: bool,
    is_terminal: bool,
    hit_move_limit: bool,
    available_mv_count: usize,
    played_mv: isize,
    kdl_policy: f32,
    final_values: ZeroValuesPov, // z
    zero_values: ZeroValuesPov,  // q
    net_values: ZeroValuesPov,   // v
}

impl BinaryOutput {
    pub fn new(path: impl AsRef<Path>, game: &str) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        assert!(
            path.extension().is_none(),
            "Binary output path should not have an extension, .bin and .json are added automatically"
        );

        let bin_write = BufWriter::new(File::create(path.with_extension("bin"))?);
        let off_write = BufWriter::new(File::create(path.with_extension("off"))?);
        let json_tmp_write = BufWriter::new(File::create(path.with_extension("json.tmp"))?);

        Ok(BinaryOutput {
            game: game.to_string(),

            bin_write,
            off_write,
            json_tmp_write,

            path,

            game_count: 0,
            position_count: 0,

            max_game_length: None,
            min_game_length: None,

            total_root_wdl: [0, 0, 0],
            hit_move_limit_count: 0,

            next_offset: 0,
            game_start_indices: vec![],

            finished: false,
        })
    }

    pub fn append(&mut self, simulation: &Simulation) -> io::Result<()> {
        let Simulation {
            positions,
            final_board,
        } = simulation;

        // collect metadata statistics
        let game_id = self.game_count;
        let game_length = positions.len();

        self.game_start_indices.push(self.position_count as u64);

        self.game_count += 1;
        self.position_count += 1 + game_length;

        self.max_game_length = Some(max(game_length as i32, self.max_game_length.unwrap_or(-1)));
        self.min_game_length = Some(min(
            game_length as i32,
            self.min_game_length.unwrap_or(i32::MAX),
        ));

        let is_terminal = final_board.is_terminal();
        let hit_move_limit = !is_terminal;
        if hit_move_limit {
            self.hit_move_limit_count += 1;
        }
        let winner: Option<Color> = match final_board.status() {
            GameStatus::Drawn => None,
            GameStatus::Won => Some(!final_board.board().side_to_move()),
            GameStatus::Ongoing => None,
        }; // colour (if any) that won the game

        // KZero: `outcome.pov(start_board.next_player())`, i.e. from the point of view of the
        // player to move in the first position, not White.
        let root_pov = simulation.start_board().board().side_to_move();
        match winner {
            Some(player) if player == root_pov => self.total_root_wdl[0] += 1,
            Some(_) => self.total_root_wdl[2] += 1,
            None => self.total_root_wdl[1] += 1,
        }

        // write the positions
        for (pos_index, position) in positions.iter().enumerate() {
            let &Position {
                ref board,
                is_full_search,
                played_mv,
                zero_visits,
                ref zero_evaluation,
                ref net_evaluation,
            } = position;

            let (available_mv_count, policy_indices) = collect_policy_indices(board);
            assert_eq!(available_mv_count, zero_evaluation.policy.len());
            assert_eq!(available_mv_count, net_evaluation.policy.len());
            let played_mv_index = policy_index(board, played_mv, get_contents());
            let kdl_policy = kdl_divergence(&zero_evaluation.policy, &net_evaluation.policy);
            // KZero convention (kz-selfplay binary_output.rs): non-final positions store
            // `game_length + 1 - pos_index` and the appended final position stores 0. The inherited
            // sampler relies on this: the final position is at `pi + final_moves_left - 1`.
            let moves_left = (game_length + 1 - pos_index) as f32;
            let stored_policy = &zero_evaluation.policy;
            let final_values = final_values_pov(winner, board.board().side_to_move(), moves_left);

            let scalars = Scalars {
                game_id,
                pos_index,
                game_length,
                zero_visits,
                is_full_search,
                is_final_position: false,
                is_terminal: false,
                hit_move_limit: false,
                available_mv_count: stored_policy.len(),
                played_mv: played_mv_index as isize,
                kdl_policy,
                final_values,
                zero_values: zero_evaluation.values,
                net_values: net_evaluation.values,
            };

            self.append_position(board, &scalars, &policy_indices, stored_policy)?;
        }

        let final_values = final_values_pov(winner, final_board.board().side_to_move(), 0.0);

        let scalars = Scalars {
            game_id,
            pos_index: game_length,
            game_length,
            zero_visits: 0,
            is_full_search: false,
            is_final_position: true,
            is_terminal,
            hit_move_limit,
            available_mv_count: 0,
            played_mv: -1,
            kdl_policy: f32::NAN,
            final_values,
            zero_values: ZeroValuesPov::nan(),
            //TODO in theory we could ask the network, but this is only really meaningful for muzero
            net_values: ZeroValuesPov::nan(),
        };

        self.append_position(final_board, &scalars, &[], &[])?;

        Ok(())
    }

    fn append_position(
        &mut self,
        board: &BoardStack,
        scalars: &Scalars,
        policy_indices: &[u32],
        policy_values: &[f32],
    ) -> io::Result<()> {
        // encode board

        let (board_scalars, board_bools) = board_data(board);

        // converting bools into u8s (8 bools = 1 u8)

        assert_eq!(board_bools.len() % 8, 0);

        let mut byte_vec: Vec<u8> = Vec::new();

        for i in 0..board_bools.len() / 8 {
            let mut byte: u8 = 0;
            for j in 0..8 {
                let b = board_bools[i * 8 + j];
                byte |= (b as u8) << j;
                // byte[j] = b
            }
            byte_vec.push(byte);
        }

        // check that everything makes sense
        let policy_len = policy_indices.len();
        assert_eq!(policy_len, policy_values.len());
        assert_normalized_or_nan(
            scalars.zero_values.wdl.w + scalars.zero_values.wdl.d + scalars.zero_values.wdl.l,
        );
        assert_normalized_or_nan(
            scalars.net_values.wdl.w + scalars.net_values.wdl.d + scalars.net_values.wdl.l,
        );
        assert_normalized_or_nan(
            scalars.final_values.wdl.w + scalars.final_values.wdl.d + scalars.final_values.wdl.l,
        );
        if policy_len != 0 {
            assert_normalized_or_nan(policy_values.iter().sum());
        }

        // save current offset
        // we keep track of the offset ourselves because seeking/stream_position flushes the buffer and is slow
        debug_assert_eq!(self.next_offset, self.bin_write.stream_position()?);
        self.off_write.write_all(&self.next_offset.to_le_bytes())?;

        // actually write stuff to the bin file
        let scalars = scalars.to_vec();
        let data_to_write: &[&[u8]] = &[
            cast_slice(&scalars),
            &byte_vec,
            cast_slice(&board_scalars),
            cast_slice(policy_indices),
            cast_slice(policy_values),
        ];
        for &data in data_to_write {
            self.bin_write.write_all(data)?;
            self.next_offset += data.len() as u64;
        }

        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if self.finished {
            panic!("This output is already finished")
        }
        self.finished = true;

        let meta = MetaData {
            game: &self.game,
            scalar_names: Scalars::NAMES,
            input_bool_shape: &[13, 8, 8], // pieces + EP
            input_scalar_count: 8,         // move turn, counters, castling
            policy_shape: &[1880_usize],
            game_count: self.game_count,
            position_count: self.position_count,
            includes_terminal_positions: true,
            includes_game_start_indices: true,
            max_game_length: self.max_game_length.unwrap_or(-1),
            min_game_length: self.min_game_length.unwrap_or(-1),
            // KZero stores the mean root WDL over the games in this file, not raw counts.
            root_wdl: self
                .total_root_wdl
                .map(|count| count as f32 / self.game_count as f32),
            hit_move_limit: self.hit_move_limit_count as f32 / self.game_count as f32,
        };

        serde_json::to_writer_pretty(&mut self.json_tmp_write, &meta)?;
        self.off_write
            .write_all(cast_slice(&self.game_start_indices))?;

        self.json_tmp_write.flush()?;
        self.bin_write.flush()?;
        self.off_write.flush()?;

        let path_json_tmp = self.path.with_extension("json.tmp");
        let path_json = self.path.with_extension("json");
        std::fs::rename(path_json_tmp, path_json)?;

        Ok(())
    }

    pub fn game_count(&self) -> usize {
        self.game_count
    }
    pub fn position_count(&self) -> usize {
        self.position_count
    }
}

fn final_values_pov(
    winner: Option<Color>,
    player_to_move: Color,
    moves_left: f32,
) -> ZeroValuesPov {
    match winner {
        Some(winner) if winner == player_to_move => ZeroValuesPov {
            value: 1.0,
            wdl: Wdl {
                w: 1.0,
                d: 0.0,
                l: 0.0,
            },
            moves_left,
        },
        Some(_) => ZeroValuesPov {
            value: -1.0,
            wdl: Wdl {
                w: 0.0,
                d: 0.0,
                l: 1.0,
            },
            moves_left,
        },
        None => ZeroValuesPov {
            value: 0.0,
            wdl: Wdl {
                w: 0.0,
                d: 1.0,
                l: 0.0,
            },
            moves_left,
        },
    }
}

fn collect_policy_indices(board: &BoardStack) -> (usize, Vec<u32>) {
    match board.status() {
        GameStatus::Ongoing => {
            let (_, indices) = extract_policy(board, get_contents());
            let policy_indices: Vec<u32> = indices.into_iter().map(|i| i as u32).collect();
            (policy_indices.len(), policy_indices)
        }
        GameStatus::Drawn | GameStatus::Won => (0, vec![]),
    }
}

/// Port of KZero's `kz_util::math::kdl_divergence`, including its behaviour for zero entries.
fn kdl_divergence(p: &[f32], q: &[f32]) -> f32 {
    assert_eq!(p.len(), q.len());
    p.iter().zip(q).map(|(&p, &q)| p * (p / q).ln()).sum()
}

fn assert_normalized_or_nan(x: f32) {
    assert!(x.is_nan() || (1.0 - x).abs() < 0.001);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cozy_chess::Board;

    #[test]
    fn black_board_data_flips_rank_only() {
        let board = Board::from_fen("8/k7/8/8/8/8/8/7K b - - 0 1", false).unwrap();
        let (_, bools) = board_data(&BoardStack::new(board));

        let rank_only_reflection_index = 8; // a7 -> a2
        let rotated_index = 15; // a7 -> h2

        assert!((0..12).any(|plane| bools[plane * 64 + rank_only_reflection_index]));
        assert!(!(0..12).any(|plane| bools[plane * 64 + rotated_index]));
    }

    #[test]
    fn final_values_are_player_relative_with_no_moves_remaining() {
        let values = final_values_pov(Some(Color::White), Color::Black, 0.0);

        assert_eq!(values.value, -1.0);
        assert_eq!(
            values.wdl,
            Wdl {
                w: 0.0,
                d: 0.0,
                l: 1.0
            }
        );
        assert_eq!(values.moves_left, 0.0);
    }

    #[test]
    fn truncated_simulation_counts_as_a_move_limit_result() {
        let path =
            std::env::temp_dir().join(format!("tzrust-fileformat-test-{}", std::process::id()));
        let mut output = BinaryOutput::new(&path, "chess").unwrap();
        let simulation = Simulation {
            positions: vec![],
            final_board: BoardStack::new(Board::default()),
        };

        output.append(&simulation).unwrap();
        assert_eq!(output.hit_move_limit_count, 1);

        for extension in ["bin", "off", "json", "json.tmp"] {
            let _ = std::fs::remove_file(path.with_extension(extension));
        }
    }

    // ---- interoperability with files written by KZero itself ----
    //
    // `tests/fixtures/kzero/kzero_chess.*` was written by kZero's own `BinaryOutput` with
    // `ChessStdMapper` (kZero commit 5bf3343, see `write_fixture.rs.txt`) from the games in
    // `fixture_games.txt`. Here TrueZero writes the same games with the same synthetic search
    // results, and every record must match.

    use crate::dataformat::{ZeroEvaluationPov, ZeroValuesPov};
    use crate::mvs::get_contents;
    use cozy_chess::{Move, Piece, Square};
    use std::path::PathBuf;

    const SCALAR_COUNT: usize = 26;
    const KDL_SCALAR: usize = 10;
    const BOOL_BYTES: usize = 13 * 64 / 8;
    const INPUT_SCALARS: usize = 8;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/kzero")
    }

    /// UCI castling (e1g1) to cozy-chess castling (king takes rook, e1h1)
    fn parse_uci(bs: &BoardStack, s: &str) -> Move {
        let mut mv: Move = s.parse().unwrap();
        if bs.board().piece_on(mv.from) == Some(Piece::King) {
            let df = mv.to.file() as i32 - mv.from.file() as i32;
            if df.abs() == 2 {
                let rook_file = if df > 0 {
                    cozy_chess::File::H
                } else {
                    cozy_chess::File::A
                };
                mv.to = Square::new(rook_file, mv.from.rank());
            }
        }
        mv
    }

    fn write_fixture_games(path: &Path) {
        let games = std::fs::read_to_string(fixture_dir().join("fixture_games.txt")).unwrap();
        let mut output = BinaryOutput::new(path, "chess").unwrap();
        for line in games.lines().filter(|l| !l.trim().is_empty()) {
            let (fen, moves) = line.split_once('|').unwrap();
            let board = match fen.trim() {
                "startpos" => Board::default(),
                fen => Board::from_fen(fen, false).unwrap(),
            };
            let mut bs = BoardStack::new(board);
            let mut positions = vec![];
            for (i, mv_str) in moves.split_whitespace().enumerate() {
                let mv = parse_uci(&bs, mv_str);
                // depends only on the absolute from-square, like the KZero fixture generator
                let (moves, _) = extract_policy(&bs, get_contents());
                let weights: Vec<f32> = moves
                    .iter()
                    .map(|m| 1.0 + (m.from as usize % 7) as f32)
                    .collect();
                let total: f32 = weights.iter().sum();
                let n = weights.len();
                positions.push(Position {
                    board: bs.clone(),
                    is_full_search: i % 2 == 0,
                    played_mv: mv,
                    zero_visits: 100 + i as u64,
                    zero_evaluation: ZeroEvaluationPov {
                        values: ZeroValuesPov {
                            value: 0.25,
                            wdl: Wdl {
                                w: 0.5,
                                d: 0.25,
                                l: 0.25,
                            },
                            moves_left: i as f32,
                        },
                        policy: weights.iter().map(|w| w / total).collect(),
                    },
                    net_evaluation: ZeroEvaluationPov {
                        values: ZeroValuesPov {
                            value: -0.1,
                            wdl: Wdl {
                                w: 0.2,
                                d: 0.3,
                                l: 0.5,
                            },
                            moves_left: 7.0,
                        },
                        policy: vec![1.0 / n as f32; n],
                    },
                });
                bs.play(mv);
            }
            output
                .append(&Simulation {
                    positions,
                    final_board: bs,
                })
                .unwrap();
        }
        output.finish().unwrap();
    }

    /// One position as stored on disk; `policy` is sorted by policy index.
    pub(crate) struct Record {
        pub(crate) scalars: Vec<f32>,
        pub(crate) bools: Vec<u8>,
        pub(crate) input_scalars: Vec<f32>,
        pub(crate) policy: Vec<(u32, f32)>,
    }

    pub(crate) fn read_records(base: &Path) -> (Vec<Record>, Vec<u8>, serde_json::Value) {
        let bin = std::fs::read(base.with_extension("bin")).unwrap();
        let off = std::fs::read(base.with_extension("off")).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(base.with_extension("json")).unwrap()).unwrap();
        let position_count = json["position_count"].as_u64().unwrap() as usize;
        let f32_at = |o: usize| f32::from_le_bytes(bin[o..o + 4].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(bin[o..o + 4].try_into().unwrap());

        let mut records = vec![];
        for pi in 0..position_count {
            let start = u64::from_le_bytes(off[pi * 8..pi * 8 + 8].try_into().unwrap()) as usize;
            let scalars: Vec<f32> = (0..SCALAR_COUNT).map(|i| f32_at(start + 4 * i)).collect();
            let mut o = start + 4 * SCALAR_COUNT;
            let bools = bin[o..o + BOOL_BYTES].to_vec();
            o += BOOL_BYTES;
            let input_scalars: Vec<f32> = (0..INPUT_SCALARS).map(|i| f32_at(o + 4 * i)).collect();
            o += 4 * INPUT_SCALARS;
            let n = scalars[8] as usize;
            let mut policy: Vec<(u32, f32)> = (0..n)
                .map(|i| (u32_at(o + 4 * i), f32_at(o + 4 * (n + i))))
                .collect();
            policy.sort_by_key(|&(i, _)| i);
            records.push(Record {
                scalars,
                bools,
                input_scalars,
                policy,
            });
        }
        (records, off, json)
    }

    fn same(a: f32, b: f32, tol: f32) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol
    }

    #[test]
    fn writes_the_same_records_as_kzero() {
        let path =
            std::env::temp_dir().join(format!("tzrust-kzero-interop-{}", std::process::id()));
        write_fixture_games(&path);

        let (expected, expected_off, expected_json) =
            read_records(&fixture_dir().join("kzero_chess"));
        let (actual, actual_off, actual_json) = read_records(&path);

        assert_eq!(actual_off, expected_off, ".off files differ");
        assert_eq!(actual_json, expected_json, ".json metadata differs");
        assert_eq!(actual.len(), expected.len());

        for (pi, (a, e)) in actual.iter().zip(&expected).enumerate() {
            for i in 0..SCALAR_COUNT {
                // kdl_policy sums floats in move-generation order, which differs per crate
                let tol = if i == KDL_SCALAR { 1e-5 } else { 0.0 };
                assert!(
                    same(a.scalars[i], e.scalars[i], tol),
                    "position {pi}: scalar {i} is {} but KZero wrote {}",
                    a.scalars[i],
                    e.scalars[i]
                );
            }
            assert_eq!(a.bools, e.bools, "position {pi}: input bools differ");
            assert_eq!(
                a.input_scalars, e.input_scalars,
                "position {pi}: input scalars differ"
            );
            assert_eq!(
                a.policy.len(),
                e.policy.len(),
                "position {pi}: policy length"
            );
            for (&(ai, av), &(ei, ev)) in a.policy.iter().zip(&e.policy) {
                assert_eq!(ai, ei, "position {pi}: policy indices differ");
                assert!(
                    same(av, ev, 1e-6),
                    "position {pi}: policy value {av} vs {ev}"
                );
            }
        }

        for extension in ["bin", "off", "json"] {
            let _ = std::fs::remove_file(path.with_extension(extension));
        }
    }
}

impl Scalars {
    const NAMES: &'static [&'static str] = &[
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
    ];

    fn to_vec(&self) -> Vec<f32> {
        let mut result = vec![
            self.game_id as f32,
            self.pos_index as f32,
            self.game_length as f32,
            self.zero_visits as f32,
            self.is_full_search as u8 as f32,
            self.is_final_position as u8 as f32,
            self.is_terminal as u8 as f32,
            self.hit_move_limit as u8 as f32,
            self.available_mv_count as f32,
            self.played_mv as f32,
            self.kdl_policy,
        ];

        result.extend_from_slice(&self.final_values.to_slice());
        result.extend_from_slice(&self.zero_values.to_slice());
        result.extend_from_slice(&self.net_values.to_slice());

        assert_eq!(result.len(), Self::NAMES.len());
        result
    }
}
