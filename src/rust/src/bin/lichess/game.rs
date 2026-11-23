//! Playing one Lichess game.

use std::{num::NonZeroUsize, time::Duration};

use cozy_chess::{Board, Color, Move, Piece, Square};
use futures_util::StreamExt;
use litchee::{
    api::gameplay::{
        board::{LichessBoardEvent, LichessGameFull, LichessGameState},
        challenges::LichessChallengeColor,
        games::LichessGameStatusName,
    },
    model::LichessColor,
};
use lru::LruCache;
use tokio::time::{sleep, timeout};
use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::{Outcome, Position, Simulation, ZeroEvaluationAbs, ZeroEvaluationPov},
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest},
    selfplay::CollectorMessage,
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
    uci::{check_castling_move_on_output, time_to_nodes, UCIMsg},
};

use super::{runtime::Bot, BoxError};

const CACHE_SIZE: usize = 100_000;

const MOVES_TO_GO: u64 = 30;
const MAX_SEARCH_TIME_MS: u128 = 10_000;
const MIN_SEARCH_TIME_MS: u64 = 25;
const SHORT_TIME_RESERVE_MS: u64 = 35;
const LONG_TIME_RESERVE_FRACTION: f64 = 0.15;
const NO_TIME_LIMIT_NODES: u128 = 2_000;
const NO_TIME_LIMIT_SEARCH_TIME: Duration = Duration::from_secs(60);
/// Inference can be delayed while batches are flushed; give a search this much time on top of
/// its budget before falling back to a legal move.
const SEARCH_TIMEOUT_GRACE: Duration = Duration::from_secs(5);

/// Accept a draw offer, or offer a draw, when our clock is this low.
const DRAW_LOW_TIME_MS: i64 = 10_000;
/// Accept a draw offer, or offer a draw, when our evaluation (-1 to 1, from our point of view)
/// is at most this.
const DRAW_MAX_VALUE: f32 = 0.0;

const MOVE_ATTEMPTS: u32 = 3;
const STREAM_MAX_RECONNECTS: u32 = 10;
const STREAM_RECONNECT_DELAY: Duration = Duration::from_secs(1);

struct Game<'a> {
    bot: &'a Bot,
    id: &'a str,
    us: Color,
    initial_fen: String,
    opponent: Option<String>,
    rated: Option<bool>,
    /// (initial seconds, increment seconds); `None` for games without a clock
    clock: Option<(u32, u32)>,
    /// Our searched positions, recorded only when collecting training data.
    positions: Vec<Position>,
    cache: LruCache<CacheEntryKey, ZeroEvaluationAbs>,
    /// Number of plies in the game when we last moved, so repeated states are not searched twice.
    moved_at_ply: Option<usize>,
}

enum Search {
    Searched {
        mv: Move,
        /// Root value from our point of view.
        value: f32,
        position: Position,
    },
    /// No search was possible or it timed out.
    Fallback(Move),
}

pub async fn run_game(bot: &Bot, id: &str, color: LichessColor) -> Result<(), BoxError> {
    let mut game = Game {
        bot,
        id,
        us: to_color(color),
        initial_fen: "startpos".into(),
        opponent: None,
        rated: None,
        clock: None,
        positions: Vec::new(),
        cache: LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap()),
        moved_at_ply: None,
    };

    let mut failures = 0;
    loop {
        match bot.client.bot().stream_game(id).await {
            Ok(mut events) => {
                while let Some(event) = events.next().await {
                    let state = match event {
                        Ok(LichessBoardEvent::GameFull(full)) => {
                            game.set_info(&full);
                            full.state
                        }
                        Ok(LichessBoardEvent::GameState(state)) => state,
                        Ok(_) => continue,
                        Err(error) => {
                            eprintln!("[Lichess] Game {id} stream error: {error}");
                            break;
                        }
                    };
                    failures = 0;
                    if game.on_state(&state).await? {
                        return Ok(());
                    }
                }
            }
            Err(error) => eprintln!("[Lichess] Could not open game {id} stream: {error}"),
        }
        failures += 1;
        if failures > STREAM_MAX_RECONNECTS {
            return Err(format!("game stream failed {failures} times in a row").into());
        }
        sleep(STREAM_RECONNECT_DELAY).await;
    }
}

impl Game<'_> {
    fn set_info(&mut self, full: &LichessGameFull) {
        self.initial_fen = full
            .initial_fen
            .clone()
            .unwrap_or_else(|| "startpos".into());
        let opponent = match self.us {
            Color::White => full.black.as_ref(),
            Color::Black => full.white.as_ref(),
        };
        // the built-in Lichess AI has no account to challenge
        self.opponent = opponent
            .filter(|player| player.ai_level.is_none())
            .and_then(|player| player.name.clone().or_else(|| player.id.clone()));
        self.rated = full.rated;
        self.clock = full.clock.and_then(|clock| {
            let initial = clock.initial.unwrap_or(0) / 1000;
            let increment = clock.increment.unwrap_or(0).max(0) / 1000;
            (initial > 0).then_some((initial as u32, increment as u32))
        });
    }

    /// Reacts to a game state. Returns `true` once the game is over.
    async fn on_state(&mut self, state: &LichessGameState) -> Result<bool, BoxError> {
        if !matches!(
            state.status,
            LichessGameStatusName::Created | LichessGameStatusName::Started
        ) {
            self.finish(state).await;
            return Ok(true);
        }

        let bs = replay(&self.initial_fen, &state.moves)?;
        let ply = state.moves.split_whitespace().count();
        if bs.board().side_to_move() != self.us || self.moved_at_ply == Some(ply) {
            return Ok(false);
        }

        let our_time_ms = match self.us {
            Color::White => state.wtime,
            Color::Black => state.btime,
        };
        let low_on_time = !is_unlimited(state) && our_time_ms <= DRAW_LOW_TIME_MS;
        let draw_offered = match self.us {
            Color::White => state.bdraw,
            Color::Black => state.wdraw,
        }
        .unwrap_or(false);

        if draw_offered && low_on_time && self.accept_draw().await {
            return Ok(false);
        }

        let search = if bs.is_terminal() {
            // drawn by our rules (repetition, 50 moves), but Lichess waits for a claim
            if self.bot.client.bot().claim_draw(self.id).await.is_ok() {
                println!("[Lichess] Game {}: claimed a draw.", self.id);
                return Ok(false);
            }
            // search without the history that made the position terminal
            let bs = BoardStack::new(bs.board().clone());
            if bs.is_terminal() {
                match first_legal_move(bs.board()) {
                    Some(mv) => Search::Fallback(mv),
                    None => return Ok(false),
                }
            } else {
                self.search(&bs, state).await
            }
        } else {
            self.search(&bs, state).await
        };

        let (mv, value, mut position) = match search {
            Search::Searched {
                mv,
                value,
                position,
            } => {
                if draw_offered && value <= DRAW_MAX_VALUE && self.accept_draw().await {
                    return Ok(false);
                }
                (mv, value, Some(position))
            }
            Search::Fallback(mv) => (mv, f32::NAN, None),
        };

        let uci = check_castling_move_on_output(&bs, mv).to_string();
        let offer_draw = value <= DRAW_MAX_VALUE || low_on_time;
        for attempt in 1..=MOVE_ATTEMPTS {
            match self
                .bot
                .client
                .bot()
                .make_move(self.id, &uci, offer_draw)
                .await
            {
                Ok(()) => {
                    println!(
                        "[Lichess] Game {}: played {uci} (value {value:.3}{}).",
                        self.id,
                        if offer_draw { ", offering a draw" } else { "" }
                    );
                    self.moved_at_ply = Some(ply);
                    if let Some(position) = position.take().filter(|_| self.bot.collector.is_some()) {
                        self.positions.push(position);
                    }
                    return Ok(false);
                }
                Err(error) => eprintln!(
                    "[Lichess] Game {}: move {uci} failed (attempt {attempt}/{MOVE_ATTEMPTS}): {error}",
                    self.id
                ),
            }
        }
        Ok(false)
    }

    async fn accept_draw(&self) -> bool {
        match self.bot.client.bot().handle_draw(self.id, true).await {
            Ok(()) => {
                println!("[Lichess] Game {}: accepted a draw offer.", self.id);
                true
            }
            Err(error) => {
                eprintln!(
                    "[Lichess] Game {}: could not accept a draw: {error}",
                    self.id
                );
                false
            }
        }
    }

    async fn search(&mut self, bs: &BoardStack, state: &LichessGameState) -> Search {
        let (nodes, time) = search_budget(state, self.us, self.bot.active_games());
        let settings = SearchSettings {
            fpu: FPUSettings {
                root_fpu: 0.5,
                children_fpu: 0.5,
            },
            wdl: EvalMode::Wdl,
            moves_left: Some(MovesLeftSettings {
                moves_left_weight: 0.03,
                moves_left_clip: 20.0,
                moves_left_sharpness: 0.5,
            }),
            c_puct: CPUCTSettings {
                root_c_puct: 3.0,
                children_c_puct: 2.0,
            },
            max_nodes: Some(nodes),
            alpha: 0.03,
            eps: 0.25,
            // root noise and policy temperature only when the game is training data
            search_type: if self.bot.collector.is_some() {
                TypeRequest::TrainerSearch(None)
            } else {
                TypeRequest::NonTrainerSearch
            },
            pst: PSTSettings {
                root_pst: 1.75,
                children_pst: 1.5,
            },
            batch_size: 1,
        };

        let (stop_send, stop_recv) = flume::bounded(1);
        let stopper = tokio::spawn(async move {
            sleep(time).await;
            let _ = stop_send.send(UCIMsg::UCIStopMessage);
        });
        let result = timeout(
            time + SEARCH_TIMEOUT_GRACE,
            get_move(
                bs.clone(),
                self.bot.tensor_send.clone(),
                settings,
                Some(stop_recv),
                &mut self.cache,
            ),
        )
        .await;
        stopper.abort();

        let Ok((mv, net_evaluation, _, search_evaluation, visits)) = result else {
            eprintln!(
                "[Lichess] Game {}: search did not finish in {time:?} (+{SEARCH_TIMEOUT_GRACE:?}); \
                 playing a legal move instead.",
                self.id
            );
            return Search::Fallback(first_legal_move(bs.board()).expect("our turn, not terminal"));
        };

        let side = bs.board().side_to_move();
        let values = search_evaluation.values.to_relative(side);
        Search::Searched {
            mv,
            value: values.value,
            position: Position {
                board: bs.clone(),
                is_full_search: true,
                played_mv: mv,
                zero_visits: visits as u64,
                zero_evaluation: ZeroEvaluationPov {
                    values,
                    policy: search_evaluation.policy,
                },
                net_evaluation: ZeroEvaluationPov {
                    values: net_evaluation.values.to_relative(side),
                    policy: net_evaluation.policy,
                },
            },
        }
    }

    async fn finish(&mut self, state: &LichessGameState) {
        // games that never really started, or whose result is unknown, are not real results
        let outcome = match state.status {
            LichessGameStatusName::Aborted
            | LichessGameStatusName::NoStart
            | LichessGameStatusName::UnknownFinish
            | LichessGameStatusName::Cheat => None,
            _ => Some(match state.winner {
                Some(winner) => Outcome::Win(to_color(winner)),
                None => Outcome::Draw,
            }),
        };
        println!(
            "[Lichess] Game {} over: {:?}, {outcome:?}.",
            self.id, state.status
        );
        let Some(outcome) = outcome else {
            return;
        };

        if let Some(collector) = &self.bot.collector {
            if !self.positions.is_empty() {
                match replay(&self.initial_fen, &state.moves) {
                    Ok(final_board) => {
                        let simulation = Simulation {
                            positions: std::mem::take(&mut self.positions),
                            final_board,
                            outcome: Some(outcome),
                        };
                        if collector
                            .send_async(CollectorMessage::FinishedGame(simulation))
                            .await
                            .is_err()
                        {
                            eprintln!("[Lichess] Game {}: the collector stopped.", self.id);
                        }
                    }
                    Err(error) => eprintln!(
                        "[Lichess] Game {}: not recorded, final position invalid: {error}",
                        self.id
                    ),
                }
            }
        }

        self.offer_rematch().await;
    }

    async fn offer_rematch(&self) {
        let Some(opponent) = &self.opponent else {
            return;
        };
        let color = match self.us {
            Color::White => LichessChallengeColor::Black,
            Color::Black => LichessChallengeColor::White,
        };
        let challenges = self.bot.client.challenges();
        let mut request = challenges.challenge(opponent).color(color);
        if let Some(rated) = self.rated {
            request = request.rated(rated);
        }
        if let Some((initial, increment)) = self.clock {
            request = request.clock(initial, increment);
        }
        match request.send().await {
            Ok(_) => println!("[Lichess] Offered {opponent} a rematch."),
            Err(error) => eprintln!("[Lichess] Could not offer {opponent} a rematch: {error}"),
        }
    }
}

fn to_color(color: LichessColor) -> Color {
    match color {
        LichessColor::White => Color::White,
        LichessColor::Black => Color::Black,
    }
}

/// Correspondence and unlimited games report no (or a huge) clock.
fn is_unlimited(state: &LichessGameState) -> bool {
    let no_clock = state.wtime <= 0 && state.btime <= 0 && state.winc <= 0 && state.binc <= 0;
    no_clock || state.wtime >= 86_400_000 || state.btime >= 86_400_000
}

/// Node and time budget for one move.
///
/// The clock is per game, so the time is not shared between games, but the executors are, so
/// the node budget is.
fn search_budget(state: &LichessGameState, us: Color, active_games: usize) -> (u128, Duration) {
    if is_unlimited(state) {
        return (NO_TIME_LIMIT_NODES, NO_TIME_LIMIT_SEARCH_TIME);
    }
    let ms = |value: i64| Some(value.max(0) as u64);
    let (time, nodes) = time_to_nodes(
        us,
        [ms(state.wtime), ms(state.btime)],
        [ms(state.winc), ms(state.binc)],
        MOVES_TO_GO,
        Some(MAX_SEARCH_TIME_MS),
    );
    let time = time.unwrap_or(0) as u64;
    let reserve = if time < 1_000 {
        SHORT_TIME_RESERVE_MS.min(time / 3)
    } else {
        (time as f64 * LONG_TIME_RESERVE_FRACTION) as u64
    };
    let time = time.saturating_sub(reserve).max(MIN_SEARCH_TIME_MS);
    let nodes = (nodes / active_games.max(1) as u128).max(1);
    (nodes, Duration::from_millis(time))
}

/// Replays a Lichess game into a `BoardStack`, keeping the history needed to detect repetitions.
///
/// Lichess keeps playing after positions our rules call drawn until a player claims the draw.
/// The history is dropped from such a position on, because `BoardStack` will not play past it.
fn replay(initial_fen: &str, moves: &str) -> Result<BoardStack, BoxError> {
    let board = if initial_fen == "startpos" {
        Board::default()
    } else {
        Board::from_fen(initial_fen, false)
            .map_err(|error| format!("invalid initial FEN {initial_fen:?}: {error:?}"))?
    };
    let mut bs = BoardStack::new(board);
    let mut without_history: Option<Board> = None;
    for uci in moves.split_whitespace() {
        let board = without_history.as_ref().unwrap_or(bs.board());
        let mv = parse_uci_move(board, uci)
            .filter(|&mv| board.is_legal(mv))
            .ok_or_else(|| format!("illegal move {uci} in {board}"))?;
        match &mut without_history {
            Some(board) => board.play_unchecked(mv),
            None if bs.is_terminal() => {
                let mut board = bs.board().clone();
                board.play_unchecked(mv);
                without_history = Some(board);
            }
            None => bs.play(mv),
        }
    }
    Ok(without_history.map_or(bs, BoardStack::new))
}

/// Parses a standard UCI move. cozy-chess encodes castling as the king capturing its rook.
fn parse_uci_move(board: &Board, uci: &str) -> Option<Move> {
    let mut mv: Move = uci.parse().ok()?;
    if board.piece_on(mv.from) == Some(Piece::King) {
        mv.to = match (mv.from, mv.to) {
            (Square::E1, Square::G1) => Square::H1,
            (Square::E1, Square::C1) => Square::A1,
            (Square::E8, Square::G8) => Square::H8,
            (Square::E8, Square::C8) => Square::A8,
            _ => mv.to,
        };
    }
    Some(mv)
}

fn first_legal_move(board: &Board) -> Option<Move> {
    let mut first = None;
    board.generate_moves(|moves| {
        first = moves.into_iter().next();
        first.is_some()
    });
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_tracks_side_to_move_and_castling() {
        let bs = replay("startpos", "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1").unwrap();
        assert_eq!(bs.board().side_to_move(), Color::Black);
        assert_eq!(bs.board().piece_on(Square::G1), Some(Piece::King));
        assert_eq!(bs.board().piece_on(Square::F1), Some(Piece::Rook));
    }

    #[test]
    fn replay_keeps_repetition_history() {
        let bs = replay("startpos", "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8").unwrap();
        assert!(
            bs.is_terminal(),
            "threefold repetition must be visible to the search"
        );
    }

    #[test]
    fn replay_continues_past_an_unclaimed_draw() {
        let bs = replay("startpos", "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 e2e4").unwrap();
        assert!(!bs.is_terminal());
        assert_eq!(bs.board().side_to_move(), Color::Black);
    }

    #[test]
    fn replay_rejects_illegal_moves() {
        assert!(replay("startpos", "e2e5").is_err());
    }

    #[test]
    fn castling_is_sent_in_standard_notation() {
        let bs = replay("startpos", "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6").unwrap();
        let castle = parse_uci_move(bs.board(), "e1g1").unwrap();
        assert!(bs.board().is_legal(castle));
        assert_eq!(
            check_castling_move_on_output(&bs, castle).to_string(),
            "e1g1"
        );
    }
}
