//! Fixed anchor ladder: measures a net against things that never change, so Elo is not
//! relative to the last accepted net (gate Elo inflates over time).
//!
//! ```text
//! ladder match   --net NET.pt --anchor ANCHOR --games N --nodes K [--openings FILE]
//! ladder tactics --net NET.pt --suite FILE --nodes K
//! ```
//! ANCHOR is a frozen net path, `random` (uniform random legal move) or
//! `stockfish:<path>:<skill 0-20>` (UCI engine, `go nodes K`, one thread).
//!
//! The result is one line on stdout starting with `LADDER_RESULT ` followed by JSON, for
//! the trainer (`src/python/ladder.py`) to log. Search settings are fixed here (pinned by the
//! commit), not taken from truezero.toml, so every rung of the ladder is measured the same way.

use cozy_chess::{Board, Color, GameStatus, Move, Piece, Square};
use flume::Sender;
use lru::LruCache;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::{
    io::{BufRead, BufReader, Write},
    num::NonZeroUsize,
    process::{self, Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    executor::{executor_main, Packet},
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::NonTrainerSearch},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
    uci::check_castling_move_on_output,
};

/// Games longer than this are adjudicated a draw (the random anchor can shuffle for a
/// long time before the 50-move rule).
const MAX_PLIES: usize = 600;
/// Opening selection is seeded so every ladder run plays the same openings.
const OPENING_SEED: u64 = 0;

/// The gate's (trainenginetest) evaluation settings, with the node count from the
/// command line. Fixed for the ladder's lifetime so its numbers stay comparable.
fn search_settings(nodes: u32) -> SearchSettings {
    SearchSettings {
        fpu: FPUSettings {
            root_fpu: 0.6,
            children_fpu: 0.6,
        },
        wdl: EvalMode::Wdl,
        moves_left: Some(MovesLeftSettings {
            moves_left_weight: 0.03,
            moves_left_clip: 20.0,
            moves_left_sharpness: 0.5,
        }),
        c_puct: CPUCTSettings {
            root_c_puct: 4.0,
            children_c_puct: 2.0,
        },
        max_nodes: Some(nodes as u128),
        alpha: 0.03,
        eps: 0.25,
        search_type: NonTrainerSearch,
        pst: PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        },
        batch_size: 1,
    }
}

/// Elo of a score, `None` at 0 or 1 (infinite).
fn elo_of(score: f64) -> Option<f64> {
    (score > 0.0 && score < 1.0).then(|| -400.0 * (1.0 / score - 1.0).log10())
}

/// (low, mid, high) Elo with a 95% normal interval on the per-game score, as `elo::elo_wld`
/// does, except that a bound at 0% or 100% is `None` rather than 0. (elo_wld is the gate's
/// and is left alone.)
fn elo_interval(scores: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let n = scores.len() as f64;
    if n == 0.0 {
        return (None, None, None);
    }
    let mu = scores.iter().sum::<f64>() / n;
    let sd = (scores.iter().map(|s| (s - mu).powi(2)).sum::<f64>() / n).sqrt() / n.sqrt();
    let z = 1.959_963_984_540_054;
    (elo_of(mu - z * sd), elo_of(mu), elo_of(mu + z * sd))
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn required(args: &[String], name: &str) -> String {
    arg(args, name).unwrap_or_else(|| fail(&format!("missing {name}")))
}

fn fail(msg: &str) -> ! {
    eprintln!("[ladder] {msg}");
    process::exit(2)
}

/// Starts an executor for `net` and returns the channel searches send positions to.
/// The net sender is leaked on purpose: an executor exits the process when it disconnects.
fn start_executor(net: &str, id: usize, batch_size: usize) -> Sender<Packet> {
    let (net_send, net_recv) = flume::bounded::<String>(1);
    let (tensor_send, tensor_recv) = flume::bounded::<Packet>(4 * batch_size);
    let (ready_send, ready_recv) = flume::bounded::<bool>(1);
    std::thread::Builder::new()
        .name(format!("executor-{id}"))
        .spawn(move || {
            executor_main(
                net_recv,
                tensor_recv,
                batch_size,
                None,
                Some(ready_send),
                id,
            )
        })
        .expect("spawn executor");
    net_send.send(net.to_owned()).expect("send net");
    ready_recv.recv().expect("executor failed to load net");
    std::mem::forget(net_send);
    tensor_send
}

/// Standard UCI notation for a move (cozy-chess encodes castling as king takes rook).
fn uci_string(bs: &BoardStack, mv: Move) -> String {
    check_castling_move_on_output(bs, mv).to_string()
}

fn parse_standard_uci(board: &Board, uci: &str) -> Option<Move> {
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
    board.is_legal(mv).then_some(mv)
}

fn legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::new();
    board.generate_moves(|m| {
        moves.extend(m);
        false
    });
    moves
}

fn parse_fen(text: &str) -> Option<Board> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() >= 6 {
        if let Ok(b) = Board::from_fen(&fields[..6].join(" "), false) {
            return Some(b);
        }
    }
    // EPD: four FEN fields followed by operations.
    if fields.len() >= 4 {
        return Board::from_fen(&format!("{} 0 1", fields[..4].join(" ")), false).ok();
    }
    None
}

struct Stockfish {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Stockfish {
    fn new(path: &str, skill: u32) -> Self {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| fail(&format!("cannot start {path}: {e}")));
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut sf = Self {
            child,
            stdin,
            stdout,
        };
        sf.send("uci");
        sf.wait_for("uciok");
        sf.send(&format!("setoption name Skill Level value {skill}"));
        sf.send("setoption name Threads value 1");
        sf.send("setoption name Hash value 16");
        sf.send("isready");
        sf.wait_for("readyok");
        sf
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").expect("write to stockfish");
        self.stdin.flush().expect("flush stockfish");
    }

    fn wait_for(&mut self, prefix: &str) -> String {
        let mut line = String::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line).expect("read stockfish") == 0 {
                fail("stockfish exited");
            }
            if line.starts_with(prefix) {
                return line.trim().to_owned();
            }
        }
    }

    fn best_move(&mut self, start_fen: &str, history: &[String], nodes: u32) -> String {
        let moves = if history.is_empty() {
            String::new()
        } else {
            format!(" moves {}", history.join(" "))
        };
        self.send(&format!("position fen {start_fen}{moves}"));
        self.send(&format!("go nodes {nodes}"));
        let line = self.wait_for("bestmove");
        line.split_whitespace().nth(1).unwrap_or("").to_owned()
    }
}

impl Drop for Stockfish {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

#[derive(Clone)]
enum Anchor {
    Net(Sender<Packet>),
    Random,
    Stockfish(String, u32),
}

enum Player {
    Net(Sender<Packet>, LruCache<CacheEntryKey, ZeroEvaluationAbs>),
    Random(StdRng),
    Stockfish(Stockfish),
}

impl Player {
    fn new(anchor: &Anchor, nodes: u32, seed: u64) -> Self {
        match anchor {
            Anchor::Net(s) => Player::Net(
                s.clone(),
                LruCache::new(NonZeroUsize::new(nodes.max(1) as usize).unwrap()),
            ),
            Anchor::Random => Player::Random(StdRng::seed_from_u64(seed)),
            Anchor::Stockfish(path, skill) => Player::Stockfish(Stockfish::new(path, *skill)),
        }
    }

    fn choose(&mut self, bs: &BoardStack, start_fen: &str, history: &[String], nodes: u32) -> Move {
        match self {
            Player::Net(sender, cache) => {
                let (mv, ..) = futures::executor::block_on(get_move(
                    bs.clone(),
                    sender.clone(),
                    search_settings(nodes),
                    None,
                    cache,
                ));
                mv
            }
            Player::Random(rng) => *legal_moves(bs.board()).choose(rng).unwrap(),
            Player::Stockfish(sf) => {
                let uci = sf.best_move(start_fen, history, nodes);
                parse_standard_uci(bs.board(), &uci)
                    .unwrap_or_else(|| fail(&format!("stockfish played illegal {uci:?}")))
            }
        }
    }
}

/// Plays one game; returns the score of `net` (1, 0.5 or 0).
fn play_game(
    net: &mut Player,
    anchor: &mut Player,
    start: &Board,
    net_white: bool,
    nodes: u32,
) -> f64 {
    let start_fen = format!("{start}");
    let mut bs = BoardStack::new(start.clone());
    let mut history: Vec<String> = Vec::new();
    while bs.status() == GameStatus::Ongoing && history.len() < MAX_PLIES {
        let white_to_move = bs.board().side_to_move() == Color::White;
        let player = if white_to_move == net_white {
            &mut *net
        } else {
            &mut *anchor
        };
        let mv = player.choose(&bs, &start_fen, &history, nodes);
        history.push(uci_string(&bs, mv));
        bs.play(mv);
    }
    match bs.status() {
        // BoardStack::winner: the side to move at a won terminal is the mated side.
        GameStatus::Won => match bs.winner() {
            Some(c) if (c == Color::White) == net_white => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        },
        _ => 0.5,
    }
}

fn load_openings(path: Option<String>, pairs: usize) -> (Vec<Board>, String) {
    let mut rng = StdRng::seed_from_u64(OPENING_SEED);
    if let Some(p) = path.as_ref().filter(|p| !std::path::Path::new(p).is_file()) {
        eprintln!("[ladder] openings file {p} not found; using seeded random openings");
    }
    if let Some(path) = path.filter(|p| std::path::Path::new(p).is_file()) {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| fail(&format!("{e}")));
        let mut boards: Vec<Board> = text.lines().filter_map(parse_fen).collect();
        if !boards.is_empty() {
            boards.shuffle(&mut rng);
            boards.truncate(pairs);
            return (boards, path);
        }
    }
    // No book: short seeded random openings, so colour-swapped pairs still differ.
    let boards = (0..pairs)
        .map(|_| loop {
            let mut b = Board::default();
            for _ in 0..4 {
                let moves = legal_moves(&b);
                b.play(moves[rng.gen_range(0..moves.len())]);
            }
            if !legal_moves(&b).is_empty() {
                break b;
            }
        })
        .collect();
    (boards, "seeded random 4-ply openings".to_owned())
}

fn parse_anchor(spec: &str, batch: usize) -> Anchor {
    if spec == "random" {
        Anchor::Random
    } else if let Some(rest) = spec.strip_prefix("stockfish:") {
        let (path, skill) = rest
            .rsplit_once(':')
            .unwrap_or_else(|| fail("anchor stockfish:<path>:<skill>"));
        let skill = skill
            .parse()
            .unwrap_or_else(|_| fail("stockfish skill must be 0-20"));
        Anchor::Stockfish(path.to_owned(), skill)
    } else {
        Anchor::Net(start_executor(spec, 1, batch))
    }
}

fn run_match(args: &[String]) {
    let net_path = required(args, "--net");
    let anchor_spec = required(args, "--anchor");
    let games: usize = required(args, "--games")
        .parse()
        .unwrap_or_else(|_| fail("--games"));
    let nodes: u32 = required(args, "--nodes")
        .parse()
        .unwrap_or_else(|_| fail("--nodes"));
    let pairs = games.div_ceil(2);
    let (openings, opening_source) = load_openings(arg(args, "--openings"), pairs);
    let pairs = openings.len();
    // One OS thread per pair (stockfish anchors are processes, so fewer of those).
    let workers = if anchor_spec.starts_with("stockfish:") {
        pairs.min(8)
    } else {
        pairs.min(64)
    }
    .max(1);

    let net = start_executor(&net_path, 0, workers);
    let anchor = parse_anchor(&anchor_spec, workers);
    let next = AtomicUsize::new(0);
    let scores = Mutex::new(Vec::<f64>::new());
    std::thread::scope(|s| {
        for w in 0..workers {
            let (net, anchor, next, scores, openings) =
                (net.clone(), anchor.clone(), &next, &scores, &openings);
            s.spawn(move || {
                let mut me = Player::new(&Anchor::Net(net), nodes, 0);
                let mut them = Player::new(&anchor, nodes, w as u64 + 1);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(start) = openings.get(i) else { break };
                    for net_white in [true, false] {
                        let score = play_game(&mut me, &mut them, start, net_white, nodes);
                        scores.lock().unwrap().push(score);
                    }
                }
            });
        }
    });
    let scores = scores.into_inner().unwrap();
    let wins = scores.iter().filter(|&&x| x == 1.0).count() as u32;
    let losses = scores.iter().filter(|&&x| x == 0.0).count() as u32;
    let draws = scores.len() as u32 - wins - losses;
    let (elo_low, elo, elo_high) = elo_interval(&scores);
    let result = serde_json::json!({
        "kind": "match",
        "net": net_path,
        "anchor": anchor_spec,
        "nodes": nodes,
        "games": scores.len(),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": scores.iter().sum::<f64>() / scores.len().max(1) as f64,
        "elo": elo,
        "elo_low": elo_low,
        "elo_high": elo_high,
        "openings": opening_source,
    });
    println!("LADDER_RESULT {result}");
    process::exit(0)
}

struct Puzzle {
    board: Board,
    solutions: Vec<String>,
    id: String,
}

/// Suite lines: `<FEN or EPD position> | <solution moves, standard UCI> | <id>`, with
/// `#` comments. The id's first `-`-separated word is the category (e.g. `mate2-17`).
fn load_suite(path: &str) -> Vec<Puzzle> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| {
            let parts: Vec<&str> = l.split('|').map(str::trim).collect();
            if parts.len() != 3 {
                fail(&format!("bad suite line {l:?}"));
            }
            let board = parse_fen(parts[0]).unwrap_or_else(|| fail(&format!("bad FEN {l:?}")));
            let solutions: Vec<String> = parts[1].split_whitespace().map(str::to_owned).collect();
            for s in &solutions {
                if parse_standard_uci(&board, s).is_none() {
                    fail(&format!("illegal solution {s} in {l:?}"));
                }
            }
            Puzzle {
                board,
                solutions,
                id: parts[2].to_owned(),
            }
        })
        .collect()
}

fn run_tactics(args: &[String]) {
    let net_path = required(args, "--net");
    let suite_path = required(args, "--suite");
    let nodes: u32 = required(args, "--nodes")
        .parse()
        .unwrap_or_else(|_| fail("--nodes"));
    let puzzles = load_suite(&suite_path);
    let workers = puzzles.len().clamp(1, 64);
    let net = start_executor(&net_path, 0, workers);
    let next = AtomicUsize::new(0);
    let solved = Arc::new(Mutex::new(Vec::<(String, bool)>::new()));
    std::thread::scope(|s| {
        for _ in 0..workers {
            let (net, next, solved, puzzles) = (net.clone(), &next, solved.clone(), &puzzles);
            s.spawn(move || {
                let mut player = Player::new(&Anchor::Net(net), nodes, 0);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(p) = puzzles.get(i) else { break };
                    let bs = BoardStack::new(p.board.clone());
                    let mv = player.choose(&bs, "", &[], nodes);
                    let ok = p.solutions.contains(&uci_string(&bs, mv));
                    solved.lock().unwrap().push((p.id.clone(), ok));
                }
            });
        }
    });
    let solved = solved.lock().unwrap();
    let mut by_category = serde_json::Map::new();
    let mut categories: Vec<String> = solved
        .iter()
        .map(|(id, _)| id.split('-').next().unwrap_or("").to_owned())
        .collect();
    categories.sort();
    categories.dedup();
    for c in categories {
        let in_c: Vec<bool> = solved
            .iter()
            .filter(|(id, _)| id.split('-').next() == Some(c.as_str()))
            .map(|&(_, ok)| ok)
            .collect();
        let n = in_c.iter().filter(|&&ok| ok).count();
        by_category.insert(c, serde_json::json!(n as f64 / in_c.len() as f64));
    }
    let n_solved = solved.iter().filter(|(_, ok)| *ok).count();
    let result = serde_json::json!({
        "kind": "tactics",
        "net": net_path,
        "suite": suite_path,
        "nodes": nodes,
        "positions": solved.len(),
        "solved": n_solved,
        "solved_fraction": n_solved as f64 / solved.len().max(1) as f64,
        "by_category": by_category,
    });
    println!("LADDER_RESULT {result}");
    process::exit(0)
}

fn main() {
    tch::set_num_threads(1);
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("match") => run_match(&args),
        Some("tactics") => run_tactics(&args),
        _ => fail(
            "usage: ladder match --net NET --anchor (NET|random|stockfish:PATH:SKILL) \
             --games N --nodes K [--openings FILE]\n       ladder tactics --net NET \
             --suite FILE --nodes K",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_interval_matches_elo_wld_inside_and_is_none_at_the_edges() {
        // 6 wins, 2 draws, 2 losses: interval inside (0, 1), so it agrees with elo_wld
        // (whose inverse erf is an approximation, hence a few Elo tolerance on the bounds).
        let scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0];
        let (lo, mid, hi) = elo_interval(&scores);
        let (wlo, wmid, whi) = tzrust::elo::elo_wld(6, 2, 2);
        assert!((lo.unwrap() - wlo).abs() < 3.0, "{lo:?} {wlo}");
        assert!((mid.unwrap() - wmid).abs() < 1e-9);
        assert!((hi.unwrap() - whi).abs() < 3.0, "{hi:?} {whi}");
        // All wins: Elo is unbounded above, not 0.
        assert_eq!(elo_interval(&[1.0; 4]), (None, None, None));
        let (lo, mid, hi) = elo_interval(&[1.0, 1.0, 1.0, 0.5]);
        assert!(lo.is_some() && mid.unwrap() > 0.0 && hi.is_none());
    }

    #[test]
    fn standard_castling_parses_to_cozy_and_back() {
        let board = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false).unwrap();
        let mv = parse_standard_uci(&board, "e1g1").unwrap();
        assert_eq!(mv.to, Square::H1);
        assert_eq!(uci_string(&BoardStack::new(board.clone()), mv), "e1g1");
        assert!(parse_standard_uci(&board, "e1e3").is_none());
    }

    #[test]
    fn epd_and_fen_lines_parse() {
        assert!(parse_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").is_some());
        assert!(
            parse_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - bm e7e5;").is_some()
        );
    }

    /// The bundled mate suite loads, every solution is legal, and each position is what it
    /// claims: every solution of a mate-in-1 mates.
    #[test]
    fn bundled_suite_is_consistent() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../python/eval/mate_suite.txt");
        let puzzles = load_suite(path);
        assert!(puzzles.len() >= 50);
        for p in puzzles.iter().filter(|p| p.id.starts_with("mate1")) {
            for s in &p.solutions {
                let mut b = p.board.clone();
                b.play(parse_standard_uci(&p.board, s).unwrap());
                assert_eq!(b.status(), GameStatus::Won, "{} {s}", p.id);
            }
        }
    }
}
