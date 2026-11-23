//! Invariant tests for search, inference input and replay-file output.
//!
//! Every test here drives the real code (`Tree::step`, `DataGen::play_game`, `BinaryOutput`)
//! against a stub network instead of a TorchScript model. The stub is a pure function of the
//! input tensor the search sends it, so each expected value can be recomputed independently
//! from the position. The policy is uniform unless a test needs to tell moves apart.
//!
//! `INVARIANTS.md` at the repository root lists each invariant with the test that checks it.
//! The ids in the comments below (S1, P1, ...) refer to that file.

use crate::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::{Simulation, ZeroEvaluationAbs, ZeroValuesAbs, ZeroValuesPov},
    decoder::{board_data, convert_board, extract_policy, policy_index},
    executor::{Packet, ReturnMessage, ReturnPacket},
    fileformat::{
        tests::{read_records, Record},
        BinaryOutput,
    },
    mcts_trainer::{get_move_with_early_stop, EvalMode, Tree, TypeRequest, Wdl},
    mvs::get_contents,
    selfplay::{CollectorMessage, DataGen},
    settings::{CPUCTSettings, FPUSettings, PSTSettings, PlayoutCapSettings, SearchSettings},
};
use cozy_chess::{Board, Color, Move};
use lru::LruCache;
use std::num::NonZeroUsize;
use tch::Tensor;

// ---- the stub network ----------------------------------------------------------------------

const INPUT_SCALARS: usize = 8;
const PLANE: usize = 64;
/// Material weights for the 6 piece planes of one colour (pawn to king).
const PIECE_VALUES: [f32; 6] = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0];

#[derive(Clone, Copy, Debug)]
enum StubValue {
    /// The same side-to-move evaluation for every position. `value == w - l`, so any
    /// average of these and of terminal values also satisfies `value == w - l`.
    Constant,
    /// Side-to-move material balance read from the input planes (us minus them).
    Material,
}

#[derive(Clone, Copy, Debug)]
enum StubPolicy {
    /// All logits zero: a uniform prior over legal moves.
    Uniform,
    /// A distinct logit per policy index, so a prior attached to the wrong move shows up.
    Marked,
}

/// WDL logits of the constant stub; softmax gives w - l = 0.5752.
const CONSTANT_WDL_LOGITS: [f32; 3] = [1.0, 0.0, -1.0];
/// atanh(0.5752), so that tanh(value logit) == w - l for the constant stub.
const CONSTANT_VALUE_LOGIT: f32 = 0.655_3;
const STUB_MOVES_LEFT: f32 = 7.0;

fn marked_logit(index: usize) -> f32 {
    ((index * 7919) % 211) as f32 / 50.0
}

/// Raw network outputs `([v_logit, w, d, l, moves_left], policy_logits)` for one input.
fn stub_outputs(input: &[f32], value: StubValue, policy: StubPolicy) -> ([f32; 5], Vec<f32>) {
    let head = match value {
        StubValue::Constant => [
            CONSTANT_VALUE_LOGIT,
            CONSTANT_WDL_LOGITS[0],
            CONSTANT_WDL_LOGITS[1],
            CONSTANT_WDL_LOGITS[2],
            STUB_MOVES_LEFT,
        ],
        StubValue::Material => {
            let plane_sum = |plane: usize| -> f32 {
                let start = (INPUT_SCALARS + plane) * PLANE;
                input[start..start + PLANE].iter().sum()
            };
            let us: f32 = (0..6).map(|p| PIECE_VALUES[p] * plane_sum(p)).sum();
            let them: f32 = (0..6).map(|p| PIECE_VALUES[p] * plane_sum(6 + p)).sum();
            let x = 0.2 * (us - them);
            [x, x, 0.0, -x, STUB_MOVES_LEFT]
        }
    };
    let n = get_contents().len();
    let logits = match policy {
        StubPolicy::Uniform => vec![0.0; n],
        StubPolicy::Marked => (0..n).map(marked_logit).collect(),
    };
    (head, logits)
}

/// Stands in for `executor.rs`: answers each request with `stub_outputs` of its input.
fn spawn_stub(value: StubValue, policy: StubPolicy) -> flume::Sender<Packet> {
    let (send, recv) = flume::unbounded::<Packet>();
    std::thread::spawn(move || {
        for packet in recv.iter() {
            let input = Vec::<f32>::try_from(packet.job.reshape([-1])).unwrap();
            let (head, logits) = stub_outputs(&input, value, policy);
            let _ = packet
                .resender
                .send(ReturnMessage::ReturnMessage(Ok(ReturnPacket {
                    packet: (Tensor::from_slice(&head), Tensor::from_slice(&logits)),
                    id: packet.id,
                })));
        }
    });
    send
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// What `process_board_output` should store for `bs`, from the side to move's point of view,
/// recomputed here without the search: the decoding rules are tanh for the value, softmax for
/// WDL and the legal-move policy, and moves left clamped at 0.
fn expected_pov(bs: &BoardStack, value: StubValue) -> ZeroValuesPov {
    let input = Vec::<f32>::try_from(convert_board(bs)).unwrap();
    let (head, _) = stub_outputs(&input, value, StubPolicy::Uniform);
    let wdl = softmax(&head[1..4]);
    ZeroValuesPov {
        value: head[0].tanh(),
        wdl: Wdl {
            w: wdl[0],
            d: wdl[1],
            l: wdl[2],
        },
        moves_left: head[4].max(0.0),
    }
}

fn expected_priors(bs: &BoardStack, policy: StubPolicy) -> Vec<(Move, f32)> {
    let input = Vec::<f32>::try_from(convert_board(bs)).unwrap();
    let (_, logits) = stub_outputs(&input, StubValue::Constant, policy);
    let (moves, indices) = extract_policy(bs, get_contents());
    let legal_logits: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
    moves.into_iter().zip(softmax(&legal_logits)).collect()
}

// ---- search helpers ------------------------------------------------------------------------

fn settings(search_type: TypeRequest, nodes: u128) -> SearchSettings {
    SearchSettings {
        fpu: FPUSettings {
            root_fpu: 0.1,
            children_fpu: 0.1,
        },
        wdl: EvalMode::Wdl,
        moves_left: None,
        c_puct: CPUCTSettings {
            root_c_puct: 2.0,
            children_c_puct: 2.0,
        },
        max_nodes: Some(nodes),
        alpha: 0.03,
        eps: 0.25,
        search_type,
        pst: PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        },
        batch_size: 1,
    }
}

fn new_cache() -> LruCache<CacheEntryKey, ZeroEvaluationAbs> {
    LruCache::new(NonZeroUsize::new(100_000).unwrap())
}

/// Deterministic search (no Dirichlet noise, no root PST) with the given stub.
fn search(fen: &str, nodes: u32, value: StubValue, policy: StubPolicy) -> Tree {
    let bs = BoardStack::new(Board::from_fen(fen, false).unwrap());
    let mut tree = Tree::new(bs, settings(TypeRequest::NonTrainerSearch, nodes as u128));
    let sender = spawn_stub(value, policy);
    let mut cache = new_cache();
    let sw = std::time::Instant::now();
    futures::executor::block_on(async {
        while tree.nodes[0].visits < nodes {
            tree.step(&sender, sw, 1, &mut cache).await;
        }
    });
    tree
}

/// The position at `node`, rebuilt by replaying the moves on the path from the root.
fn board_at(tree: &Tree, node: usize) -> BoardStack {
    let mut path = vec![];
    let mut cur = node;
    while let Some(parent) = tree.nodes[cur].parent {
        path.push(tree.nodes[cur].mv.unwrap());
        cur = parent;
    }
    let mut bs = tree.board.clone();
    for mv in path.into_iter().rev() {
        bs.play(mv);
    }
    bs
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn assert_values_close(actual: ZeroValuesPov, expected: ZeroValuesPov, what: &str) {
    let pairs = [
        ("value", actual.value, expected.value),
        ("w", actual.wdl.w, expected.wdl.w),
        ("d", actual.wdl.d, expected.wdl.d),
        ("l", actual.wdl.l, expected.wdl.l),
        ("moves_left", actual.moves_left, expected.moves_left),
    ];
    for (name, a, e) in pairs {
        assert!(close(a, e, 1e-5), "{what}: {name} is {a}, expected {e}");
    }
}

/// Positions where both colours have captures, checks and quiet moves, in both colours.
const SEARCH_FENS: [&str; 4] = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 3",
    "r3k2r/pPpp1ppp/8/3Pp3/8/8/PPP2PPP/R3K2R w KQkq e6 0 1",
    "r3k2r/ppp2ppp/8/8/3pP3/8/PpPP1PPP/R3K2R b KQkq e3 0 1",
];

// ---- S: perspective and sign ---------------------------------------------------------------

/// S1. The network's output is from the side to move's point of view, and the tree stores
/// it in absolute (White) terms: `net_evaluation.to_relative(stm)` of every expanded node
/// equals the stub's output for that node's own position.
#[test]
fn s1_tree_stores_the_side_to_move_net_output_as_absolute() {
    for fen in SEARCH_FENS {
        let tree = search(fen, 300, StubValue::Material, StubPolicy::Uniform);
        let mut checked = 0;
        for node in 0..tree.nodes.len() {
            if tree.nodes[node].visits == 0 {
                continue;
            }
            let bs = board_at(&tree, node);
            if bs.is_terminal() {
                continue;
            }
            let stm = bs.board().side_to_move();
            let stored = tree.nodes[node].net_evaluation.to_relative(stm);
            assert_values_close(stored, expected_pov(&bs, StubValue::Material), fen);
            checked += 1;
        }
        assert!(checked > 50, "{fen}: only {checked} expanded nodes");
    }
}

/// S2. A terminal node is scored from the result alone, in absolute terms: White win is
/// (+1, [1, 0, 0]), Black win is (-1, [0, 0, 1]), a draw is (0, [0, 1, 0]); moves left 0.
#[test]
fn s2_terminal_nodes_are_scored_by_the_winner() {
    for fen in SEARCH_FENS {
        let tree = search(fen, 300, StubValue::Material, StubPolicy::Uniform);
        for node in 0..tree.nodes.len() {
            let bs = board_at(&tree, node);
            if tree.nodes[node].visits == 0 || !bs.is_terminal() {
                continue;
            }
            let expected = match bs.winner() {
                Some(Color::White) => (1.0, [1.0, 0.0, 0.0]),
                Some(Color::Black) => (-1.0, [0.0, 0.0, 1.0]),
                None => (0.0, [0.0, 1.0, 0.0]),
            };
            let e = tree.nodes[node].net_evaluation;
            assert_eq!((e.value, [e.wdl.w, e.wdl.d, e.wdl.l]), expected, "{fen}");
            assert_eq!(e.moves_left, 0.0);
        }
    }
}

/// S3. With a material-only evaluator and a uniform prior, the search must win a hanging
/// queen, for White and for Black, and report a positive root Q for the side to move.
/// A sign error anywhere between net output, backup and PUCT turns this into avoiding it.
#[test]
fn s3_search_wins_a_hanging_queen_for_either_colour() {
    for (fen, capture) in [
        ("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "d2d5"),
        ("4k3/3r4/8/8/3Q4/8/8/4K3 b - - 0 1", "d7d4"),
    ] {
        let tree = search(fen, 400, StubValue::Material, StubPolicy::Uniform);
        let capture: Move = capture.parse().unwrap();
        let best = tree.nodes[0]
            .children
            .clone()
            .max_by_key(|&c| tree.nodes[c].visits)
            .unwrap();
        assert_eq!(tree.nodes[best].mv, Some(capture), "{fen}");

        let stm = tree.board.board().side_to_move();
        let root = tree.nodes[0].total_evaluation.to_relative(stm);
        let q = (root.wdl.w - root.wdl.l) / tree.nodes[0].visits as f32;
        assert!(q > 0.0, "{fen}: root Q for the side to move is {q}");
    }
}

// ---- B: backup -----------------------------------------------------------------------------

/// B1. Backup adds absolute values with no sign change. For every node:
///   visits = own + sum(child visits), with own >= 1 once visited
///   total  = own * net_evaluation + sum(child totals)            (value and WDL)
///   total.moves_left = own * net.moves_left + sum(child total.moves_left + child visits)
#[test]
fn b1_backup_conserves_visits_and_values() {
    for fen in SEARCH_FENS {
        let tree = search(fen, 300, StubValue::Material, StubPolicy::Uniform);
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.visits == 0 {
                continue;
            }
            let child_visits: u32 = node.children.clone().map(|c| tree.nodes[c].visits).sum();
            assert!(node.visits > child_visits, "{fen}: node {i}");
            let own = (node.visits - child_visits) as f32;

            let mut expected = ZeroValuesAbs::zeros();
            expected.value = own * node.net_evaluation.value;
            expected.wdl = Wdl {
                w: own * node.net_evaluation.wdl.w,
                d: own * node.net_evaluation.wdl.d,
                l: own * node.net_evaluation.wdl.l,
            };
            expected.moves_left = own * node.net_evaluation.moves_left;
            for c in node.children.clone() {
                let child = &tree.nodes[c];
                if child.visits == 0 {
                    continue;
                }
                expected.value += child.total_evaluation.value;
                expected.wdl.w += child.total_evaluation.wdl.w;
                expected.wdl.d += child.total_evaluation.wdl.d;
                expected.wdl.l += child.total_evaluation.wdl.l;
                expected.moves_left += child.total_evaluation.moves_left + child.visits as f32;
            }
            let t = node.total_evaluation;
            for (name, a, e) in [
                ("value", t.value, expected.value),
                ("w", t.wdl.w, expected.wdl.w),
                ("d", t.wdl.d, expected.wdl.d),
                ("l", t.wdl.l, expected.wdl.l),
                ("moves_left", t.moves_left, expected.moves_left),
            ] {
                assert!(close(a, e, 1e-4), "{fen}: node {i} total {name} {a} != {e}");
            }
        }
    }
}

// ---- P: policy alignment -------------------------------------------------------------------

/// P1. Child i of a node holds the prior of its own move: the softmax over legal moves of
/// `logits[policy_index(move)]`. The root has no PST in a non-trainer search; other nodes
/// get `p^children_pst`, renormalised. `move_idx` lists the policy indices in child order.
#[test]
fn p1_child_priors_follow_their_own_policy_index() {
    for fen in SEARCH_FENS {
        let tree = search(fen, 40, StubValue::Constant, StubPolicy::Marked);
        let pst = tree.settings.pst.children_pst;
        let mut checked = 0;
        for node in 0..tree.nodes.len() {
            let n = &tree.nodes[node];
            if n.children.is_empty() {
                continue;
            }
            let bs = board_at(&tree, node);
            let mut expected = expected_priors(&bs, StubPolicy::Marked);
            if node != 0 {
                let sum: f32 = expected.iter().map(|(_, p)| p.powf(pst)).sum();
                for (_, p) in expected.iter_mut() {
                    *p = p.powf(pst) / sum;
                }
            }
            let children: Vec<usize> = n.children.clone().collect();
            assert_eq!(children.len(), expected.len(), "{fen}");
            for (&c, (mv, p)) in children.iter().zip(&expected) {
                assert_eq!(tree.nodes[c].mv, Some(*mv), "{fen}: child order");
                assert!(
                    close(tree.nodes[c].policy, *p, 1e-5),
                    "{fen}: prior of {mv} is {}, expected {p}",
                    tree.nodes[c].policy
                );
            }
            let indices: Vec<usize> = children
                .iter()
                .map(|&c| policy_index(&bs, tree.nodes[c].mv.unwrap(), get_contents()))
                .collect();
            assert_eq!(n.move_idx.as_ref(), Some(&indices), "{fen}: move_idx");
            checked += 1;
        }
        assert!(checked > 5, "{fen}: only {checked} expanded nodes");
    }
}

// ---- R: self-play records ------------------------------------------------------------------

// Scalar positions in a record, in `Scalars::NAMES` order.
const POS_INDEX: usize = 1;
const GAME_LENGTH: usize = 2;
const ZERO_VISITS: usize = 3;
const IS_FINAL: usize = 5;
const IS_TERMINAL: usize = 6;
const HIT_MOVE_LIMIT: usize = 7;
const AVAILABLE: usize = 8;
const PLAYED_MV: usize = 9;
const FINAL: usize = 11;
const ZERO: usize = 16;
const NET: usize = 21;

fn values_at(r: &Record, at: usize) -> ZeroValuesPov {
    let s = &r.scalars;
    ZeroValuesPov {
        value: s[at],
        wdl: Wdl {
            w: s[at + 1],
            d: s[at + 2],
            l: s[at + 3],
        },
        moves_left: s[at + 4],
    }
}

fn packed_bools(bs: &BoardStack) -> Vec<u8> {
    let (_, bools) = board_data(bs);
    bools
        .chunks(8)
        .map(|c| c.iter().enumerate().fold(0u8, |b, (j, &x)| b | ((x as u8) << j)))
        .collect()
}

/// The 21x8x8 network input rebuilt from a stored record.
fn input_from_record(r: &Record) -> Vec<f32> {
    let mut x = Vec::with_capacity(21 * 64);
    for &s in &r.input_scalars {
        x.extend(std::iter::repeat(s).take(64));
    }
    for i in 0..13 * 64 {
        x.push(((r.bools[i / 8] >> (i % 8)) & 1) as f32);
    }
    x
}

fn play_selfplay_game(start: &str, nodes: u128, sender: &flume::Sender<Packet>) -> Simulation {
    play_selfplay_game_with_cap(start, nodes, sender, None)
}

fn play_selfplay_game_with_cap(
    start: &str,
    nodes: u128,
    sender: &flume::Sender<Packet>,
    playout_cap: Option<PlayoutCapSettings>,
) -> Simulation {
    let (nps_send, _nps_recv) = flume::unbounded::<CollectorMessage>();
    let mut cache = new_cache();
    let bs = BoardStack::new(Board::from_fen(start, false).unwrap());
    let settings = settings(TypeRequest::TrainerSearch(None), nodes);
    futures::executor::block_on(
        DataGen {
            iterations: 1,
            playout_cap,
        }
        .play_game(sender, &nps_send, &settings, 0, &mut cache, Some(bs), None),
    )
}

/// R1-R7. Self-play with the constant stub, written by `BinaryOutput` and read back raw.
/// For every game and every position:
///   R1 the stored input is exactly `convert_board` of the position (the network input);
///   R2 consecutive records are linked by the played move, and `played_mv` is its index;
///   R3 the policy target is (policy_index(move), visits / sum of child visits) per legal move;
///   R4 v is the stub's side-to-move output, so v > 0 and w > l for both colours;
///   R5 z is the result from the side to move's point of view (+1 winner, -1 loser, 0 draw);
///   R6 q, v and z satisfy value == w - l (values and WDL flip together);
///   R7 moves left is game_length + 1 - pos_index, and 0 at the appended final position.
#[test]
fn r_selfplay_records_match_search_inputs_and_results() {
    let sender = spawn_stub(StubValue::Constant, StubPolicy::Uniform);
    let starts = [
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 64),
        ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", 64),
        ("8/8/4k3/8/3PK3/8/8/8 w - - 0 1", 8),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 8),
    ];
    // Moves are sampled from visit counts, so a mate in one is usually but not always played.
    // Replay each mate-in-one start until the side with the mate wins, so z for a White win
    // and a Black win is always exercised (each try succeeds well over half the time).
    let sims: Vec<Simulation> = starts
        .iter()
        .enumerate()
        .map(|(i, &(fen, nodes))| {
            let mut sim = play_selfplay_game(fen, nodes, &sender);
            for _ in 0..30 {
                let mating_side = [Some(Color::White), Some(Color::Black)].get(i);
                if mating_side.map_or(true, |&w| sim.final_board.winner() == w) {
                    break;
                }
                sim = play_selfplay_game(fen, nodes, &sender);
            }
            sim
        })
        .collect();

    let path = std::env::temp_dir().join(format!("tzrust-invariants-{}", std::process::id()));
    let mut out = BinaryOutput::new(&path, "chess").unwrap();
    for sim in &sims {
        out.append(sim).unwrap();
    }
    out.finish().unwrap();
    let (records, _, json) = read_records(&path);
    for ext in ["bin", "off", "json"] {
        let _ = std::fs::remove_file(path.with_extension(ext));
    }
    assert_eq!(json["game_count"].as_u64(), Some(sims.len() as u64));

    let net = expected_pov(&BoardStack::new(Board::default()), StubValue::Constant);
    assert!(net.value > 0.0 && net.wdl.w > net.wdl.l);
    let mut colours_seen = [false; 2];
    let mut r = records.iter();

    for sim in &sims {
        let n = sim.positions.len();
        let winner = sim.final_board.winner();
        let boards: Vec<&BoardStack> = sim
            .positions
            .iter()
            .map(|p| &p.board)
            .chain(std::iter::once(&sim.final_board))
            .collect();

        for (pi, bs) in boards.iter().enumerate() {
            let rec = r.next().expect("fewer records than positions");
            let stm = bs.board().side_to_move();
            let is_final = pi == n;
            let ctx = format!("game from {}, position {pi}", sim.start_board().board());

            // R1
            assert_eq!(rec.bools, packed_bools(bs), "{ctx}: input bools");
            assert_eq!(
                input_from_record(rec),
                Vec::<f32>::try_from(convert_board(bs)).unwrap(),
                "{ctx}: stored input differs from the network input"
            );

            assert_eq!(rec.scalars[POS_INDEX], pi as f32, "{ctx}");
            assert_eq!(rec.scalars[GAME_LENGTH], n as f32, "{ctx}");
            assert_eq!(rec.scalars[IS_FINAL], is_final as u8 as f32, "{ctx}");

            // R5, R6, R7 for z
            let z = values_at(rec, FINAL);
            let expected_z = match winner {
                Some(w) if w == stm => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            assert_eq!(z.value, expected_z, "{ctx}: z from the side to move");
            assert_eq!(z.value, z.wdl.w - z.wdl.l, "{ctx}: z value vs WDL");
            let ml = if is_final { 0 } else { n + 1 - pi };
            assert_eq!(z.moves_left, ml as f32, "{ctx}: final moves left");

            if is_final {
                assert_eq!(
                    rec.scalars[IS_TERMINAL],
                    sim.final_board.is_terminal() as u8 as f32
                );
                assert_eq!(
                    rec.scalars[HIT_MOVE_LIMIT],
                    !sim.final_board.is_terminal() as u8 as f32
                );
                assert_eq!(rec.scalars[PLAYED_MV], -1.0);
                assert!(rec.policy.is_empty());
                continue;
            }
            colours_seen[(stm == Color::Black) as usize] = true;
            let pos = &sim.positions[pi];

            // R2
            let mut next = (*bs).clone();
            next.play(pos.played_mv);
            assert_eq!(
                packed_bools(&next),
                packed_bools(boards[pi + 1]),
                "{ctx}: next record is not this position after the played move"
            );
            assert_eq!(
                rec.scalars[PLAYED_MV],
                policy_index(bs, pos.played_mv, get_contents()) as f32,
                "{ctx}: played_mv index"
            );

            // R3
            let (_, indices) = extract_policy(bs, get_contents());
            let mut expected_policy: Vec<(u32, f32)> = indices
                .iter()
                .map(|&i| i as u32)
                .zip(pos.zero_evaluation.policy.iter().copied())
                .collect();
            expected_policy.sort_by_key(|&(i, _)| i);
            assert_eq!(rec.scalars[AVAILABLE], indices.len() as f32);
            assert_eq!(rec.policy, expected_policy, "{ctx}: policy target");
            let child_visits = rec.scalars[ZERO_VISITS] - 1.0;
            let sum: f32 = rec.policy.iter().map(|&(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-5, "{ctx}: policy sums to {sum}");
            for &(_, p) in &rec.policy {
                let visits = p * child_visits;
                assert!(
                    (visits - visits.round()).abs() < 1e-3,
                    "{ctx}: policy {p} is not visits / {child_visits}"
                );
            }

            // R4
            assert_values_close(values_at(rec, NET), net, &format!("{ctx}: v"));

            // R6 for q
            let q = values_at(rec, ZERO);
            assert!(
                close(q.value, q.wdl.w - q.wdl.l, 1e-4),
                "{ctx}: q value {} vs w - l {}",
                q.value,
                q.wdl.w - q.wdl.l
            );
            assert!(close(q.wdl.w + q.wdl.d + q.wdl.l, 1.0, 1e-4));
        }
    }
    assert!(r.next().is_none(), "more records than positions");
    assert_eq!(colours_seen, [true, true], "records must cover both colours");
    assert_eq!(sims[0].final_board.winner(), Some(Color::White), "decisive game for White");
    assert_eq!(sims[1].final_board.winner(), Some(Color::Black), "decisive game for Black");
}

// ---- M: match play (enginetest, trainenginetest) -------------------------------------------

/// M1. The winner of a finished game is the side that is not to move at a `Won` terminal,
/// however many plies were played and whichever colour started.
#[test]
fn m1_winner_is_the_side_not_to_move_at_checkmate() {
    for (fen, moves, winner) in [
        // White mates in one ply from a White-to-move start (odd ply count)
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", &["a1a8"][..], Some(Color::White)),
        // Black mates in one ply from a Black-to-move start (odd ply count)
        ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", &["a8a1"][..], Some(Color::Black)),
        // fool's mate from the start position: Black wins after an even ply count
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["f2f3", "e7e5", "g2g4", "d8h4"][..],
            Some(Color::Black),
        ),
        // threefold repetition is a draw
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"][..],
            None,
        ),
    ] {
        let mut bs = BoardStack::new(Board::from_fen(fen, false).unwrap());
        for mv in moves {
            assert!(!bs.is_terminal());
            bs.play(mv.parse().unwrap());
        }
        assert!(bs.is_terminal(), "{fen}");
        assert_eq!(bs.winner(), winner, "{fen} {moves:?}");
    }
}

/// M2. In a match game engine 0 plays White iff `engine0_white`, decided by the side to move
/// (so an opening with Black to move does not swap the engines' colours).
#[test]
fn m2_engine_is_chosen_by_side_to_move() {
    use crate::boardmanager::engine_index;
    assert_eq!(engine_index(Color::White, true), 0);
    assert_eq!(engine_index(Color::Black, true), 1);
    assert_eq!(engine_index(Color::White, false), 1);
    assert_eq!(engine_index(Color::Black, false), 0);
}

// ---- U: UCI --------------------------------------------------------------------------------

/// U1. `go nodes N` is honoured; with a clock as well the stricter limit applies; with no
/// limit the default of 1600 visits is used.
#[test]
fn u1_go_nodes_sets_the_visit_limit() {
    use crate::uci::go_visit_limit;
    let none = [None, None];
    assert_eq!(go_visit_limit(Some(800), Color::White, none, none, 30, None), 800);
    assert_eq!(go_visit_limit(None, Color::White, none, none, 30, None), 1600);
    // 30 s on White's clock, 30 moves to go: 1000 ms -> 100 visits
    let clock = [Some(30_000), Some(1)];
    assert_eq!(go_visit_limit(None, Color::White, clock, none, 30, None), 100);
    assert_eq!(go_visit_limit(Some(50), Color::White, clock, none, 30, None), 50);
    assert_eq!(go_visit_limit(Some(5000), Color::White, clock, none, 30, None), 100);
    // movetime 2000 ms -> 200 visits
    assert_eq!(go_visit_limit(Some(5000), Color::Black, none, none, 30, Some(2000)), 200);
}

// ---- E, PCR: early stop and playout cap randomisation --------------------------------------

/// E1. Early stopping returns the move a full search would play and never uses more visits.
/// In a position with one clearly best move (mate in one, a hanging queen) it stops early.
#[test]
fn e1_early_stop_plays_the_full_search_move() {
    let sender = spawn_stub(StubValue::Material, StubPolicy::Uniform);
    let nodes = 400;
    for (fen, decided) in [
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", true),
        ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", true),
        ("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", true),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        ),
    ] {
        // NonTrainerSearch adds no root noise, so both searches are deterministic.
        let settings = settings(TypeRequest::NonTrainerSearch, nodes);
        let run = |early_stop: bool| {
            let bs = BoardStack::new(Board::from_fen(fen, false).unwrap());
            let mut cache = new_cache();
            futures::executor::block_on(get_move_with_early_stop(
                bs, &sender, settings, 0, &mut cache, None, early_stop,
            ))
        };
        let (full_mv, _, _, _, full_visits) = run(false);
        let (early_mv, _, _, _, early_visits) = run(true);
        assert_eq!(early_mv, full_mv, "{fen}");
        assert_eq!(full_visits, nodes as u32, "{fen}");
        assert!(early_visits <= full_visits, "{fen}");
        if decided {
            assert!(
                early_visits < full_visits,
                "{fen}: no early stop ({early_visits})"
            );
        }
    }
}

/// E2. `root_move_decided` is exact: whenever it holds during a search, the most-visited
/// root child at that moment is still the unique most-visited child at the end.
#[test]
fn e2_root_move_decided_never_changes_the_final_move() {
    let sender = spawn_stub(StubValue::Material, StubPolicy::Marked);
    let nodes = 300u32;
    for fen in [
        "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ] {
        let bs = BoardStack::new(Board::from_fen(fen, false).unwrap());
        let mut tree = Tree::new(bs, settings(TypeRequest::NonTrainerSearch, nodes as u128));
        let mut cache = new_cache();
        let sw = std::time::Instant::now();
        let most_visited = |tree: &Tree| {
            tree.nodes[0]
                .children
                .clone()
                .max_by_key(|&c| tree.nodes[c].visits)
                .unwrap()
        };
        let mut decided_at: Vec<usize> = vec![];
        futures::executor::block_on(async {
            while tree.nodes[0].visits < nodes {
                if tree.nodes[0].visits >= 2 && tree.root_move_decided(nodes) {
                    decided_at.push(most_visited(&tree));
                }
                tree.step(&sender, sw, 1, &mut cache).await;
            }
        });
        let final_best = most_visited(&tree);
        let best_visits = tree.nodes[final_best].visits;
        let ties = tree.nodes[0]
            .children
            .clone()
            .filter(|&c| tree.nodes[c].visits == best_visits)
            .count();
        for best in decided_at {
            assert_eq!(best, final_best, "{fen}");
            assert_eq!(ties, 1, "{fen}");
        }
    }
}

/// PCR1. A fast search uses the fast cap (never above the full cap) and no root noise; a
/// full search uses the settings unchanged.
#[test]
fn pcr1_move_settings_for_full_and_fast_searches() {
    let base = settings(TypeRequest::TrainerSearch(None), 800);
    let cap = PlayoutCapSettings::new(0.25, 100, true);
    assert_eq!(cap.move_settings(&base, true), base);
    let fast = cap.move_settings(&base, false);
    assert_eq!(fast.max_nodes, Some(100));
    assert_eq!(fast.eps, 0.0);
    assert_eq!(
        SearchSettings {
            max_nodes: Some(800),
            eps: base.eps,
            ..fast
        },
        base
    );
    let small = settings(TypeRequest::TrainerSearch(None), 50);
    assert_eq!(cap.move_settings(&small, false).max_nodes, Some(50));
}

/// PCR2. In self-play, a move recorded with `is_full_search` used the full node count, and
/// every other move used at most the fast cap. With p = 0 every move is fast, with p = 1
/// every move is full, and in between both kinds appear. The flag is written to the file.
#[test]
fn pcr2_selfplay_records_full_and_fast_searches() {
    let sender = spawn_stub(StubValue::Constant, StubPolicy::Uniform);
    let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let (full_nodes, fast_visits) = (24u128, 6u32);
    let check = |p: f32| {
        let cap = PlayoutCapSettings::new(p, fast_visits, true);
        let sim = play_selfplay_game_with_cap(start, full_nodes, &sender, Some(cap));
        let full_count = sim.positions.iter().filter(|p| p.is_full_search).count();
        for (i, pos) in sim.positions.iter().enumerate() {
            if pos.is_full_search {
                assert_eq!(pos.zero_visits, full_nodes as u64, "ply {i}");
            } else {
                assert!(pos.zero_visits <= fast_visits as u64, "ply {i}");
                assert!(pos.zero_visits >= 2, "ply {i}");
                // early stop only in the greedy phase; sampled moves use the whole fast cap
                if i < crate::selfplay::TEMPERATURE_CUTOFF_PLIES {
                    assert_eq!(pos.zero_visits, fast_visits as u64, "ply {i}");
                }
            }
        }
        (sim, full_count)
    };

    let (sim, full) = check(0.0);
    assert_eq!(full, 0);
    assert!(!sim.positions.is_empty());
    let (sim, full) = check(1.0);
    assert_eq!(full, sim.positions.len());
    let (sim, full) = check(0.5);
    assert!(sim.positions.len() >= 20, "game too short to test mixing");
    assert!(
        full > 0 && full < sim.positions.len(),
        "{full} of {}",
        sim.positions.len()
    );

    let path = std::env::temp_dir().join(format!("tzrust-pcr-{}", std::process::id()));
    let mut out = BinaryOutput::new(&path, "chess").unwrap();
    out.append(&sim).unwrap();
    out.finish().unwrap();
    let (records, _, _) = read_records(&path);
    for ext in ["bin", "off", "json"] {
        let _ = std::fs::remove_file(path.with_extension(ext));
    }
    for (pos, rec) in sim.positions.iter().zip(&records) {
        // scalars: [game_id, pos_index, game_length, zero_visits, is_full_search, ...]
        assert_eq!(rec.scalars[3], pos.zero_visits as f32);
        assert_eq!(rec.scalars[4], pos.is_full_search as u8 as f32);
    }
}
