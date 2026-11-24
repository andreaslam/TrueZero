use crate::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::{ZeroEvaluationAbs, ZeroValuesPov},
    mcts_trainer::{Net, Node, Tree, Wdl},
    mvs::get_contents,
};
use cozy_chess::{Color, File, Move, Piece, Square};
use lru::LruCache;
use tch::{Device, IValue, Kind, Tensor};

pub fn eval_state(board: Tensor, net: &Net) -> anyhow::Result<(Tensor, Tensor)> {
    // miracle mps fix???
    let _no_grad = tch::no_grad_guard();

    let b = board.reshape([-1, 21, 8, 8]);
    let b = b.to(net.device);
    let board = IValue::Tensor(b);

    let output = net.net.forward_is(&[board])?;

    let output_tensor = match output {
        IValue::Tuple(b) => b,
        a => panic!("the output is not an IValue {:?}", a),
    };

    let (board_eval, policy) = (&output_tensor[0], &output_tensor[1]);

    let board_eval = match board_eval {
        IValue::Tensor(b) => b,
        a => panic!("the output is not a Tensor {:?}", a),
    };

    let policy = match policy {
        IValue::Tensor(b) => b,
        a => panic!("the output is not a Tensor {:?}", a),
    };

    let board_eval = board_eval.to(Device::Cpu);
    let policy = policy.to(Device::Cpu);

    drop(output_tensor);

    Ok((board_eval, policy))
}


fn canonical_square(square: Square, side: Color) -> Square {
    if side == Color::Black {
        square.flip_rank()
    } else {
        square
    }
}

/// Convert a cozy-chess move into the move KZero's `ChessStdMapper` indexes.
///
/// cozy-chess encodes castling as "king takes own rook" (e1h1, e1a1), whereas KZero (via the
/// `chess` crate) encodes it as the king's two-square move (e1g1, e1c1). Without this
/// conversion castling lands on the e1h1/e1a1 rook-move slots of the 1880-move policy.
fn standard_move(bs: &BoardStack, mv: Move) -> Move {
    let board = bs.board();
    let is_castle = board.piece_on(mv.from) == Some(Piece::King)
        && board.color_on(mv.to) == Some(board.side_to_move());
    if !is_castle {
        return mv;
    }
    let king_file = if mv.to.file() > mv.from.file() {
        File::G
    } else {
        File::C
    };
    Move {
        from: mv.from,
        to: Square::new(king_file, mv.from.rank()),
        promotion: None,
    }
}

/// Move from the side to move's point of view: ranks are flipped for Black (KZero `move_pov`).
fn canonical_move(bs: &BoardStack, mv: Move) -> Move {
    let mv = standard_move(bs, mv);
    if bs.board().side_to_move() == Color::Black {
        Move {
            from: mv.from.flip_rank(),
            to: mv.to.flip_rank(),
            promotion: mv.promotion,
        }
    } else {
        mv
    }
}

/// Index of a legal cozy-chess move in the 1880-entry policy, identical to KZero's
/// `ChessStdMapper::move_to_index`.
pub fn policy_index(bs: &BoardStack, mv: Move, contents: &[Move]) -> usize {
    let canonical_mv = canonical_move(bs, mv);
    contents
        .iter()
        .position(|x| x == &canonical_mv)
        .unwrap_or_else(|| {
            panic!(
                "Legal canonical move {} (actual {}) is missing \
                 from the 1880-entry policy move list",
                canonical_mv, mv
            )
        })
}

/// The double-pushed pawn's square, only when a pawn of the side to move stands next to it.
///
/// This matches the `chess` crate that KZero's `ChessStdMapper` reads: it records en passant
/// only when an enemy pawn could capture (pins are not considered). cozy-chess, and therefore
/// `BoardStack::en_passant`, reports every double push.
fn capturable_en_passant(bs: &BoardStack) -> Option<Square> {
    let pawn_square = bs.en_passant()?;
    let us = bs.board().side_to_move();
    let neighbours = pawn_square.rank().bitboard() & pawn_square.file().adjacent();
    let capturers = bs.board().colored_pieces(us, Piece::Pawn) & neighbours;
    if capturers.is_empty() {
        None
    } else {
        Some(pawn_square)
    }
}

pub fn board_data(bs: &BoardStack) -> (Vec<f32>, Vec<bool>) {
    let us = bs.board().side_to_move();

    let mut scalar_data = vec![0.0_f32; 8];

    if us == Color::White {
        scalar_data[0] = 1.0;
    } else {
        scalar_data[1] = 1.0;
    }

    // put us first and opponent second
    let colours: [Color; 2] = if us == Color::White {
        [Color::White, Color::Black]
    } else {
        [Color::Black, Color::White]
    };

    let mut c = 2;

    // KZero ChessStdMapper order: per colour (us, then them) kingside then queenside
    for colour in colours {
        let rights = bs.board().castle_rights(colour);

        scalar_data[c] = if rights.short.is_some() { 1.0 } else { 0.0 };

        scalar_data[c + 1] = if rights.long.is_some() { 1.0 } else { 0.0 };

        c += 2;
    }

    scalar_data[6] = bs.get_reps() as f32;

    scalar_data[7] = bs.board().halfmove_clock() as f32;

    // encode pieces using canonical board coordinates
    let mut pieces_sqs = vec![false; 64 * 12];

    let mut counter = 0;

    for colour in colours {
        for piece in Piece::ALL {
            for square in bs.board().colored_pieces(colour, piece) {
                let canonical = canonical_square(square, us);

                let index = canonical.rank() as usize * 8 + canonical.file() as usize;

                pieces_sqs[index + 64 * counter] = true;
            }

            counter += 1;
        }
    }

    let mut en_passant_plane = vec![false; 64];

    if let Some(pawn_square) = capturable_en_passant(bs) {
        let canonical = canonical_square(pawn_square, us);

        let index = canonical.rank() as usize * 8 + canonical.file() as usize;

        en_passant_plane[index] = true;
    }

    pieces_sqs.extend(en_passant_plane);

    debug_assert_eq!(pieces_sqs.len(), 13 * 64);

    (scalar_data, pieces_sqs)
}

// expand scalar and board data into the network input tensor

/// Network input, identical to what the replay files store (KZero: the stored input is the
/// network input). Channel 7 is the raw halfmove clock; TrueNet scales it internally.
pub fn convert_board(bs: &BoardStack) -> Tensor {
    let (scalar_data, pieces_sqs) = board_data(bs);

    let mut all_data = Vec::with_capacity(21 * 64);

    for value in &scalar_data {
        for _ in 0..64 {
            all_data.push(*value);
        }
    }

    all_data.extend(pieces_sqs.iter().map(|&x| x as u8 as f32));

    debug_assert_eq!(all_data.len(), 21 * 64);

    Tensor::from_slice(&all_data)
}

// map actual legal moves to their canonical policy indices
//
// actual_moves always stay in board coordinates
// only the move used for policy lookup is canonicalised
//
// this keeps the mapping aligned so actual_moves[i] corresponds to idx_li[i]

pub fn extract_policy(bs: &BoardStack, contents: &'static [Move]) -> (Vec<Move>, Vec<usize>) {
    let mut actual_moves: Vec<Move> = Vec::new();

    bs.board().generate_moves(|moves| {
        actual_moves.extend(moves);
        false
    });

    let idx_li = actual_moves
        .iter()
        .map(|&actual_mv| policy_index(bs, actual_mv, contents))
        .collect();

    (actual_moves, idx_li)
}

// decode the network output and update the tree and cache

pub fn process_board_output(
    output: (&Tensor, &Tensor),
    selected_node_idx: &usize,
    tree: &mut Tree,
    bs: &BoardStack,
    cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
) -> Vec<usize> {
    let contents = get_contents();

    let (board_eval, policy) = output;

    // decode the value head

    let board_eval = board_eval.squeeze();

    let board_evals: Vec<f32> =
        Vec::try_from(board_eval).expect("Failed to convert board evaluation tensor");

    assert!(
        board_evals.len() >= 5,
        "Expected at least 5 board-head outputs, got {}",
        board_evals.len()
    );

    let value = board_evals[0].tanh();

    let wdl_logits = Tensor::from_slice(&board_evals[1..4]);

    let wdl_tensor = Tensor::softmax(&wdl_logits, 0, Kind::Float);

    let wdl_values: Vec<f32> = Vec::try_from(wdl_tensor).expect("Failed to convert WDL tensor");

    let wdl = Wdl {
        w: wdl_values[0],
        d: wdl_values[1],
        l: wdl_values[2],
    };

    let moves_left = board_evals[4];

    // decode the policy head

    let policy = policy.squeeze();

    let policy_values: Vec<f32> = Vec::try_from(policy).expect("Failed to convert policy tensor");

    assert_eq!(
        policy_values.len(),
        contents.len(),
        "Network policy head has {} entries but move list has {} entries",
        policy_values.len(),
        contents.len()
    );

    // keep legal moves in actual board coordinates
    // use canonical coordinates only for policy lookup
    let (actual_moves, idx_li) = extract_policy(bs, contents);

    assert_eq!(
        actual_moves.len(),
        idx_li.len(),
        "Legal move/index mapping length mismatch"
    );

    let pol_logits: Vec<f32> = idx_li.iter().map(|&idx| policy_values[idx]).collect();

    let logits = Tensor::from_slice(&pol_logits);

    let probabilities = Tensor::softmax(&logits, 0, Kind::Float);

    let pol_list: Vec<f32> =
        Vec::try_from(probabilities).expect("Failed to convert policy probabilities");

    assert_eq!(
        pol_list.len(),
        actual_moves.len(),
        "Policy probability count does not match legal move count"
    );

    if *selected_node_idx == 0 {
        tree.root_net_policy = Some(pol_list.clone());
    }

    // store the network evaluation

    let moves_left = 0.0_f32.max(moves_left);

    let selected_node_net_evaluation = ZeroValuesPov {
        value,
        wdl,
        moves_left,
    }
    .to_absolute(bs.board().side_to_move());

    tree.nodes[*selected_node_idx].net_evaluation = selected_node_net_evaluation;

    let child_start = tree.nodes.len();

    for (actual_mv, probability) in actual_moves.iter().zip(pol_list.iter()) {
        let child = Node::new(*probability, Some(*selected_node_idx), Some(*actual_mv));

        tree.nodes.push(child);
    }

    let child_end = tree.nodes.len();

    tree.nodes[*selected_node_idx].children = child_start..child_end;

    // cache the network evaluation

    cache.put(
        CacheEntryKey {
            hash: bs.board().hash(),

            halfmove_clock: bs.board().halfmove_clock(),

            repetitions: bs.get_reps() as u8,
        },
        ZeroEvaluationAbs {
            values: selected_node_net_evaluation,

            policy: pol_list,
        },
    );

    idx_li
}
