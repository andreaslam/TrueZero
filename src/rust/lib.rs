use std::path::{Path, PathBuf};

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Data folders (nets/, games/, hidden/, ...) live under the workspace root.
pub fn data_path(rel: &str) -> PathBuf {
    workspace_root().join(rel)
}
pub fn data_path_str(rel: &str) -> String {
    data_path(rel).to_string_lossy().into_owned()
}

pub mod boardmanager;
pub mod cache;
pub mod dataformat;
pub mod decoder;
pub mod dirichlet;
pub mod elo;
pub mod executor;
pub mod fileformat;
pub mod mcts;
pub mod mcts_trainer;
pub mod message_types;
pub mod mvs;
pub mod selfplay;
pub mod settings;
pub mod uci;
pub mod utils;

#[cfg(test)]
mod invariant_tests;

#[cfg(test)]

mod tests {
    use super::*;

    use boardmanager::BoardStack;
    use cozy_chess::{Board, Color, GameStatus, Move, Piece, Square};
    use mcts_trainer::{EvalMode, Node, Tree, TypeRequest};
    use settings::{CPUCTSettings, FPUSettings, PSTSettings, SearchSettings};

    extern crate flume;

    /// helper function to create a dummy BoardStack

    fn create_board_stack() -> BoardStack {
        let board = Board::default();
        BoardStack::new(board)
    }
    /// test threefold repetition
    #[test]
    fn test_rep_in_3s() {
        let mut bs = create_board_stack();
        bs.play("g1f3".parse().unwrap());
        bs.play("g8f6".parse().unwrap());
        bs.play("f3g1".parse().unwrap());
        bs.play("f6g8".parse().unwrap());
        bs.play("g1f3".parse().unwrap());
        bs.play("g8f6".parse().unwrap());
        bs.play("f3g1".parse().unwrap());
        bs.play("f6g8".parse().unwrap());

        assert_eq!(bs.status(), GameStatus::Drawn);
    }
    /// test repetition counter
    #[test]
    fn test_count_reps() {
        let mut bs = create_board_stack();
        assert_eq!(bs.get_reps(), 0);
        bs.play("g1f3".parse().unwrap());
        assert_eq!(bs.get_reps(), 0);
        bs.play("g8f6".parse().unwrap());
        assert_eq!(bs.get_reps(), 0);
    }
    /// test draw_conditions for bare kings
    #[test]
    fn test_draw_conditions_bare_kings() {
        let board = Board::from_fen("8/5K2/8/3k4/8/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }
    /// test draw_conditions for knight and king only
    #[test]
    fn test_draw_conditions_knight_king() {
        let board = Board::from_fen("8/5K2/4N3/8/2k5/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }
    /// test draw_condition for bishop and king only
    #[test]
    fn test_draw_conditions_bishop_king() {
        let board = Board::from_fen("8/5K2/5B2/8/4k3/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }

    fn play_uci(board: &mut BoardStack, uci: &str) {
        let from = uci[0..2].parse::<Square>().unwrap();
        let to = uci[2..4].parse::<Square>().unwrap();

        let promotion = uci.chars().nth(4).map(|c| match c {
            'q' => Piece::Queen,
            'r' => Piece::Rook,
            'b' => Piece::Bishop,
            'n' => Piece::Knight,
            _ => panic!("Invalid promotion piece"),
        });

        board.play(Move {
            from,
            to,
            promotion,
        });
    }

    #[test]
    fn en_passant_white_to_move_returns_black_pawn_square() {
        // 1. e2-e4
        //
        // Black to move.
        //  e4.
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "e2e4");

        assert_eq!(board.en_passant(), Some(Square::E4));
    }

    #[test]
    fn en_passant_after_black_double_pawn_push() {
        let mut board = BoardStack::new(Board::default());

        // 1. d2-d4
        play_uci(&mut board, "d2d4");

        // 1... h7-h5
        play_uci(&mut board, "h7h5");

        // 2. d4-d5
        play_uci(&mut board, "d4d5");

        // 2... e7-e5
        play_uci(&mut board, "e7e5");

        assert_eq!(board.en_passant(), Some(Square::E5));
    }
    #[test]
    fn en_passant_d_file() {
        // 1. d2-d4
        //
        // Black to move.
        //  d4.
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "d2d4");

        assert_eq!(board.en_passant(), Some(Square::D4));
    }

    #[test]
    fn en_passant_h_file() {
        // 1. h7-h5
        //
        // Need White to make a legal waiting move first.
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "a2a3");
        play_uci(&mut board, "h7h5");

        assert_eq!(board.en_passant(), Some(Square::H5));
    }

    #[test]
    fn en_passant_disappears_after_non_double_pawn_move() {
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "e2e4");

        assert_eq!(board.en_passant(), Some(Square::E4));

        // Black makes a normal move.
        play_uci(&mut board, "a7a6");

        assert_eq!(board.en_passant(), None);
    }

    #[test]
    fn en_passant_disappears_after_pawn_moves_again() {
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "e2e4");

        assert_eq!(board.en_passant(), Some(Square::E4));

        // Black moves.
        play_uci(&mut board, "a7a6");

        // White moves the pawn again.
        play_uci(&mut board, "e4e5");

        assert_eq!(board.en_passant(), None);
    }

    /// helper function to create dummy SearchSettings
    fn create_search_settings() -> SearchSettings {
        SearchSettings {
            max_nodes: Some(100),
            c_puct: CPUCTSettings {
                root_c_puct: 2.0,
                children_c_puct: 2.0,
            },
            fpu: FPUSettings {
                root_fpu: 0.1,
                children_fpu: 0.1,
            },
            pst: PSTSettings {
                root_pst: 1.75,
                children_pst: 1.5,
            },
            eps: 0.25,
            alpha: 0.03,
            search_type: TypeRequest::UCISearch,
            wdl: EvalMode::Wdl,
            moves_left: None,
            batch_size: 1,
        }
    }

    // test tree initialisation
    #[test]
    fn test_tree_initialisation() {
        let board_stack = create_board_stack();
        let settings = create_search_settings();
        let tree = Tree::new(board_stack, settings);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.nodes[0].visits, 0);
    }

    /// test node initialisation
    #[test]
    fn test_node_initialisation() {
        let node = Node::new(0.5, None, None);
        assert_eq!(node.policy, 0.5);
        assert!(node.parent.is_none());
        assert!(node.mv.is_none());
        assert_eq!(node.visits, 0);
        assert!(node.net_evaluation.value.is_nan());
    }

    /// test node selection
    #[test]
    fn test_node_selection() {
        let board_stack = create_board_stack();
        let settings = create_search_settings();
        let mut tree = Tree::new(board_stack, settings);
        let (selected_node, _, _) = tree.select();
        assert_eq!(selected_node, 0);
    }

    /// test PUCT with non-zero FPU, where the initial visit count is 0
    #[test]

    fn test_puct_with_nonzero_fpu() {
        let settings = create_search_settings();
        let node = Node::new(0.5, None, None);
        assert_eq!(
            settings.fpu.root_fpu,
            node.puct_formula(1, 2.0, Color::White, settings)
        )
    }

    /// root FPU and c_puct must apply to the root's children, children settings deeper down
    #[test]
    fn test_root_settings_apply_to_root_children() {
        let mut settings = create_search_settings();
        settings.fpu = FPUSettings {
            root_fpu: 0.7,
            children_fpu: -0.3,
        };
        settings.c_puct = CPUCTSettings {
            root_c_puct: 4.0,
            children_c_puct: 1.0,
        };
        let root_child = Node::new(0.5, Some(0), None);
        let deeper_child = Node::new(0.5, Some(5), None);

        // parent_visits = 1 makes U = 0, so PUCT is exactly the FPU
        assert_eq!(root_child.puct_formula(1, 2.0, Color::White, settings), 0.7);
        assert_eq!(deeper_child.puct_formula(1, 2.0, Color::White, settings), -0.3);

        // U = c_puct * P * sqrt(parent_visits - 1) / (1 + visits) = c_puct * 0.5 * 2
        assert_eq!(root_child.get_u_val(5, settings), 4.0);
        assert_eq!(deeper_child.get_u_val(5, settings), 1.0);
    }

    #[test]
    fn test_backup_adds_no_extra_move_at_evaluated_node() {
        let board_stack = create_board_stack();
        let settings = create_search_settings();
        let mut tree = Tree::new(board_stack, settings);
        tree.nodes[0].net_evaluation.moves_left = 0.0;

        tree.backpropagate(0);

        assert_eq!(tree.nodes[0].total_evaluation.moves_left, 0.0);
    }

    // ---- end-to-end sign checks on forced mates -------------------------------------------
    //
    // These drive the real `Tree::step` (selection, expansion, terminal scoring, backup and
    // PUCT) against a stub network that knows nothing: every non-terminal position gets
    // value 0, uniform WDL and a uniform policy. Any preference for the mating move can
    // therefore only come from the terminal value and the perspective handling around it.

    use cache::CacheEntryKey;
    use dataformat::ZeroEvaluationAbs;
    use executor::{Packet, ReturnMessage, ReturnPacket};
    use lru::LruCache;
    use std::num::NonZeroUsize;
    use tch::{Device, Kind, Tensor};

    const STUB_SEARCH_NODES: u32 = 200;

    fn stub_search_settings() -> SearchSettings {
        SearchSettings {
            // no Dirichlet noise and no root PST, so the search is deterministic
            search_type: TypeRequest::NonTrainerSearch,
            ..create_search_settings()
        }
    }

    /// stands in for `executor.rs`: answers every evaluation request with an uninformative
    /// output (all logits zero), so value = tanh(0) = 0, WDL = (1/3, 1/3, 1/3) and the policy
    /// is uniform over legal moves
    fn spawn_stub_executor() -> flume::Sender<Packet> {
        let (send, recv) = flume::unbounded::<Packet>();
        let policy_len = mvs::get_contents().len() as i64;
        std::thread::spawn(move || {
            for packet in recv.iter() {
                let board_eval = Tensor::zeros([5], (Kind::Float, Device::Cpu));
                let policy = Tensor::zeros([policy_len], (Kind::Float, Device::Cpu));
                let _ = packet
                    .resender
                    .send(ReturnMessage::ReturnMessage(Ok(ReturnPacket {
                        packet: (board_eval, policy),
                        id: packet.id,
                    })));
            }
        });
        send
    }

    fn new_cache() -> LruCache<CacheEntryKey, ZeroEvaluationAbs> {
        LruCache::new(NonZeroUsize::new(10_000).unwrap())
    }

    fn run_stub_search(fen: &str, nodes: u32) -> Tree {
        let board = Board::from_fen(fen, false).unwrap();
        let mut tree = Tree::new(BoardStack::new(board), stub_search_settings());
        let sender = spawn_stub_executor();
        let mut cache = new_cache();
        let sw = std::time::Instant::now();
        futures::executor::block_on(async {
            while tree.nodes[0].visits < nodes {
                tree.step(&sender, sw, 1, &mut cache).await;
            }
        });
        tree
    }

    /// mean W - L of a node from `player`'s point of view
    fn q_from(tree: &Tree, node: usize, player: Color) -> f32 {
        let n = &tree.nodes[node];
        let rel = n.total_evaluation.to_relative(player);
        (rel.wdl.w - rel.wdl.l) / n.visits as f32
    }

    /// mean scalar value of a node from `player`'s point of view
    fn v_from(tree: &Tree, node: usize, player: Color) -> f32 {
        let n = &tree.nodes[node];
        n.total_evaluation.to_relative(player).value / n.visits as f32
    }

    fn assert_mate_in_one_found(fen: &str, mating_move: &str) {
        let tree = run_stub_search(fen, STUB_SEARCH_NODES);
        let mover = tree.board.board().side_to_move();
        let mating_move: Move = mating_move.parse().unwrap();

        let children: Vec<usize> = tree.nodes[0].children.clone().collect();
        assert!(children.len() > 1, "test position needs alternatives");
        let mate_child = *children
            .iter()
            .find(|&&c| tree.nodes[c].mv == Some(mating_move))
            .expect("mating move is not a root child");

        // the mating child is a terminal win for the mover
        let mut after = tree.board.clone();
        after.play(mating_move);
        assert_eq!(after.status(), GameStatus::Won);

        // it must be the uniquely most visited root child
        let mate_visits = tree.nodes[mate_child].visits;
        for &c in &children {
            if c != mate_child {
                assert!(
                    tree.nodes[c].visits < mate_visits,
                    "{fen}: child {:?} has {} visits, mating move {} has {}",
                    tree.nodes[c].mv,
                    tree.nodes[c].visits,
                    mating_move,
                    mate_visits
                );
            }
        }
        // and should take most of the search, not just edge out the others
        assert!(
            mate_visits * 2 > STUB_SEARCH_NODES,
            "{fen}: mating move got only {mate_visits}/{STUB_SEARCH_NODES} visits"
        );

        // Q of the mating child is +1 for the mover and -1 for the mated side,
        // in both the WDL and the scalar value heads
        let eps = 1e-6;
        assert!((q_from(&tree, mate_child, mover) - 1.0).abs() < eps);
        assert!((v_from(&tree, mate_child, mover) - 1.0).abs() < eps);
        assert!((q_from(&tree, mate_child, !mover) + 1.0).abs() < eps);
        assert!((v_from(&tree, mate_child, !mover) + 1.0).abs() < eps);

        // the root's backed-up value must favour the mover as well
        assert!(
            q_from(&tree, 0, mover) > 0.5,
            "{fen}: root Q for the mover is {}",
            q_from(&tree, 0, mover)
        );
    }

    /// back-rank mate, White to move: Ra8#
    #[test]
    fn mcts_finds_mate_in_one_white() {
        assert_mate_in_one_found("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8");
    }

    /// colour-mirrored back-rank mate, Black to move: ...Ra1#
    #[test]
    fn mcts_finds_mate_in_one_black() {
        assert_mate_in_one_found("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1");
    }

    /// a root where the side to move is already checkmated must be scored as a loss for
    /// that side (and a win for the opponent) on its very first, terminal, visit
    fn assert_mated_root_is_loss(fen: &str) {
        let board = Board::from_fen(fen, false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Won);
        let mated = bs.board().side_to_move();

        let mut tree = Tree::new(bs, stub_search_settings());
        // the network must never be consulted for a terminal node: no executor behind this
        let (sender, _recv) = flume::unbounded::<Packet>();
        let mut cache = new_cache();
        // one descent per step, so one step is one visit whatever VIRTUAL_LEAVES says
        tree.virtual_leaves = 1;
        futures::executor::block_on(tree.step(&sender, std::time::Instant::now(), 1, &mut cache));

        assert_eq!(tree.nodes[0].visits, 1);
        assert!(tree.nodes[0].children.is_empty());

        let net = tree.nodes[0].net_evaluation.to_relative(mated);
        assert_eq!(net.value, -1.0);
        assert_eq!((net.wdl.w, net.wdl.d, net.wdl.l), (0.0, 0.0, 1.0));
        assert_eq!(net.moves_left, 0.0);

        assert_eq!(q_from(&tree, 0, mated), -1.0);
        assert_eq!(v_from(&tree, 0, mated), -1.0);
        assert_eq!(q_from(&tree, 0, !mated), 1.0);
    }

    #[test]
    fn mated_root_is_loss_for_black() {
        assert_mated_root_is_loss("R5k1/5ppp/8/8/8/8/8/6K1 b - - 1 1");
    }

    #[test]
    fn mated_root_is_loss_for_white() {
        assert_mated_root_is_loss("6k1/8/8/8/8/8/5PPP/r5K1 w - - 1 1");
    }

    // ---- end to end: real TorchScript network -> self-play -> KZero data file ----
    //
    // Opt-in, because it needs a TorchScript net on disk. From src/python:
    //   python -c "import torch, network; torch.jit.save(torch.jit.script(network.TrueNetXS(num_resBlocks=1, num_hidden=8)).eval(), '/tmp/tz_e2e.pt')"
    //   cargo test --lib -- --ignored selfplay_with_a_real_network
    // then load `E2E_OUT` with `lib.data.file.DataFile.open` and train on it.
    #[test]
    #[ignore]
    fn selfplay_with_a_real_network_writes_a_kzero_file() {
        use crate::executor::{executor_static, Message};
        use crate::fileformat::BinaryOutput;
        use crate::mcts_trainer::TypeRequest::TrainerSearch;
        use crate::selfplay::{CollectorMessage, DataGen};

        const E2E_NET: &str = "/tmp/tz_e2e.pt";
        const E2E_OUT: &str = "/tmp/tz_e2e_games";
        let net = E2E_NET.to_string();
        let out = E2E_OUT.to_string();

        let (tensor_send, tensor_recv) = flume::bounded::<Packet>(1);
        let (ctrl_send, ctrl_recv) = flume::bounded::<Message>(1);
        let (nps_send, _nps_recv) = flume::unbounded::<CollectorMessage>();
        let executor = std::thread::spawn(move || executor_static(net, tensor_recv, ctrl_recv, 1));

        let mut settings = stub_search_settings();
        settings.search_type = TrainerSearch(None);
        settings.max_nodes = Some(8);
        let mut cache = LruCache::new(NonZeroUsize::new(1000).unwrap());
        let mut output = BinaryOutput::new(&out, "chess").unwrap();
        let datagen = DataGen { iterations: 1 };
        for game in 0..2 {
            let simulation = futures::executor::block_on(datagen.play_game(
                &tensor_send,
                &nps_send,
                &settings,
                game,
                &mut cache,
                None,
                None,
            ));
            assert!(!simulation.positions.is_empty());
            output.append(&simulation).unwrap();
        }
        output.finish().unwrap();

        ctrl_send.send(Message::StopServer).unwrap();
        drop(tensor_send);
        executor.join().unwrap();
    }
}
