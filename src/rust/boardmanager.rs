use cozy_chess::*;

#[derive(PartialEq, Clone, Debug)]
pub struct BoardStack {
    board: Board,
    move_stack: Vec<u64>,
    status: GameStatus,
}

impl BoardStack {
    pub fn new(board: Board) -> Self {
        Self {
            status: board.status(),
            board,
            move_stack: Vec::new(),
        }
    }

    // get the number of previous occurrences of the current position

    pub fn get_reps(&self) -> usize {
        let target = repetition_hash(&self.board);

        self.move_stack
            .iter()
            .filter(|&hash| *hash == target)
            .count()
    }

    // return the square occupied by the pawn that just double-stepped
    // only return a square when an en passant capture is currently possible
    //
    // e2-e4 -> Some(e4)
    // e7-e5 -> Some(e5)

    pub fn en_passant(&self) -> Option<Square> {
        let file = self.board.en_passant()?;

        let rank = match self.board.side_to_move() {
            // black just moved e7-e5, so the pawn is on rank 5
            Color::White => Rank::Fifth,

            // white just moved e2-e4, so the pawn is on rank 4
            Color::Black => Rank::Fourth,
        };

        Some(Square::new(file, rank))
    }

    // play a move

    pub fn play(&mut self, mv: Move) {
        assert!(
            self.status == GameStatus::Ongoing,
            "Cannot play a move after the game has ended"
        );

        self.move_stack.push(repetition_hash(&self.board));

        self.board.play(mv);

        self.status = self.status();
    }

    pub fn is_terminal(&self) -> bool {
        self.status() != GameStatus::Ongoing
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Winner of a finished game, or `None` for a draw or an ongoing game.
    ///
    /// Invariant: at a `Won` terminal the side to move is the checkmated side,
    /// so the winner is the other colour. This does not depend on how many
    /// plies were played or on which colour moved first in the opening.
    pub fn winner(&self) -> Option<Color> {
        match self.status() {
            GameStatus::Won => Some(!self.board.side_to_move()),
            GameStatus::Drawn | GameStatus::Ongoing => None,
        }
    }

    pub fn status(&self) -> GameStatus {
        let is_all_gone = self.board.occupied().len() == 2;

        let is_sure_draw = self.board.occupied().len() <= 3
            && (self.board.pieces(Piece::Bishop).len() == 1
                || self.board.pieces(Piece::Knight).len() == 1);

        // The 50-move rule is left to `Board::status`, which gives checkmate
        // precedence when mate is delivered on the 100th halfmove.
        if self.get_reps() == 2 || is_all_gone || is_sure_draw {
            GameStatus::Drawn
        } else {
            self.board.status()
        }
    }
}

/// Hash identifying a position for repetition purposes.
///
/// `Board::hash` includes the en passant file after every double pawn push.
/// FIDE only distinguishes positions by en passant when the capture is legal.
fn repetition_hash(board: &Board) -> u64 {
    let Some(file) = board.en_passant() else {
        return board.hash();
    };
    let color = board.side_to_move();
    let target = Square::new(file, Rank::Sixth.relative_to(color));
    let can_capture = get_pawn_attacks(target, !color).into_iter().any(|from| {
        board.piece_on(from) == Some(Piece::Pawn)
            && board.is_legal(Move {
                from,
                to: target,
                promotion: None,
            })
    });
    if can_capture {
        board.hash()
    } else {
        board.hash_without_ep()
    }
}

/// Which of two engines moves in a match game.
///
/// Engine 0 plays White when `engine0_white`, otherwise Black. The choice is
/// made from the side to move, never from a ply counter, so it stays correct
/// for openings with Black to move.
pub fn engine_index(side_to_move: Color, engine0_white: bool) -> usize {
    if (side_to_move == Color::White) == engine0_white {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(fen: &str, moves: &[&str]) -> BoardStack {
        let mut bs = BoardStack::new(fen.parse().unwrap());
        for mv in moves {
            bs.play(mv.parse().unwrap());
        }
        bs
    }

    #[test]
    fn mate_on_hundredth_halfmove_wins() {
        let bs = play("6k1/5ppp/8/8/8/8/8/R3K3 w - - 99 80", &["a1a8"]);
        assert_eq!(bs.status(), GameStatus::Won);
        assert_eq!(bs.winner(), Some(Color::White));
    }

    #[test]
    fn quiet_hundredth_halfmove_draws() {
        let bs = play("6k1/5ppp/8/8/8/8/8/R3K3 w - - 99 80", &["a1a7"]);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }

    const SHUFFLE: [&str; 4] = ["e8d8", "g1f3", "d8e8", "f3g1"];

    #[test]
    fn uncapturable_en_passant_does_not_hide_a_repetition() {
        let mut moves = vec!["e2e4"];
        moves.extend(SHUFFLE);
        let fen = "4k3/8/8/8/8/8/4P3/4K1N1 w - - 0 1";
        assert_eq!(play(fen, &moves).get_reps(), 1);
        moves.extend(SHUFFLE);
        assert_eq!(play(fen, &moves).status(), GameStatus::Drawn);
    }

    #[test]
    fn legal_en_passant_is_a_different_position() {
        let mut moves = vec!["e2e4"];
        moves.extend(SHUFFLE);
        let bs = play("4k3/8/8/8/3p4/8/4P3/4K1N1 w - - 0 1", &moves);
        assert_eq!(bs.get_reps(), 0);
    }

    #[test]
    fn pinned_en_passant_does_not_hide_a_repetition() {
        // d4 is pinned to d8 by the rook, so dxe3 is never legal
        let mut moves = vec!["e2e4"];
        let shuffle = ["d8c8", "g1f3", "c8d8", "f3g1"];
        moves.extend(shuffle);
        let bs = play("3k4/8/8/8/3p4/8/4P3/3RK1N1 w - - 0 1", &moves);
        assert_eq!(bs.get_reps(), 1);
    }
}
