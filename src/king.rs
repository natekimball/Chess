use std::{any::Any, fmt::{Display, Formatter}};

use colored::Colorize;

use crate::{game::Game, rook::Rook, piece::{Construct, Move, Piece, DynClone}, player::Player};

#[derive(Clone, Debug)]
pub struct King {
    player: Player
}

impl Piece for King {
    fn get_legal_moves(&self, position: (u8,u8), game: &mut Game) -> Vec<(u8, u8)> {
        let mut moves = Vec::new();
        for (x,y) in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]{
            let new_pos = (position.0 as i8 + x, position.1 as i8 + y);
            if new_pos.0 < 0 || new_pos.0 > 7 || new_pos.1 < 0 || new_pos.1 > 7 {
                continue;
            }
            let new_pos = (new_pos.0 as u8, new_pos.1 as u8);
            if game.is_not_player(new_pos, self.player) {
                if !game.try_move_for_check(position, new_pos, self.player) {
                    moves.push(new_pos);
                }
            }
        }
        if game.has_king_moved(self.player) || game.in_check(self.player) {
            return moves;
        }
        if self.can_castle_left(position, game) {
            moves.push((2,position.1));
        }
        if self.can_castle_right(position, game) {
            moves.push((6,position.1));
        }
        moves
    }

    //doesn't  handle friendly fire or moving into check
    fn valid_move(&self, from: (u8,u8), to: (u8,u8), game: &mut Game) -> Move {
        let (x, y) = (to.0 as i8 - from.0 as i8, to.1 as i8 - from.1 as i8);
        if x.abs() < 2 && y.abs() < 2 {
            Move::Normal
        // } else if game.can_castle(from, to) {
        } else if to.0 == 2 && to.1 == from.1 && self.can_castle_left(from, game) {
            Move::Castle
        } else if to.0 == 6 && to.1 == from.1 && self.can_castle_right(from, game) {
            Move::Castle
        } else {
            Move::Invalid
        }
    }

    fn player(&self) -> Player {
        self.player
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "king"
    }
}

impl King {
    fn can_castle_left(&self, position: (u8,u8), game: &mut Game) -> bool {
        let y = match self.player {
            Player::One => 7,
            Player::Two => 0
        };
        if game.has_king_moved(self.player) || position != (4,y) || game.in_check(self.player) {
            return false;
        }
        if game.square_is_none((3,y)) && game.square_is_none((2,y)) && game.square_is_none((1,y)) {
            if let Some(rook) = game.get((0,y)) {
                if rook.is_type::<Rook>() {
                    if !game.has_left_rook_moved(self.player) {
                        return !game.try_move_for_check((4,y), (2,y), self.player);
                    }
                }
            }
        }
        false
    }

    fn can_castle_right(&self , position: (u8,u8), game: &mut Game) -> bool {
        let y = match self.player {
            Player::One => 7,
            Player::Two => 0
        };
        if game.has_king_moved(self.player) || position != (4,y) || game.in_check(self.player) {
            return false;
        }
        if game.square_is_none((5,y)) && game.square_is_none((6,y)) {
            if let Some(rook) = game.get((7,y)) {
                if rook.is_type::<Rook>() {
                    if !game.has_right_rook_moved(self.player) {
                        return !game.try_move_for_check((4,y), (6,y), self.player);
                    }
                }
            }
        }
        false
    }
}

impl Construct for King {
    fn new(player: Player) -> Self {
        Self {
            player
        }
    }
}

impl DynClone for King {
    fn clone_box(&self) -> Box<dyn Piece> {
        Box::new(self.clone())
    }
}

impl Display for King {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let c = match self.player() {
            Player::One => "♚".white().bold(),
            Player::Two => "♔".black()
        };
        write!(f, "{}", c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{game::Board, knight::Knight};

    #[test]
    fn simple_castles() {
        let mut board: Board = vec![vec![None;8];8];
        board[0][0] = Some(Box::new(Rook::new(Player::Two)));
        board[0][4] = Some(Box::new(King::new(Player::Two)));
        board[0][7] = Some(Box::new(Rook::new(Player::Two)));
        board[7][0] = Some(Box::new(Rook::new(Player::One)));
        board[7][4] = Some(Box::new(King::new(Player::One)));
        board[7][7] = Some(Box::new(Rook::new(Player::One)));

        let mut game = Game::new();
        game.set_board(board);

        print!("{game}");

        let king1 = game.get((4,7)).unwrap();
        let king1 = king1.get_piece::<King>().unwrap();
        let king2 = game.get((4,0)).unwrap();
        let king2 = king2.get_piece::<King>().unwrap();

        assert!(king1.can_castle_left((4,7), &mut game));
        assert!(king1.can_castle_right((4,7), &mut game));
        assert!(king2.can_castle_left((4,0), &mut game));
        assert!(king2.can_castle_right((4,0), &mut game));

        assert_eq!(king1.get_legal_moves((4,7), &mut game).len(), 7);
        assert_eq!(king2.get_legal_moves((4,0), &mut game).len(), 7);
    }

    #[test]
    fn cant_castle_into_check() {
        let mut board: Board = vec![vec![None;8];8];
        board[0][0] = Some(Box::new(Rook::new(Player::Two)));
        board[0][4] = Some(Box::new(King::new(Player::Two)));
        board[0][7] = Some(Box::new(Rook::new(Player::Two)));
        board[7][2] = Some(Box::new(Rook::new(Player::One)));
        board[7][4] = Some(Box::new(King::new(Player::One)));
        board[7][6] = Some(Box::new(Rook::new(Player::One)));

        let mut game = Game::new();
        game.set_board(board);

        print!("{game}");

        let king1 = game.get((4,0)).unwrap();
        let king1 = king1.get_piece::<King>().unwrap();
        let king2 = game.get((4,7)).unwrap();
        let king2 = king2.get_piece::<King>().unwrap();

        assert!(!king1.can_castle_left((4,7), &mut game));
        assert!(!king1.can_castle_right((4,7), &mut game));
        assert!(!king2.can_castle_left((4,0), &mut game));
        assert!(!king2.can_castle_right((4,0), &mut game));
    }

    #[test]
    fn moving_out_of_check() {
        let mut board: Board = vec![vec![None;8];8];
        board[5][3] = Some(Box::new(Knight::new(Player::Two)));
        board[7][0] = Some(Box::new(Rook::new(Player::One)));
        board[7][4] = Some(Box::new(King::new(Player::One)));
        board[7][7] = Some(Box::new(Rook::new(Player::One)));

        let mut game = Game::new();
        game.set_board(board);

        print!("{game}");

        let king1 = game.get((4,7)).unwrap();
        let king1 = king1.get_piece::<King>().unwrap();

        assert_eq!(king1.get_legal_moves((4,7), &mut game).len(), 4);
    }
}