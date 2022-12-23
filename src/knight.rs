use std::{any::Any, fmt::{Display, Formatter}};

use colored::Colorize;

use crate::{game::Game, piece::{Construct, Move, Piece, DynClone}, player::Player};

#[derive(Clone, Debug)]
pub struct Knight {
    player: Player
}


impl Piece for Knight {
    fn get_legal_moves(&self, position: (u8,u8), game: &Game) -> Vec<(u8,u8)> {
        let mut moves = Vec::new();
        for (x_sign, y_sign) in [(1,2), (2,1), (-1,2), (-2,1), (1,-2), (2,-1), (-1,-2), (-2,-1)] {
            let new_pos = (position.0 as i8 + x_sign, position.1 as i8 + y_sign);
            if new_pos.0 < 0 || new_pos.0 > 7 || new_pos.1 < 0 || new_pos.1 > 7 {
                continue;
            }
            let new_pos = (new_pos.0 as u8, new_pos.1 as u8);
            if game.is_not_ally(new_pos) {
                moves.push(new_pos);
            }
        }
        moves
    }

    fn valid_move(&self, from: (u8,u8), to: (u8,u8), _: &Game) -> Move {
        let (x, y) = (to.0 as i8 - from.0 as i8, to.1 as i8 - from.1 as i8);
        let valid = (x.abs() == 2 && y.abs() == 1) || (x.abs() == 1 && y.abs() == 2);
        if valid {Move::Normal} else {Move::Invalid}
    }

    fn player(&self) -> Player {
        self.player
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "knight"
    }
}

impl Construct for Knight {
    fn new(player: Player) -> Self {
        Self {
            player
        }
    }
}

impl DynClone for Knight {
    fn clone_box(&self) -> Box<dyn Piece> {
        Box::new(self.clone())
    }
}

impl Display for Knight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self.player() {
            Player::One => "♞".white().bold(),
            Player::Two => "♘".black()
        })
    }
}