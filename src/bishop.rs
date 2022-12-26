use std::{any::Any, fmt::{Display, Formatter}};

use colored::Colorize;

use crate::{game::Game, piece::{Construct, Move, Piece, DynClone}, player::Player};

#[derive(Clone, Debug)]
pub struct Bishop {
    player: Player
}

impl Piece for Bishop {
    fn get_legal_moves(&self, position: (u8,u8), game: &mut Game) -> Vec<(u8,u8)> {
        let mut moves = Vec::new();
        for (x_sign, y_sign) in [(1,1), (-1,1), (-1,-1), (1,-1)] {
            let mut x = position.0 as i8;
            let mut y = position.1 as i8;
            loop {
                x += x_sign;
                y += y_sign;
                if x < 0 || x > 7 || y < 0 || y > 7 {
                    break;
                }
                let new_pos = (x as u8, y as u8);
                if game.is_not_player(new_pos, self.player) {
                    moves.push(new_pos);
                }
                if game.square_is_none(new_pos) {
                    continue;
                }
                break;
            }
        }
        moves
    }

    fn valid_move(&self, from: (u8,u8), to: (u8,u8), game: &mut Game) -> Move {
        let (x, y) = (to.0 as i8 - from.0 as i8, to.1 as i8 - from.1 as i8);
        let valid = x.abs() == y.abs() && game.check_diag(from, (x,y));
        if valid {Move::Normal} else {Move::Invalid}
    }

    fn player(&self) -> Player {
        self.player
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "bishop"
    }
}

impl Construct for Bishop {
    fn new(player: Player) -> Self {
        Self {
            player
        }
    }
}

impl DynClone for Bishop {
    fn clone_box(&self) -> Box<dyn Piece> {
        Box::new(self.clone())
    }
}

impl Display for Bishop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self.player() {
            Player::One => "♝".white().bold(),
            Player::Two => "♗".black()
        })
    }
}