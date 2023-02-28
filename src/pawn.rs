use std::{any::Any, fmt::{Display, Formatter}};

use colored::Colorize;

use crate::{game::Game, piece::{Construct, Move, Piece, DynClone}, player::Player};

#[derive(Clone, Debug)]
pub struct Pawn {
    player: Player
}


impl Piece for Pawn {
    fn get_legal_moves(&self, position: (u8,u8), game: &mut Game) -> Vec<(u8,u8)> {
        let mut moves = Vec::new();
        let sign = match self.player {
            Player::One => -1,
            Player::Two => 1
        };
        let end: i8 = match self.player {
            Player::One => 1,
            Player::Two => 6
        };
        let new_y = position.1 as i8 + sign;
        if new_y < 0 || new_y > 7 {
            return moves;
        }
        let new_pos = (position.0, new_y as u8);
        if game.square_is_none(new_pos) && !game.try_move_for_check(position, new_pos, self.player) {
            moves.push(new_pos);
            if position.1 == (end - 5*sign) as u8 {
                let new_pos = (new_pos.0, (new_y + sign) as u8);
                if game.square_is_none(new_pos) && !game.try_move_for_check(position, new_pos, self.player){
                    moves.push(new_pos);
                }
            }
        }
        for (x,y) in [(position.0 as i8 + 1, position.1 as i8 + sign), (position.0 as i8 - 1, position.1 as i8 + sign)] {
            if x < 0 || x > 7 || y < 0 || y > 7 {
                continue;
            }
            let new_pos = (x as u8, y as u8);
            if game.is_player(new_pos, self.player.other()) && !game.try_move_for_check(position, new_pos, self.player){
                moves.push(new_pos);
            } else if y == end - sign {
                if let Some(last_double) = game.get_last_double() {
                    if last_double == (x as u8, (y - sign) as u8) && game.square_is_none(new_pos) && !game.try_move_for_check(position, new_pos, self.player) {
                        moves.push(new_pos);
                    }
                }
            }
        }
        moves
    }

    fn valid_move(&self, from: (u8,u8), to: (u8,u8), game: &mut Game) -> Move {
        let (x, y) = (to.0 as i8 - from.0 as i8, to.1 as i8 - from.1 as i8);
        let sign = match self.player {
            Player::One => -1,
            Player::Two => 1
        };
        let end = match self.player {
            Player::One => 2,
            Player::Two => 5
        };
        let single = (x,y) == (0, sign*1) && game.square_is_none(to);
        let double = from.1==(end as i8-sign*4) as u8 && (x,y) == (0, sign*2) && game.check_horiz(from, (from.0, (from.1 as i8 + sign*3) as u8));
        if double {
            return Move::Double(to);
        }
        let diag = (x.abs() == 1) && (y == sign*1);
        if diag {
            if game.square_is_opponent(to) {
                return Move::Normal;
            } else if let Some(last_move) = game.get_last_double() {
                if to.0==last_move.0 && to.1==(end as i8) as u8 && game.square_is_opponent((to.0,(end as i8 - sign) as u8)) {
                    return Move::EnPassant((to.0,(end as i8 - sign) as u8));
                }
            } else {
                return Move::Invalid;
            }
        }
        if single {Move::Normal} else {Move::Invalid}
    }

    fn player(&self) -> Player {
        self.player
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "pawn"
    }

    fn value(&self) -> i32 {
        1
    }
}

impl Construct for Pawn {
    fn new(player: Player) -> Self {
        Pawn {player}
    }
}

impl DynClone for Pawn {
    fn clone_box(&self) -> Box<dyn Piece> {
        Box::new(self.clone())
    }
}

impl Display for Pawn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self.player() {
            Player::One => "♟".white().bold(),
            Player::Two => "♙".black()
        })
    }
}