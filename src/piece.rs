use std::fmt::{Display, Formatter, Error};
use colored::{Colorize};
use crate::game::{Game, Player};

#[derive(Debug, Copy, Clone)]
pub enum Piece {
    King(Player),
    Queen(Player),
    Rook(Player),
    Bishop(Player),
    Knight(Player),
    Pawn(Player)
}

impl Piece {
    pub fn is_valid(&self, from: (u8,u8), to: (u8, u8), game: &mut Game) -> bool {
        if from == to {
            return false;
        }
        let last_move = game.can_en_passant();
        game.set_can_enpassant(false);

        // clone board, try turn, check for check, then undo turn?

        let (x, y) = (to.0 as i8 - from.0 as i8, to.1 as i8 - from.1 as i8);
        match self {
            Piece::Queen(_) => ((x == 0 || y == 0) && game.check_horiz(from, to)) || ((x.abs() == y.abs()) && game.check_diag(from, (x,y))),
            Piece::King(_) => (x.abs() < 2) && (y.abs() < 2) && !game.in_check(to),
            Piece::Bishop(_) => (x.abs() == y.abs()) && game.check_diag(from, (x,y)),
            Piece::Rook(_) => (x == 0 || y == 0) && game.check_horiz(from, to),
            Piece::Knight(_) => (x.abs() == 2 && y.abs() == 1) || (x.abs() == 1 && y.abs() == 2),
            Piece::Pawn(player) => {
                let sign = match player {
                    Player::One => 1,
                    Player::Two => -1
                };
                let end = match player {
                    Player::One => 5,
                    Player::Two => 2
                };
                let single = (x,y) == (0, sign*1) && game.check_horiz(from, (from.0, (from.1 as i8 + sign*2) as u8));
                let double = from.1==(end as i8-sign*4) as u8 && (x,y) == (0, sign*2) && game.check_horiz(from, (from.0, (from.1 as i8 + sign*3) as u8));
                if double {
                    game.set_can_enpassant(true);
                    return true;
                }
                let diag = (x.abs() == 1) && (y == sign*1);
                if diag {
                    if game.check_spot_for_opponent(to) {
                        return true;
                    } else if last_move && to.1==(end as i8) as u8 && game.check_spot_for_opponent((to.0,(end as i8 - sign) as u8)) {
                        game.take((to.0,(end as i8 - sign) as u8), None);
                        return true;
                    } else {
                        return false;
                    }
                }
                single
            }
        }
    }

    pub fn player(&self) -> Player {
        match self {
            Piece::King(player) => *player,
            Piece::Queen(player) => *player,
            Piece::Rook(player) => *player,
            Piece::Bishop(player) => *player,
            Piece::Knight(player) => *player,
            Piece::Pawn(player) => *player
        }
    }

    pub(crate) fn is_king(&self) -> bool {
        matches!(self, Piece::King(_))
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Piece::King(player) => write!(f, " {} ", if *player == Player::One { format!("K").red().bold() } else { format!("K").blue().bold() }),
            Piece::Queen(player) => write!(f, " {} ", if *player == Player::One { format!("Q").red().bold() } else { format!("Q").blue().bold() }),
            Piece::Rook(player) => write!(f, " {} ", if *player == Player::One { format!("R").red().bold() } else { format!("R").blue().bold() }),
            Piece::Bishop(player) => write!(f, " {} ", if *player == Player::One { format!("B").red().bold() } else { format!("B").blue().bold() }),
            Piece::Knight(player) => write!(f, " {} ", if *player == Player::One { format!("N").red().bold() } else { format!("N").blue().bold() }),
            Piece::Pawn(player) => write!(f, " {} ", if *player == Player::One { format!("P").red().bold() } else { format!("P").blue().bold() }),
        }
    }
}