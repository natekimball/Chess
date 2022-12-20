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
        let (x, y) = (to.0 as i8 - from.0 as i8, to.1 as i8 - from.1 as i8);
        match self {
            Piece::Queen(_) => ((x == 0 || y == 0) && game.check_horiz(from, to)) || ((x.abs() == y.abs()) && game.check_diag(from, (x,y))),
            Piece::King(_) => (x.abs() < 2) && (y.abs() < 2),
            Piece::Bishop(_) => (x.abs() == y.abs()) && game.check_diag(from, (x,y)),
            Piece::Rook(_) => (x == 0 || y == 0) && game.check_horiz(from, to),
            Piece::Knight(_) => (x.abs() == 2 && y.abs() == 1) || (x.abs() == 1 && y.abs() == 2),
            Piece::Pawn(player) => {
                match player {
                    Player::Player1 => {
                        let forwards = if from.1==1 { ((x,y) == (0, 1) && game.check_horiz(from, (from.0,from.1+2))) || ((x,y) == (0,2) && game.check_horiz(from, (from.0,from.1+3))) } else { (x,y) == (0, 1) };
                        let diag = (x.abs() == 1) && (y == 1);
                        if diag {
                            if game.check_spot_for_opponent(to) {
                                return true;
                            } else if to.1==5 && game.check_spot_for_opponent((to.0,4)) {
                                game.take(to);
                                return true;
                            }
                            return false;
                        }
                        forwards
                    },
                    Player::Player2 => {
                        let forwards = if from.1==6 { ((x,y) == (0, -1) && game.check_horiz(from, (from.0,from.1-2))) || ((x,y) == (0,-2) && game.check_horiz(from,(from.0,from.1-3))) } else { (x,y) == (0, -1) };
                        forwards || ((x.abs() == 1) && (y == 1) && (game.check_spot_for_opponent(to) || (to.1==2 && game.check_spot_for_opponent((to.0,3)))))
                    }
                }
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
            Piece::Pawn(player) => *player,
        }
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            // Piece::King(player) => write!(f, " K{} ", player.num()),
            Piece::King(player) => write!(f, " {} ", if player.num() == 1 { format!("K").red().bold() } else { format!("K").blue().bold() }),
            Piece::Queen(player) => write!(f, " {} ", if player.num() == 1 { format!("Q").red().bold() } else { format!("Q").blue().bold() }),
            Piece::Rook(player) => write!(f, " {} ", if player.num() == 1 { format!("R").red().bold() } else { format!("R").blue().bold() }),
            Piece::Bishop(player) => write!(f, " {} ", if player.num() == 1 { format!("B").red().bold() } else { format!("B").blue().bold() }),
            Piece::Knight(player) => write!(f, " {} ", if player.num() == 1 { format!("N").red().bold() } else { format!("N").blue().bold() }),
            Piece::Pawn(player) => write!(f, " {} ", if player.num() == 1 { format!("P").red().bold() } else { format!("P").blue().bold() }),
        }
    }
}