use std::{io, fmt::{Display, Formatter, Error}, cmp::{min, max}};

use crate::piece::Piece;

pub struct Game {
    board: Vec<Vec<Option<Piece>>>,
    current_player: Player,
    game_over: bool
}
impl Game {
    pub fn new() -> Game {
        let mut board = Vec::new();
        for _ in 0..8 {
            let mut row = Vec::new();
            for _ in 0..8 {
                row.push(None);
            }
            board.push(row);
        }
        board[0][0] = Some(Piece::Rook(Player::Player1));
        board[0][1] = Some(Piece::Knight(Player::Player1));
        board[0][2] = Some(Piece::Bishop(Player::Player1));
        board[0][3] = Some(Piece::Queen(Player::Player1));
        board[0][4] = Some(Piece::King(Player::Player1));
        board[0][5] = Some(Piece::Bishop(Player::Player1));
        board[0][6] = Some(Piece::Knight(Player::Player1));
        board[0][7] = Some(Piece::Rook(Player::Player1));
        for i in 0..8 {
            board[1][i] = Some(Piece::Pawn(Player::Player1));
        }
        board[7][0] = Some(Piece::Rook(Player::Player2));
        board[7][1] = Some(Piece::Knight(Player::Player2));
        board[7][2] = Some(Piece::Bishop(Player::Player2));
        board[7][3] = Some(Piece::Queen(Player::Player2));
        board[7][4] = Some(Piece::King(Player::Player2));
        board[7][5] = Some(Piece::Bishop(Player::Player2));
        board[7][6] = Some(Piece::Knight(Player::Player2));
        board[7][7] = Some(Piece::Rook(Player::Player2));
        for i in 0..8 {
            board[6][i] = Some(Piece::Pawn(Player::Player2));
        }
        Game {
            board,
            current_player: Player::Player1,
            game_over: false
        }
    }

    pub fn turn(&mut self) {
        println!("{}", self);
        println!("It's {}'s turn.", self.current_player);
        println!("Enter your move: (e.g. a2 a4)");
        let mut valid_move = false;
        while !valid_move {
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let mut input = input.split_whitespace();
            let from = input.next().unwrap().to_ascii_lowercase();
            let to = input.next().unwrap().to_ascii_lowercase();
            let from = (from.chars().nth(0).unwrap() as u8 - 'a' as u8, from.chars().nth(1).unwrap() as u8 - '1' as u8);
            let to = (to.chars().nth(0).unwrap() as u8 - 'a' as u8, to.chars().nth(1).unwrap() as u8 - '1' as u8);

            if self.is_current_player(from) {
                // if self.board[from.1 as usize][from.0 as usize].as_ref().unwrap().is_valid(from, to, self) {
                if let Some(piece) = self.board[from.1 as usize][from.0 as usize] { 
                    if !piece.is_valid(from, to, self) {
                        println!("Invalid move! go again.");
                        continue;
                    }
                    if let Some(conquered) = self.board[to.1 as usize][to.0 as usize] {
                        if conquered.player() == self.current_player {
                            println!("You can't take your own piece!");
                            continue;
                        } else {
                            println!("You took {}'s {}!", conquered.player(), match conquered {
                                Piece::Pawn(_) => "pawn",
                                Piece::Rook(_) => "rook",
                                Piece::Knight(_) => "knight",
                                Piece::Bishop(_) => "bishop",
                                Piece::Queen(_) => "queen",
                                Piece::King(_) => {
                                    self.game_over = true;
                                    "king"
                                },
                            });
                        }
                    }
                    self.board[to.1 as usize][to.0 as usize] = self.board[from.1 as usize][from.0 as usize];
                    self.board[from.1 as usize][from.0 as usize] = None;
                    valid_move = true;
                }
            } else {
                println!("You must move one of your own pieces!");
            }
            if !valid_move {
                println!("Invalid move! go again.");
            }
        }
        if self.game_over {
            println!("{self}");
            println!("{} wins!", self.current_player);
        }
        self.current_player = match self.current_player {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1,
        }
    }

    pub fn is_over(&self) -> bool {
        self.game_over
    }

    pub fn play_again(&self) -> bool {
        println!("Game over!");
        println!("{} wins!", self.current_player);
        println!("Play again? (y/n)");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input.trim().to_ascii_lowercase() == "y"
    }

    fn is_current_player(&self, from: (u8, u8)) -> bool {
        let spot = self.board[from.1 as usize][from.0 as usize].as_ref();
        if spot.is_none() {
            return false;
        }
        match spot.unwrap().player() {
            Player::Player1 => matches!(self.current_player, Player::Player1),
            Player::Player2 => matches!(self.current_player, Player::Player2),
        }
    }

    pub(crate) fn check_horiz(&self, from: (u8, u8), to: (u8, u8)) -> bool {
        if from.0 == to.0 {
            for i in min(from.1,to.1)..=max(from.1,to.1) {
                if self.board[i as usize][from.0 as usize].is_some() && i != from.1 && i != to.1 {
                    return false;
                }
            }
        } else {
            for i in min(from.0,to.0)..=max(from.0,to.0) {
                if self.board[from.1 as usize][i as usize].is_some() && i != from.0 && i != to.0 {
                    return false;
                }
            }
        }
        true
    }
    
    pub(crate) fn check_diag(&self, from: (u8, u8), delta: (i8, i8)) -> bool {
        let signs = (delta.0.signum(), delta.1.signum());
        for i in 1..delta.0.abs() as u8 {
            if self.board[(from.1 as i8 + signs.1 * i as i8) as usize][(from.0 as i8 + signs.0 * i as i8) as usize].is_some() {
                return false;
            }
        }
        true
    }

    pub(crate) fn check_spot_for_opponent(&self, to: (u8, u8)) -> bool {
        !self.is_current_player(to)
    }

    pub(crate) fn take(&mut self, to: (u8, u8)) {
        self.board[to.1 as usize][to.0 as usize] = None;
    }    
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        writeln!(f, "\t\tPlayer 1")?;
        writeln!(f, "     a    b    c    d    e    f    g    h")?;
        self.board.iter().enumerate().for_each(|(i,row)| {
            writeln!(f, "  -----------------------------------------").unwrap();
            write!(f, "{} ", i+1).unwrap();
            row.iter().for_each(|piece| {
                match piece {
                    Some(piece) => write!(f, "|{}", piece),
                    None => write!(f,"|    "),
                }.unwrap();
            });
            write!(f, "|").unwrap();
            writeln!(f, " {}", i+1).unwrap();
        });
        writeln!(f, "  -----------------------------------------")?;
        writeln!(f, "     a    b    c    d    e    f    g    h")?;
        write!(f, "\t\tPlayer 2")?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Player {
    Player1,
    Player2
}
impl Player {
    pub fn num(&self) -> u8 {
        match self {
            Player::Player1 => 1,
            Player::Player2 => 2,
        }
    }
}

impl Display for Player {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Player::Player1 => write!(f, "player 1"),
            Player::Player2 => write!(f, "player 2"),
        }
    }
}