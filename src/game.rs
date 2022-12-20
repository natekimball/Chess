use std::{io, fmt::{Display, Formatter, Error}, cmp::{min, max}};
use colored::Colorize;
use crate::piece::Piece;

pub struct Game {
    board: Vec<Vec<Option<Piece>>>,
    current_player: Player,
    game_over: bool,
    last_move_double: bool
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
        board[0][0] = Some(Piece::Rook(Player::One));
        board[0][1] = Some(Piece::Knight(Player::One));
        board[0][2] = Some(Piece::Bishop(Player::One));
        board[0][3] = Some(Piece::Queen(Player::One));
        board[0][4] = Some(Piece::King(Player::One));
        board[0][5] = Some(Piece::Bishop(Player::One));
        board[0][6] = Some(Piece::Knight(Player::One));
        board[0][7] = Some(Piece::Rook(Player::One));
        for i in 0..8 {
            board[1][i] = Some(Piece::Pawn(Player::One));
        }
        board[7][0] = Some(Piece::Rook(Player::Two));
        board[7][1] = Some(Piece::Knight(Player::Two));
        board[7][2] = Some(Piece::Bishop(Player::Two));
        board[7][3] = Some(Piece::Queen(Player::Two));
        board[7][4] = Some(Piece::King(Player::Two));
        board[7][5] = Some(Piece::Bishop(Player::Two));
        board[7][6] = Some(Piece::Knight(Player::Two));
        board[7][7] = Some(Piece::Rook(Player::Two));
        for i in 0..8 {
            board[6][i] = Some(Piece::Pawn(Player::Two));
        }
        Game {
            board,
            current_player: Player::One,
            game_over: false,
            last_move_double: false
        }
    }

    pub fn turn(&mut self) {
        println!("{}", self);
        println!("It's {}'s turn.", self.current_player);
        if self.in_check(self.get_king()) {
            println!("You're in check!");
        }
        println!("Enter your move: (e.g. a2 a4)");

        let mut valid_move = false;
        while !valid_move {
            let (from, to) = get_move();
            if self.is_current_player(from) {
                // if self.board[from.1 as usize][from.0 as usize].as_ref().unwrap().is_valid(from, to, self) {
                let piece = self.get(from);
                let conquered = self.get(to);
                if !piece.unwrap().is_valid(from, to, self) {
                    println!("Invalid move! go again.");
                    continue;
                }
                if let Some(conquered) = conquered {
                    if conquered.player() == self.current_player {
                        println!("You can't take your own piece! go again.");
                        continue;
                    } else {
                        self.take(to, piece);
                        self.set(from, None);
                    }
                } else {
                    self.set(to, piece);
                    self.set(from, None);
                }
                if self.player_in_check() {
                    println!("You can't put yourself in check!");
                    self.set(from, self.get(to));
                    self.set(to, conquered);
                    continue;
                }
                valid_move = true;
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
            Player::One => Player::Two,
            Player::Two => Player::One
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
        let spot = self.get(from);
        if spot.is_none() {
            return false;
        }
        match spot.unwrap().player() {
            Player::One => matches!(self.current_player, Player::One),
            Player::Two => matches!(self.current_player, Player::Two),
        }
    }

    pub(crate) fn check_horiz(&self, from: (u8, u8), to: (u8, u8)) -> bool {
        if from.0 == to.0 {
            for i in min(from.1,to.1)..=max(from.1,to.1) {
                if self.get((from.0, i)).is_some() && i != from.1 && i != to.1 {
                    return false;
                }
            }
        } else {
            for i in min(from.0,to.0)..=max(from.0,to.0) {
                if self.get((i, from.1)).is_some() && i != from.0 && i != to.0 {
                    return false;
                }
            }
        }
        true
    }
    
    pub(crate) fn check_diag(&self, from: (u8, u8), delta: (i8, i8)) -> bool {
        let signs = (delta.0.signum(), delta.1.signum());
        for i in 1..delta.0.abs() as u8 {
            if self.get(((from.0 as i8 + signs.0 * i as i8) as u8, (from.1 as i8 + signs.1 * i as i8) as u8)).is_some() {
                return false;
            }
        }
        true
    }

    pub(crate) fn check_spot_for_opponent(&self, to: (u8, u8)) -> bool {
        if let Some(piece) = self.get(to) {
            piece.player() != self.current_player
        } else {
            false
        }
    }

    pub(crate) fn take(&mut self,  to: (u8, u8), new_piece: Option<Piece>) {
        let piece = self.get(to).unwrap();
        self.set(to, new_piece);
        println!("You took {}'s {}!", match self.current_player {
            Player::One => Player::Two,
            Player::Two => Player::Two,
        }, match piece {
            Piece::Pawn(_) => "pawn",
            Piece::Rook(_) => "rook",
            Piece::Knight(_) => "knight",
            Piece::Bishop(_) => "bishop",
            Piece::Queen(_) => "queen",
            Piece::King(_) => {
                self.game_over = true;
                "king"
            }
        });
    }

    pub(crate) fn can_en_passant(&self) -> bool {
        self.last_move_double
    }

    pub(crate) fn set_can_enpassant(&mut self, last_move_double: bool) {
        self.last_move_double = last_move_double;
    }

    pub(crate) fn get(&self, (x, y): (u8, u8)) -> Option<Piece> {
        self.board[y as usize][x as usize]
    }

    pub(crate) fn set(&mut self, (x, y): (u8, u8), piece: Option<Piece>) {
        self.board[y as usize][x as usize] = piece;
    }

    pub(crate) fn in_check(&mut self, king: (u8, u8)) -> bool {
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.player() != self.current_player {
                        if piece.is_valid((i as u8,j as u8), king, self) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn get_king(&self) -> (u8, u8) {
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.get((j,i)) {
                    if piece.player() == self.current_player && piece.is_king() {
                        return (j as u8, i as u8);
                    }
                }
            }
        }
        panic!("No king found!");
    }

    fn player_in_check(&mut self) -> bool {
        self.in_check(self.get_king())
    }
}

fn get_move() -> ((u8, u8), (u8, u8)) {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let mut input = input.split_whitespace();
    let from = input.next().unwrap().to_ascii_lowercase();
    let to = input.next().unwrap().to_ascii_lowercase();
    let from = (from.chars().nth(0).unwrap() as u8 - 'a' as u8, from.chars().nth(1).unwrap() as u8 - '1' as u8);
    let to = (to.chars().nth(0).unwrap() as u8 - 'a' as u8, to.chars().nth(1).unwrap() as u8 - '1' as u8);
    (from, to)
}

// TODO: make set method, use get method
// TODO: implement check and checkmate

impl Display for Game {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        writeln!(f, "{}", format!("\t      Player 1").red().bold())?;
        writeln!(f, "    a   b   c   d   e   f   g   h")?;
        self.board.iter().enumerate().for_each(|(i,row)| {
            writeln!(f, "  ---------------------------------").unwrap();
            write!(f, "{} ", i+1).unwrap();
            row.iter().enumerate().for_each(|(j, piece)| {
                match piece {
                    Some(piece) => write!(f, "|{}", if (i+j)%2==0 {format!("{piece}").on_black()} else {format!("{piece}").on_bright_black()}),
                    None => write!(f,"|{}", if (i+j)%2==0 {format!("   ").on_black()} else {format!("   ").on_bright_black()}),
                    // Some(piece) => write!(f, "|{piece}"),
                    // None => write!(f,"|   "),
                }.unwrap();
            });
            write!(f, "|").unwrap();
            writeln!(f, " {}", i+1).unwrap();
        });
        writeln!(f, "  ---------------------------------")?;
        writeln!(f, "    a   b   c   d   e   f   g   h")?;
        writeln!(f, "{}", format!("\t      Player 2").blue().bold())?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Player {
    One,
    Two
}

impl Display for Player {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Player::One => write!(f, "player 1"),
            Player::Two => write!(f, "player 2"),
        }
    }
}