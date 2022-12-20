use std::{io, fmt::{Display, Formatter, Error}};

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

            if self.board[from.1 as usize][from.0 as usize].is_some() && self.is_current_player(from) {
                if self.board[from.1 as usize][from.0 as usize].as_ref().unwrap().is_valid(from, to) {
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
                println!("You can't move that piece!");
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
        match self.board[from.1 as usize][from.0 as usize].as_ref().unwrap().player() {
            Player::Player1 => matches!(self.current_player, Player::Player1),
            Player::Player2 => matches!(self.current_player, Player::Player2),
        }
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        writeln!(f, "\t    Player 1")?;
        self.board.iter().enumerate().for_each(|(i,row)| {
            writeln!(f, "  ---------------------------------").unwrap();
            write!(f, "{i} ").unwrap();
            row.iter().for_each(|piece| {
                match piece {
                    Some(piece) => write!(f, "|{}", format!("{}", piece)),
                    None => write!(f,"|   "),
                }.unwrap();
            });
            writeln!(f, "|").unwrap();
        });
        writeln!(f, "  ---------------------------------")?;
        writeln!(f, "    a   b   c   d   e   f   g   h")?;
        write!(f, "\t    Player 2")?;
        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
enum Piece {
    King(Player),
    Queen(Player),
    Rook(Player),
    Bishop(Player),
    Knight(Player),
    Pawn(Player)
}
impl Piece {
    fn is_valid(&self, from: (u8,u8), to: (u8, u8)) -> bool {
        let (x, y) = (to.0 as i8 - from.0 as i8, to.1 as i8 - from.1 as i8);
        match self {
            Piece::Queen(_) => true,
            Piece::King(_) => (x.abs() < 2) && (y.abs() < 2),
            Piece::Bishop(_) => x.abs() == y.abs(),
            Piece::Rook(_) => x == 0 || y == 0,
            Piece::Knight(_) => (x.abs() == 2 && y.abs() == 1) || (x.abs() == 1 && y.abs() == 2),
            Piece::Pawn(player) => {
                match player {
                    Player::Player1 => {
                        if from.1==1 { (x,y) == (0, 1) || (x,y) == (0,2) } else { (x,y) == (0, 1) }
                    },
                    Player::Player2 => {
                        if from.1==1 { (x,y) == (0, -1) || (x,y) == (0,-2) } else { (x,y) == (0, -1) }
                    }
                }
            }
        }
    }

    fn player(&self) -> Player {
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
            Piece::King(_) => write!(f, " {} ", format!("K")),
            Piece::Queen(_) => write!(f, " {} ", format!("Q")),
            Piece::Rook(_) => write!(f, " {} ", format!("R")),
            Piece::Bishop(_) => write!(f, " {} ", format!("B")),
            Piece::Knight(_) => write!(f, " {} ", format!("N")),
            Piece::Pawn(_) => write!(f, " {} ", format!("P")),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Player {
    Player1,
    Player2
}

impl Display for Player {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Player::Player1 => write!(f, "player 1"),
            Player::Player2 => write!(f, "player 2"),
        }
    }
}