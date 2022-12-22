use std::{io, fmt::{Display, Formatter, Error}, cmp::{min, max}};
use colored::Colorize;
use crate::piece::{Piece, Move};

pub struct Game {
    board: Vec<Vec<Option<Piece>>>,
    current_player: Player,
    game_over: bool,
    last_double: Option<(u8, u8)>
}

impl Game {
    pub fn new() -> Game {
        let mut board = vec![vec![None; 8]; 8];
        board[0] = vec![Some(Piece::Rook(Player::One)), Some(Piece::Knight(Player::One)), Some(Piece::Bishop(Player::One)), Some(Piece::Queen(Player::One)), Some(Piece::King(Player::One)), Some(Piece::Bishop(Player::One)), Some(Piece::Knight(Player::One)), Some(Piece::Rook(Player::One))];
        board[1] = vec![Some(Piece::Pawn(Player::One)); 8];
        
        board[7] = vec![Some(Piece::Rook(Player::Two)), Some(Piece::Knight(Player::Two)), Some(Piece::Bishop(Player::Two)), Some(Piece::Queen(Player::Two)), Some(Piece::King(Player::Two)), Some(Piece::Bishop(Player::Two)), Some(Piece::Knight(Player::Two)), Some(Piece::Rook(Player::Two))];
        board[6] = vec![Some(Piece::Pawn(Player::Two)); 8];
        Game {
            board,
            current_player: Player::One,
            game_over: false,
            last_double: None
        }
    }

    fn test_game(board: Vec<Vec<Option<Piece>>>, player: Player) -> Game {
        Game {
            board,
            current_player: player,
            game_over: false,
            last_double: None
        }
    }

    pub fn turn(&mut self) {
        println!("{}", self);
        println!("It's {}'s turn.", self.current_player);
        if self.player_in_check() {
            println!("You're in check!");
        }
        println!("Enter your move: (e.g. a2 a4)");

        let mut valid_move = false;
        while !valid_move {
            let (from, to) = self.get_move();
            if self.is_current_player(from) {
                let piece = self.get(from);
                let conquered = self.get(to);
                let move_status = piece.unwrap().valid_move(from, to, self);
                if move_status == Move::Invalid {
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
                    println!("Wait you can't put yourself in check! go again.");
                    self.set(from, self.get(to));
                    self.set(to, conquered);
                    continue;
                } else {
                    self.set_last_double(None);
                    match move_status {
                        Move::Normal => (),
                        Move::Double(position) => {
                            self.set_last_double(Some(position));
                        },
                        Move::Castle => {
                            self.castle(to);
                        },
                        Move::EnPassant(position) => {
                            self.take(position, None);
                        },
                        Move::Invalid => unreachable!()
                    }
                    
                }
                if (to.1 == 7 || to.1 == 0) && piece.unwrap() == Piece::Pawn(self.current_player) {
                    self.promote_piece(to);
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

    pub fn is_over(&mut self) -> bool {
        self.game_over = self.game_over || self.checkmate();
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

    pub(crate) fn get(&self, (x, y): (u8, u8)) -> Option<Piece> {
        self.board[y as usize][x as usize]
    }

    pub(crate) fn set(&mut self, (x, y): (u8, u8), piece: Option<Piece>) {
        self.board[y as usize][x as usize] = piece;
    }

    pub(crate) fn in_check(&self, king: (u8, u8)) -> bool {
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.get((j,i)) {
                    if piece.player() != self.current_player {
                        if piece.valid_move((j as u8,i as u8), king, self) != Move::Invalid {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn get_king(&self, player: Player) -> (u8, u8) {
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.get((j,i)) {
                    if piece.player() == player && piece.is_king() {
                        return (j as u8, i as u8);
                    }
                }
            }
        }
        panic!("No king found!");
    }

    fn player_in_check(&mut self) -> bool {
        self.in_check(self.get_king(self.current_player))
    }

    fn get_move(&self) -> ((u8, u8), (u8, u8)) {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let mut input = input.split_whitespace();
        let from = input.next();
        let to = input.next();
        if from.is_none() || to.is_none() {
            println!("Invalid input! Try again.");
            return self.get_move();
        }
        let from = from.unwrap().to_ascii_lowercase();
        let to = to.unwrap().to_ascii_lowercase();
        let from = (from.chars().nth(0), from.chars().nth(1));
        let to = (to.chars().nth(0), to.chars().nth(1));
        if from.0.is_none() || from.1.is_none() || to.0.is_none() || to.1.is_none() {
            println!("Invalid input! Try again.");
            return self.get_move();
        }
        let from = (from.0.unwrap() as i8 - 'a' as i8, from.1.unwrap() as i8 - '1' as i8);
        let to = (to.0.unwrap() as i8 - 'a' as i8, to.1.unwrap() as i8 - '1' as i8);
        if from.0 < 0 || from.1 < 0 || from.0 > 7 || from.1 > 7 || to.0 < 0 || to.1 < 0 || to.0 > 7 || to.1 > 7  {
            println!("Invalid input! Try again.");
            return self.get_move();
        }
        let from = (from.0 as u8, from.1 as u8);
        let to = (to.0 as u8, to.1 as u8);
        (from, to)
    }

    // fn checkmate(&mut self) -> bool {
    //     //do you need to check both or just current player/opponent?
    //     let p1 = self.player_checkmate(Player::One);
    //     let p2 = self.player_checkmate(Player::Two);
    //     p1||p2
    // }

    fn checkmate(&mut self) -> bool {
        let (x,y) = self.get_king(self.current_player);
        if !self.in_check((x,y)) {
            return false;
        }
        // for every enemy that can attack king {
        //     for every square in path from enemy to king {
        //         for every friendly piece {
        //             if friendly piece can move to square {
        //                 return false;
        //             }
        //         }
        //     }
        // }
        for i in 0..8 {
            for j in 0..8 {
                if let Some(enemy) = self.get((j,i)) {
                    if enemy.player() != self.current_player {
                        //for every enemy piece
                        if enemy.valid_move((j as u8,i as u8), (x,y), self) != Move::Invalid {
                            //if it puts the king in check
                            for k in 0..8 {
                                for l in 0..8 {
                                    if let Some(friendly) = self.get((l,k)) {
                                        if friendly.player() == self.current_player {
                                            //for every friendly piece, see if it can block the path and get us out of check
                                            for square in friendly.can_block_path((l,k), (j,i), (x,y), self) {
                                                let old = self.get(square);
                                                self.set(square, Some(friendly));
                                                self.set((l,k), None);
                                                let still_in_check = self.in_check((x,y));
                                                self.set((l,k), Some(friendly));
                                                self.set(square, old);
                                                if !still_in_check {
                                                    return false;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (i, j) in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)] {
            let (new_x, new_y) = (x as i8 + i, y as i8 + j);
            if new_x < 0 || new_x >= 8 || new_y < 0 || new_y >= 8 {
                continue;
            }
            let (new_x, new_y) = (new_x as u8, new_y as u8);
            if !self.is_current_player((new_x,new_y)) && !self.in_check((new_x,new_y)) {
                return false;
            }
        }
        true
    }

    fn promote_piece(&mut self, position: (u8, u8)) {
        println!("Pawn promotion! Enter a piece to promote to: (q, r, b, n)");
        let mut input = String::new();
        let piece;
        loop {
            io::stdin().read_line(&mut input).unwrap();
            piece = match input.trim().to_ascii_lowercase().as_str() {
                "q" => Piece::Queen(self.current_player),
                "r" => Piece::Rook(self.current_player),
                "b" => Piece::Bishop(self.current_player),
                "n" => Piece::Knight(self.current_player),
                _ => {
                    println!("Invalid piece! Enter another.");
                    continue;
                }
            };
            break;
        }
        self.set(position, Some(piece));
    }

    pub(crate) fn can_castle(&self, from: (u8, u8), to: (u8, u8)) -> bool {
        if self.in_check(from) {
            return false;
        }
        let (x,y) = from;
        let (new_x, new_y) = to;
        if y != new_y || x != 4 {
            return false;
        }
        if new_x == 2 {
            if self.check_horiz(from, (0,y)) {
                if let Some(piece) = self.get((0,y)) {
                    if piece.is_rook() && piece.player() == self.current_player {
                        // self.set((3,y), Some(Piece::Rook(self.current_player)));
                        // self.set((0,y), None);
                        return true;
                    }
                }
            }
        } else if new_x == 6 {
            if self.check_horiz(from, (7,y)) {
                if let Some(piece) = self.get((7,y)) {
                    if piece.is_rook() && piece.player() == self.current_player {
                        // self.set((3,y), Some(Piece::Rook(self.current_player)));
                        // self.set((0,y), None);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn castle(&mut self, king: (u8, u8)) {
        let (rook_from, rook_to) = match king.0 {
            6 => ((7, king.1), (5, king.1)),
            2 => ((0, king.1), (3, king.1)),
            _ => unreachable!()
        };
        self.set(rook_to, self.get(rook_from));
        self.set(rook_from, None);
    }

    pub(crate) fn set_last_double(&mut self, position: Option<(u8, u8)>) {
        self.last_double = position;
    }

    pub(crate) fn get_last_double(&self) -> Option<(u8, u8)> {
        self.last_double
    }

    pub(crate) fn square_is_none(&self, to: (u8, u8)) -> bool {
        self.get(to).is_none()
    }

    pub(crate) fn square_is_knight(&self, to: (u8, u8)) -> bool {
        if let Some(piece) = self.get(to) {
            piece.is_knight()
        } else {
            false
        }
    }
}

// TODO: implement checkmate

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkmate_no_friendlies() {
        let mut board = vec![vec![None;8];8];
        board[0][0] = Some(Piece::King(Player::One));
        board[0][1] = Some(Piece::Queen(Player::Two));
        board[1][0] = Some(Piece::Queen(Player::Two));

        let mut game = Game::test_game(board, Player::One);
        assert!(game.checkmate());
    }

    #[test]
    fn checkmate_no_friendlies2() {
        let mut board = vec![vec![None;8];8];
        board[0][0] = Some(Piece::King(Player::One));
        board[0][1] = Some(Piece::Queen(Player::Two));
        board[1][0] = Some(Piece::Rook(Player::Two));
        board[0][2] = Some(Piece::Queen(Player::Two));

        let mut game = Game::test_game(board, Player::One);
        assert!(game.checkmate());
    }

    #[test]
    fn no_checkmate_no_friendlies() {
        let mut board = vec![vec![None;8];8];
        board[0][0] = Some(Piece::King(Player::One));
        board[0][1] = Some(Piece::Rook(Player::Two));
        board[1][0] = Some(Piece::Rook(Player::Two));

        let mut game = Game::test_game(board, Player::One);
        assert!(!game.checkmate());
    }

    #[test]
    fn no_checkmate_blockable() {
        let mut board = vec![vec![None;8];8];
        board[0][0] = Some(Piece::King(Player::One));
        board[0][1] = Some(Piece::Knight(Player::One));
        board[2][0] = Some(Piece::Queen(Player::Two));

        let mut game = Game::test_game(board, Player::One);
        assert!(!game.checkmate());
    }

    #[test]
    fn checkmate_unblockable() {
        let mut board = vec![vec![None;8];8];
        board[0][0] = Some(Piece::King(Player::One));
        board[0][1] = Some(Piece::Rook(Player::One));
        board[2][0] = Some(Piece::Queen(Player::Two));
        
        let mut game = Game::test_game(board, Player::One);
        print!("{game}");
        assert!(game.checkmate());
    }
}