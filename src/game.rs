use std::{io, fmt::{Display, Formatter, Error}, cmp::{min, max}};
use colored::Colorize;
use crate::piece::{Piece, Move};
use crate::king::King;
use crate::queen::Queen;
use crate::rook::Rook;
use crate::bishop::Bishop;
use crate::knight::Knight;
use crate::pawn::Pawn;
use crate::player::Player;

type Square = Option<Box<dyn Piece>>;
type Board = Vec<Vec<Square>>;

pub struct Game {
    board: Vec<Vec<Square>>,
    current_player: Player,
    game_over: bool,
    last_double: Option<(u8, u8)>,
    has_p1_king_moved: bool,
    has_p1_left_rook_moved: bool,
    has_p1_right_rook_moved: bool,
    has_p2_king_moved: bool,
    has_p2_left_rook_moved: bool,
    has_p2_right_rook_moved: bool
}

impl Game {
    pub fn new() -> Game {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0] = vec![Some(<dyn Piece>::new_piece::<Rook>(Player::Two)), Some(<dyn Piece>::new_piece::<Knight>(Player::Two)), Some(<dyn Piece>::new_piece::<Bishop>(Player::Two)), Some(<dyn Piece>::new_piece::<Queen>(Player::Two)), Some(<dyn Piece>::new_piece::<King>(Player::Two)), Some(<dyn Piece>::new_piece::<Bishop>(Player::Two)), Some(<dyn Piece>::new_piece::<Knight>(Player::Two)), Some(<dyn Piece>::new_piece::<Rook>(Player::Two))];
        board[1] = vec![Some(<dyn Piece>::new_piece::<Pawn>(Player::Two)); 8];
        
        board[7] = vec![Some(<dyn Piece>::new_piece::<Rook>(Player::One)), Some(<dyn Piece>::new_piece::<Knight>(Player::One)), Some(<dyn Piece>::new_piece::<Bishop>(Player::One)), Some(<dyn Piece>::new_piece::<Queen>(Player::One)), Some(<dyn Piece>::new_piece::<King>(Player::One)), Some(<dyn Piece>::new_piece::<Bishop>(Player::One)), Some(<dyn Piece>::new_piece::<Knight>(Player::One)), Some(<dyn Piece>::new_piece::<Rook>(Player::One))];
        board[6] = vec![Some(<dyn Piece>::new_piece::<Pawn>(Player::One)); 8];
        Game {
            board,
            current_player: Player::One,
            game_over: false,
            last_double: None,
            has_p1_king_moved: false,
            has_p1_left_rook_moved: false,
            has_p1_right_rook_moved: false,
            has_p2_king_moved: false,
            has_p2_left_rook_moved: false,
            has_p2_right_rook_moved: false,
        }
    }

    #[cfg(test)]
    fn set_board(&mut self, board: Board) {
        self.board = board;
    }

    #[cfg(test)]
    fn set_player(&mut self, player: Player) {
        self.current_player = player;
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
                let move_status = piece.clone().unwrap().valid_move(from, to, self);
                if move_status == Move::Invalid {
                    println!("Invalid move! go again.");
                    continue;
                }
                if let Some(conquered) = conquered.clone() {
                    if conquered.player() == self.current_player {
                        println!("You can't take your own piece! go again.");
                        continue;
                    } else {
                        self.take(to, piece.clone());
                        self.set(from, None);
                    }
                } else {
                    self.set(to, piece.clone());
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
                    self.set_moved(piece.clone(), from);
                }
                if (to.1 == 7 || to.1 == 0) && piece.unwrap().is_type::<Pawn>() {
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

    pub(crate) fn square_is_opponent(&self, to: (u8, u8)) -> bool {
        if let Some(piece) = self.get(to) {
            piece.player() != self.current_player
        } else {
            false
        }
    }

    pub(crate) fn take(&mut self,  to: (u8, u8), new_piece: Square) {
        let piece = self.get(to).unwrap();
        self.set(to, new_piece);
        if piece.name() == "king" {
            self.game_over = true;
        }
        println!("You took {}'s {}!", match self.current_player {
            Player::One => Player::Two,
            Player::Two => Player::Two,
        }, piece.name());
    }

    pub(crate) fn get(&self, (x, y): (u8, u8)) -> Square {
        self.board[y as usize][x as usize].clone()
    }

    fn set(&mut self, (x, y): (u8, u8), piece: Square) {
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
                    if piece.player() == player && piece.is_type::<King>() {
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
        let from = (from.0.unwrap() as i8 - 'a' as i8, '8' as i8 - from.1.unwrap() as i8);
        let to = (to.0.unwrap() as i8 - 'a' as i8, '8' as i8 - to.1.unwrap() as i8);
        if from.0 < 0 || from.1 < 0 || from.0 > 7 || from.1 > 7 || to.0 < 0 || to.1 < 0 || to.0 > 7 || to.1 > 7  {
            println!("Invalid input! Try again.");
            return self.get_move();
        }
        let from = (from.0 as u8, from.1 as u8);
        let to = (to.0 as u8, to.1 as u8);
        (from, to)
    }

    fn is_square_king(&self, square: (u8, u8)) -> bool {
        if let Some(piece) = self.get(square) {
            piece.is_type::<King>()
        } else {
            false
        }
    }

    fn checkmate(&mut self) -> bool {
        let mut king = self.get_king(self.current_player);
        if !self.in_check(king) {
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
                        if enemy.valid_move((j as u8,i as u8), king, self) != Move::Invalid {
                            //if it puts the king in check
                            for k in 0..8 {
                                for l in 0..8 {
                                    if let Some(friendly) = self.get((l,k)) {
                                        if friendly.player() == self.current_player {
                                            //for every friendly piece, see if it can block the path and get us out of check
                                            for square in friendly.can_intercept_path((l,k), (j,i), king, self) {
                                                let old = self.get(square);
                                                self.set(square, Some(friendly.clone()));
                                                self.set((l,k), None);
                                                if !self.is_square_king(king) {
                                                    king = self.get_king(self.current_player);
                                                }
                                                let still_in_check = self.in_check(king);
                                                self.set((l,k), Some(friendly.clone()));
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

        let (x,y) = king;
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
        println!("Pawn promotion! Enter a piece to promote to: (q, r, b, k)");
        let mut input = String::new();
        let piece: Box<dyn Piece>;
        loop {
            io::stdin().read_line(&mut input).unwrap();
            piece = match input.trim().to_ascii_lowercase().as_str() {
                "q" => <dyn Piece>::new_piece::<Queen>(self.current_player),
                "r" => <dyn Piece>::new_piece::<Rook>(self.current_player),
                "b" => <dyn Piece>::new_piece::<Bishop>(self.current_player),
                "k" => <dyn Piece>::new_piece::<Knight>(self.current_player),
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
                    if piece.is_type::<Rook>() && piece.player() == self.current_player {
                        return true;
                    }
                }
            }
        } else if new_x == 6 {
            if self.check_horiz(from, (7,y)) {
                if let Some(piece) = self.get((7,y)) {
                    if piece.is_type::<Rook>() && piece.player() == self.current_player {
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

    fn set_last_double(&mut self, position: Option<(u8, u8)>) {
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
            piece.is_type::<Knight>()
        } else {
            false
        }
    }

    pub(crate) fn is_not_ally(&self, new_pos: (u8, u8)) -> bool {
        !self.is_current_player(new_pos)
    }

    fn set_moved(&mut self, piece: Square, from: (u8, u8)) {
        let piece = piece.unwrap();
        if piece.is_type::<King>() {
            match piece.player() {
                Player::One => self.has_p1_king_moved = true,
                Player::Two => self.has_p2_king_moved = true,
            }
        } else if piece.is_type::<Rook>() {
            match piece.player() {
                Player::One => {
                    if from == (0,0) {
                        self.has_p1_left_rook_moved = true;
                    } else if from == (7,0) {
                        self.has_p1_right_rook_moved = true;
                    }
                },
                Player::Two => {
                    if from == (0,7) {
                        self.has_p2_left_rook_moved = true;
                    } else if from == (7,7) {
                        self.has_p2_right_rook_moved = true;
                    }
                }
            }
        }
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        writeln!(f, "{}", format!("\t\tPlayer 2").red().bold())?;
        writeln!(f, "    a    b    c    d    e    f    g    h")?;
        self.board.iter().enumerate().for_each(|(i,row)| {
            writeln!(f, "  -----------------------------------------").unwrap();
            write!(f, "{} ", 8-i).unwrap();
            row.iter().enumerate().for_each(|(j, piece)| {
                match piece {
                    Some(piece) => write!(f, "|{}", if (i+j)%2==0 {format!(" {piece}  ").on_black()} else {format!(" {piece}  ").on_bright_black()}),
                    None => write!(f,"|{}", if (i+j)%2==0 {format!("    ").on_black()} else {format!("    ").on_bright_black()}),
                    // Some(piece) => write!(f, "|{piece}"),
                    // None => write!(f,"|   "),
                }.unwrap();
            });
            write!(f, "|").unwrap();
            writeln!(f, " {}", 8-i).unwrap();
        });
        // writeln!(f, "  ---------------------------------")?;
        writeln!(f, "  -----------------------------------------")?;
        // writeln!(f, "    a   b   c   d   e   f   g   h")?;
        writeln!(f, "    a    b    c    d    e    f    g    h")?;
        // writeln!(f, "{}", format!("\t      Player 2").blue().bold())?;
        writeln!(f, "{}", format!("\t\tPlayer 1").blue().bold())?;
        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn checkmate_no_friendlies() {
//         let mut board = vec![vec![None;8];8];
//         board[0][0] = Some(Piece::King(Player::One));
//         board[0][1] = Some(Piece::Queen(Player::Two));
//         board[1][0] = Some(Piece::Queen(Player::Two));

//         let mut game = Game::test_game(board, Player::One);
//         assert!(game.checkmate());
//     }

//     #[test]
//     fn checkmate_no_friendlies2() {
//         let mut board = vec![vec![None;8];8];
//         board[0][0] = Some(Piece::King(Player::One));
//         board[0][1] = Some(Piece::Queen(Player::Two));
//         board[1][0] = Some(Piece::Rook(Player::Two));
//         board[0][2] = Some(Piece::Queen(Player::Two));

//         let mut game = Game::test_game(board, Player::One);
//         assert!(game.checkmate());
//     }

//     #[test]
//     fn no_checkmate_no_friendlies() {
//         let mut board = vec![vec![None;8];8];
//         board[0][0] = Some(Piece::King(Player::One));
//         board[0][1] = Some(Piece::Rook(Player::Two));
//         board[1][0] = Some(Piece::Rook(Player::Two));

//         let mut game = Game::test_game(board, Player::One);
//         assert!(!game.checkmate());
//     }

//     #[test]
//     fn no_checkmate_blockable() {
//         let mut board = vec![vec![None;8];8];
//         board[0][0] = Some(Piece::King(Player::One));
//         board[0][1] = Some(Piece::Knight(Player::One));
//         board[2][0] = Some(Piece::Queen(Player::Two));

//         let mut game = Game::test_game(board, Player::One);
//         assert!(!game.checkmate());
//     }

//     #[test]
//     fn checkmate_unblockable() {
//         let mut board = vec![vec![None;8];8];
//         board[0][0] = Some(Piece::King(Player::One));
//         board[0][1] = Some(Piece::Rook(Player::One));
//         board[1][1] = Some(Piece::Pawn(Player::One));
//         board[3][0] = Some(Piece::Queen(Player::Two));
        
//         let mut game = Game::test_game(board, Player::One);
//         assert!(game.checkmate());
//     }
// }