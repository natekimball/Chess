use crate::{
    bishop::Bishop,
    king::King,
    knight::Knight,
    pawn::Pawn,
    piece::{Construct, Move, Piece},
    player::Player,
    queen::Queen,
    rook::Rook,
    model::Model,
};
use colored::Colorize;
use std::{
    cmp::{max, min},
    fmt::{Display, Error, Formatter},
    io,
};

pub type Square = Option<Box<dyn Piece>>;
pub type Board = Vec<Vec<Square>>;

#[derive(Clone)]
pub struct Game {
    board: Vec<Vec<Square>>,
    current_player: Player,
    game_over: bool,
    last_double: Option<(u8, u8)>,
    king_one: (u8, u8),
    king_two: (u8, u8),
    has_p1_king_moved: bool,
    has_p1_left_rook_moved: bool,
    has_p1_right_rook_moved: bool,
    has_p2_king_moved: bool,
    has_p2_left_rook_moved: bool,
    has_p2_right_rook_moved: bool,
    p1_pieces: Vec<(u8, u8)>,
    p2_pieces: Vec<(u8, u8)>,
    p1_taken: [u8; 5],
    p2_taken: [u8; 5],
    half_move_clock: u8,
    full_move_clock: u8,
    model: Option<Model>,
}

impl Game {
    pub fn new() -> Game {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0] = vec![Some(Box::new(Rook::new(Player::Two))), Some(Box::new(Knight::new(Player::Two))), Some(Box::new(Bishop::new(Player::Two))), Some(Box::new(Queen::new(Player::Two))), Some(Box::new(King::new(Player::Two))), Some(Box::new(Bishop::new(Player::Two))), Some(Box::new(Knight::new(Player::Two))), Some(Box::new(Rook::new(Player::Two)))];
        board[1] = vec![Some(Box::new(Pawn::new(Player::Two))); 8];
        board[7] = vec![Some(Box::new(Rook::new(Player::One))), Some(Box::new(Knight::new(Player::One))), Some(Box::new(Bishop::new(Player::One))), Some(Box::new(Queen::new(Player::One))), Some(Box::new(King::new(Player::One))), Some(Box::new(Bishop::new(Player::One))), Some(Box::new(Knight::new(Player::One))), Some(Box::new(Rook::new(Player::One)))];
        board[6] = vec![Some(Box::new(Pawn::new(Player::One))); 8];

        let args: Vec<String> = std::env::args().collect();
        let two_player = args.contains(&String::from("--2p"));
        let model = if two_player { None } else { Some(Model::new()) };

        Game {
            board,
            current_player: Player::One,
            game_over: false,
            last_double: None,
            king_one: (4, 7),
            king_two: (4, 0),
            has_p1_king_moved: false,
            has_p1_left_rook_moved: false,
            has_p1_right_rook_moved: false,
            has_p2_king_moved: false,
            has_p2_left_rook_moved: false,
            has_p2_right_rook_moved: false,
            p1_pieces: vec![(0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6)],
            p2_pieces: vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
            p1_taken: [0; 5],
            p2_taken: [0; 5],
            half_move_clock: 0,
            full_move_clock: 1,
            model
        }
    }

    fn get_best_move(&mut self) -> ((u8, u8), (u8, u8)) {
        let possible_moves = self.get_possible_moves(self.current_player);
        if possible_moves.is_empty() {
            print!("No possible moves for player {}!", self.current_player);
            self.game_over = true;
            panic!("Stalemate!");
        }
        /*
        possible_moves.iter().sort(|(from, to)| {
            let mut game = self.clone();
            game.move_piece(*from, *to);
            game.evaluate()
        });

        possible_moves.reverse();
        possible_moves.iter().take(10).collect();
         */
        let mut best_move = (possible_moves[0].0, possible_moves[0].1[0]);
        let mut best_score = f32::MIN;
        for (from, moves) in possible_moves {
            for to in moves {
                let mut game = self.clone();
                game.move_piece(from, to);
                let score = game.minimax(3, false, f32::MIN, f32::MAX);
                if score > best_score {
                    best_score = score;
                    best_move = (from, to);
                }
            }
        }
        best_move
    }

    fn minimax(&mut self, depth: u8, maximizing: bool, mut alpha: f32, mut beta: f32) -> f32 {
        if depth == 0 {
            return self.evaluate();
        }
        if maximizing {
            let mut best = f32::MIN;
            for &from in self.get_pieces(self.current_player) {
                let piece = self.get(from).unwrap();
                for to in piece.get_legal_moves(from, &mut self.clone()) {
                    let mut game = self.clone();
                    game.move_piece(from, to);
                    let score = game.minimax(depth - 1, false, alpha, beta);
                    best = f32::max(best, score);
                    alpha = f32::max(alpha, score);
                    if beta <= alpha {
                        break;
                    }
                }
            }
            return best;
        } else {
            let mut best = f32::MAX;
            for &from in self.get_pieces(self.current_player) {
                let piece = self.get(from).unwrap();
                for to in piece.get_legal_moves(from, &mut self.clone()) {
                    let mut game = self.clone();
                    game.move_piece(from, to);
                    let score = game.minimax(depth - 1, true, alpha, beta);
                    best = f32::min(best, score);
                    beta = f32::min(beta, score);
                    if beta <= alpha {
                        break;
                    }
                }
            }
            return best;
        }
    }

    pub fn turn(&mut self) {
        if self.current_player == Player::Two && self.model.is_some() {
            return self.algorithm_move();
        }
        println!("{}", self);
        println!("It's {}'s turn.", self.current_player);
        if self.stalemate() {
                println!("Player {} is in stalemate!", self.current_player.number());
            self.game_over = true;
            return;
        }
        let mut in_check = false;
        if self.player_in_check() {
            println!("You're in check!");
            in_check = true;
        }
        self.tick();
        println!(
            "Enter your move (e.g. a2 a4) or enter a position to see its possible moves (e.g. a2):"
        );
        
        let mut valid_move = false;
        while !valid_move {
            let (from, to) = self.get_move();
            let mut half_move = false;
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
                        half_move = true;
                        self.take(to, piece.clone());
                        self.set(from, None);
                    }
                } else {
                    self.set(to, piece.clone());
                    self.set(from, None);
                }
                if self.player_in_check() {
                    if in_check {
                        println!("Invalid move while you are in check! go again");
                    } else {
                        println!("Wait you can't put yourself in check! go again.");
                    }
                    self.set(from, self.get(to));
                    self.set(to, conquered);
                    continue;
                } else {
                    self.set_last_double(None);
                    match move_status {
                        Move::Normal => (),
                        Move::Double(position) => {
                            self.set_last_double(Some(position));
                        }
                        Move::Castle => {
                            self.castle(to);
                        }
                        Move::EnPassant(position) => {
                            self.take(position, None);
                        }
                        Move::Invalid => unreachable!(),
                    }
                    if piece.clone().unwrap().is_type::<Pawn>() {
                        half_move = true;
                    }
                    self.set_moved(piece.clone(), from, to);
                }
                if (to.1 == 7 || to.1 == 0) && piece.unwrap().is_type::<Pawn>() {
                    self.promote_piece(to);
                }
                if half_move {
                    self.half_move_clock = 0;
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
        self.current_player = self.current_player.other();
    }

    fn move_piece(&mut self, from: (u8, u8), to: (u8, u8)) {
        //TODO: check
        let piece = self.get(from);
        let conquered = self.get(to);
        let move_status = piece.clone().unwrap().valid_move(from, to, self);
        if move_status == Move::Invalid {
            panic!("Invalid move!");
        }
        if piece.clone().unwrap().player() != self.current_player {
            panic!("You must move one of your own pieces!");
        }
        self.tick();
        if let Some(conquered) = conquered.clone() {
            if conquered.player() == self.current_player {
                panic!("You can't take your own piece!");
            } else {
                self.take(to, piece.clone());
                self.set(from, None);
            }
        } else {
            self.set_last_double(None);
            match move_status {
                Move::Normal => (),
                Move::Double(position) => {
                    self.set_last_double(Some(position));
                }
                Move::Castle => {
                    self.castle(to);
                }
                Move::EnPassant(position) => {
                    self.take(position, None);
                }
                Move::Invalid => unreachable!(),
            }
            self.set_moved(piece, from, to)
            // self.update_piece(from, to);
        }
        self.current_player = self.current_player.other();
    }
    
    fn evaluate(&mut self) -> f32 {
        let mut data = [[[0; 8]; 8]; 13];
        for &position in self.get_pieces(Player::One) {
            let piece = self.get(position).unwrap();
            if piece.is_type::<Queen>() {
                data[0][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<King>() {
                data[2][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Rook>() {
                data[4][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Bishop>() {
                data[6][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Knight>() {
                data[8][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Pawn>() {
                data[10][position.0 as usize][position.1 as usize] = 1;
            }
        }
        for &position in self.get_pieces(Player::Two) {
            let piece = self.get(position).unwrap();
            if piece.is_type::<Queen>() {
                data[1][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<King>() {
                data[3][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Rook>() {
                data[5][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Bishop>() {
                data[7][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Knight>() {
                data[9][position.0 as usize][position.1 as usize] = 1;
            } else if piece.is_type::<Pawn>() {
                data[11][position.0 as usize][position.1 as usize] = 1;
            }
        }
        if self.last_double.is_some() {
            let (x, y) = self.last_double.unwrap();
            data[12][x as usize][y as usize] = 1;
        }
        let king_pos_1 = self.get_king(Player::One);
        let king_pos_2 = self.get_king(Player::Two);
        let king_1 = self.get(king_pos_1).unwrap();
        let king_1 = king_1.get_piece::<King>().unwrap();
        let king_2 = self.get(king_pos_2).unwrap();
        let king_2 = king_2.get_piece::<King>().unwrap();
        if king_1.can_castle_left(king_pos_1, self) {
            data[12][7][0] = 1;
        }
        if king_1.can_castle_right(king_pos_1, self) {
            data[12][7][7] = 1;
        }
        if king_2.can_castle_left(king_pos_2, self) {
            data[12][0][0] = 1;
        }
        if king_2.can_castle_right(king_pos_2, self) {
            data[12][0][7] = 1;
        }
        if self.current_player == Player::One {
            data[12][7][4] = 1;
        } else {
            data[12][0][4] = 1;
        }
        if self.half_move_clock > 0 {
            let mut half_move_clock = self.half_move_clock;
            let mut c = 7;
            while self.half_move_clock > 0 && c >= 0 {
                data[12][3][c as usize] = half_move_clock % 2;
                half_move_clock = half_move_clock / 2;
                c -= 1;
            }
        }
        if self.full_move_clock > 0 {
            let mut full_move_clock = self.full_move_clock;
            let mut c = 7;
            while self.full_move_clock > 0 && c >= 0 {
                data[12][4][c as usize] = full_move_clock % 2;
                full_move_clock = full_move_clock / 2;
                c -= 1;
            }
        }
        println!("{:?}", data);
        self.model.as_ref().unwrap().run_inference(data).unwrap()
    }

    pub fn algorithm_move(&mut self) {
        let (from, to) = self.get_best_move();
        println!("Player two moved {} -> {} ", coord_to_pos(from), coord_to_pos(to));
        self.move_piece(from, to);
    }

    fn checkmate(&mut self) -> bool {
        if !self.in_check(self.current_player) {
            return false;
        }
        // for every enemy that can attack king {
        //         for every square in path from enemy to king {
            //                 for every friendly piece {
                //                         if friendly piece can move to square {
                    //                                 return false;
        //                         }
        //                 }
        //         }
        // }
        let king = self.get_king(self.current_player);
        let pieces = self.get_pieces(self.current_player).clone();
        let enemy_pieces = self.get_pieces(self.current_player.other()).clone();
        for enemy_position in enemy_pieces {
            let enemy = self.get(enemy_position).unwrap();
            if enemy.valid_move(enemy_position, king, self) != Move::Invalid {
                //if it puts the king in check
                for friendly_position in pieces.clone() {
                    let friendly = self.get(friendly_position).unwrap();
                    //for every friendly piece, see if it can block the path and get us out of check
                    for square in
                        friendly.can_intercept_path(friendly_position, enemy_position, king, self)
                    {
                        if !self.try_move_for_check(friendly_position, square, self.current_player)
                        {
                            return false;
                        }
                    }
                }
            }
        }
        let (x, y) = king;
        for (i, j) in [(0, 1),(1, 0),(0, -1),(-1, 0),(1, 1),(1, -1),(-1, 1),(-1, -1)] {
            let (new_x, new_y) = (x as i8 + i, y as i8 + j);
            if new_x < 0 || new_x >= 8 || new_y < 0 || new_y >= 8 {
                continue;
            }
            let new_pos = (new_x as u8, new_y as u8);
            if !self.is_current_player(new_pos) {
                if !self.try_move_for_check(king, new_pos, self.current_player) {
                    return false;
                }
            }
        }
        true
    }

    #[cfg(test)]
    pub(crate) fn set_board(&mut self, board: Board) {
        self.board = board;
    }

    #[cfg(test)]
    pub(crate) fn set_pieces(&mut self, p1_pieces: Vec<(u8, u8)>, p2_pieces: Vec<(u8, u8)>) {
        self.p1_pieces = p1_pieces;
        self.p2_pieces = p2_pieces;
    }

    #[cfg(test)]
    pub(crate) fn set_model(&mut self, model: Option<Model>) {
        self.model = model;
    }

    pub fn is_over(&mut self) -> bool {
        self.game_over = self.game_over || self.checkmate();
        self.game_over
    }

    pub fn play_again(&self) -> bool {
        println!("{self}");
        println!("Game over!");
        println!("{} wins!", self.current_player);
        println!("Play again? (y/n)");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input.trim().to_ascii_lowercase() == "y"
    }

    // fn get_piece_score(&self) -> i32 {
    //     let mut score = 0;
    //     for piece in self.p1_pieces.iter() {
    //         let piece = self.get(*piece).unwrap();
    //         score += piece.value();
    //     }
    //     for piece in self.p2_pieces.iter() {
    //         let piece = self.get(*piece).unwrap();
    //         score -= piece.value();
    //     }
    //     score
    // }

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
            for i in min(from.1, to.1)..=max(from.1, to.1) {
                if self.get((from.0, i)).is_some() && i != from.1 && i != to.1 {
                    return false;
                }
            }
        } else {
            for i in min(from.0, to.0)..=max(from.0, to.0) {
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
            if self
                .get((
                    (from.0 as i8 + signs.0 * i as i8) as u8,
                    (from.1 as i8 + signs.1 * i as i8) as u8,
                ))
                .is_some()
            {
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

    pub(crate) fn take(&mut self, to: (u8, u8), new_piece: Square) {
        let piece = self.get(to).unwrap();
        if piece.name() == "king" {
            panic!("You can't take a king, something went wrong!");
        }
        println!(
            "Player {} took {}'s {}!",
            self.current_player.number(),
            self.current_player.other(),
            piece.name()
        );
        self.remove_piece(to);
        self.set(to, new_piece);
        let i = match piece.name() {
            "pawn" => 0,
            "rook" => 1,
            "bishop" => 2,
            "knight" => 3,
            "queen" => 4,
            _ => panic!("Invalid piece name!"),
        };
        let taken = match self.current_player {
            Player::One => &mut self.p1_taken[i],
            Player::Two => &mut self.p2_taken[i],
        };
        *taken += 1;
    }

    fn display_taken(&self, player: Player) -> Option<String> {
        let taken = match player {
            Player::One => &self.p1_taken,
            Player::Two => &self.p2_taken,
        };
        let mut taken_vec = vec!();
        for i in 0..5 {
            if taken[i] > 0 {
                let piece: Box<dyn Piece> = match i {
                    0 => Box::new(Pawn::new(player.other())),
                    1 => Box::new(Rook::new(player.other())),
                    2 => Box::new(Bishop::new(player.other())),
                    3 => Box::new(Knight::new(player.other())),
                    4 => Box::new(Queen::new(player.other())),
                    _ => unreachable!(),
                };
                if taken[i]>1 {
                    taken_vec.push(format!("{}×{piece} ", taken[i]));
                } else {
                    taken_vec.push(format!("{piece} "));
                }
            }
        }
        if taken_vec.is_empty() { None } else { Some(taken_vec.join(", ")) }
    }

    fn remove_piece(&mut self, position: (u8, u8)) {
        match self.get(position).unwrap().player() {
            Player::One => self.p1_pieces.retain(|&x| x != position),
            Player::Two => self.p2_pieces.retain(|&x| x != position),
        }
    }

    pub(crate) fn get(&self, (x, y): (u8, u8)) -> Square {
        self.board[y as usize][x as usize].clone()
    }

    fn set(&mut self, (x, y): (u8, u8), piece: Square) {
        self.board[y as usize][x as usize] = piece;
    }

    pub(crate) fn in_check(&mut self, player: Player) -> bool {
        let king = self.get_king(player);
        for position in self.get_pieces(player.other()).clone() {
            let piece = self.get(position).unwrap();
            if piece.valid_move(position, king, self) != Move::Invalid {
                return true;
            }
        }
        false
    }

    fn get_king(&self, player: Player) -> (u8, u8) {
        match player {
            Player::One => self.king_one,
            Player::Two => self.king_two,
        }
    }

    fn player_in_check(&mut self) -> bool {
        self.in_check(self.current_player)
    }

    fn get_move(&mut self) -> ((u8, u8), (u8, u8)) {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        if input.to_ascii_lowercase().trim() == "exit" {
            std::process::exit(0);
        } else if input.to_ascii_lowercase().trim() == "resign" {
            println!("Player {} resigned, {} wins!", self.current_player.number(), self.current_player.other());
            std::process::exit(0);
        }
        let mut input = input.split_whitespace();
        let from = input.next();
        let to = input.next();
        if from.is_none() {
            println!("Invalid input! Try again.");
            return self.get_move();
        } else if to.is_none() {
            let from = from.unwrap().to_ascii_lowercase();
            let from = (from.chars().nth(0), from.chars().nth(1));
            if from.0.is_none() || from.1.is_none() {
                println!("Invalid input! Try again.");
                return self.get_move();
            }
            let from = (
                from.0.unwrap() as i8 - 'a' as i8,
                '8' as i8 - from.1.unwrap() as i8,
            );
            if from.0 < 0 || from.0 > 7 || from.1 < 0 || from.1 > 7 {
                println!("Invalid input! Try again.");
                return self.get_move();
            }
            let from = (from.0 as u8, from.1 as u8);
            self.see_all_moves(from);
            println!("Enter your move (e.g. a2 a4):");
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
        let from = (
            from.0.unwrap() as i8 - 'a' as i8,
            '8' as i8 - from.1.unwrap() as i8,
        );
        let to = (
            to.0.unwrap() as i8 - 'a' as i8,
            '8' as i8 - to.1.unwrap() as i8,
        );
        if from.0 < 0 || from.1 < 0 || from.0 > 7 || from.1 > 7 || to.0 < 0 || to.1 < 0 || to.0 > 7 || to.1 > 7 {
            println!("Invalid input! Try again.");
            return self.get_move();
        }
        let from = (from.0 as u8, from.1 as u8);
        let to = (to.0 as u8, to.1 as u8);
        (from, to)
    }

    fn promote_piece(&mut self, position: (u8, u8)) {
        println!("Pawn promotion! Enter a piece to promote to: (q, r, b, k)");
        let mut input = String::new();
        let piece: Box<dyn Piece>;
        loop {
            io::stdin().read_line(&mut input).unwrap();
            piece = match input
                .trim()
                .chars()
                .next()
                .unwrap_or(' ')
                .to_ascii_lowercase()
            {
                'q' => Box::new(Queen::new(self.current_player)),
                'r' => Box::new(Rook::new(self.current_player)),
                'b' => Box::new(Bishop::new(self.current_player)),
                'k' => Box::new(Knight::new(self.current_player)),
                _ => {
                    println!("Invalid piece! Enter another.");
                    continue;
                }
            };
            break;
        }
        self.set(position, Some(piece));
    }

    fn castle(&mut self, king_to: (u8, u8)) {
        let (rook_from, rook_to) = match king_to.0 {
            6 => ((7, king_to.1), (5, king_to.1)),
            2 => ((0, king_to.1), (3, king_to.1)),
            _ => unreachable!(),
        };
        self.set(rook_to, self.get(rook_from));
        self.set(rook_from, None);
        self.update_piece(rook_from, rook_to);
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

    pub(crate) fn is_not_player(&self, new_pos: (u8, u8), player: Player) -> bool {
        if let Some(piece) = self.get(new_pos) {
            piece.player() != player
        } else {
            true
        }
    }

    fn set_moved(&mut self, piece: Square, from: (u8, u8), to: (u8, u8)) {
        let piece = piece.unwrap();
        if piece.is_type::<King>() {
            match piece.player() {
                Player::One => {
                    self.king_one = to;
                    self.has_p1_king_moved = true
                }
                Player::Two => {
                    self.king_two = to;
                    self.has_p2_king_moved = true
                }
            }
        } else if piece.is_type::<Rook>() {
            match piece.player() {
                Player::One => {
                    if from == (0, 0) {
                        self.has_p1_left_rook_moved = true;
                    } else if from == (7, 0) {
                        self.has_p1_right_rook_moved = true;
                    }
                }
                Player::Two => {
                    if from == (0, 7) {
                        self.has_p2_left_rook_moved = true;
                    } else if from == (7, 7) {
                        self.has_p2_right_rook_moved = true;
                    }
                }
            }
        }
        self.update_piece(from, to)
    }

    pub(crate) fn has_king_moved(&self, player: Player) -> bool {
        match player {
            Player::One => self.has_p1_king_moved,
            Player::Two => self.has_p2_king_moved,
        }
    }

    pub(crate) fn has_left_rook_moved(&self, player: Player) -> bool {
        match player {
            Player::One => self.has_p1_left_rook_moved,
            Player::Two => self.has_p2_left_rook_moved,
        }
    }

    pub(crate) fn has_right_rook_moved(&self, player: Player) -> bool {
        match player {
            Player::One => self.has_p1_right_rook_moved,
            Player::Two => self.has_p2_right_rook_moved,
        }
    }

    fn see_all_moves(&mut self, from: (u8, u8)) {
        if let Some(piece) = self.get(from) {
            let moves = piece.get_legal_moves(from, self);
            if moves.len() == 0 {
                println!("Player {}'s {} has no legal moves!", piece.player().number(), piece.name());
                return;
            }
            println!(
                "Player {}'s {} can move to:",
                piece.player().number(),
                piece.name()
            );
            let moves = moves
                .iter()
                .map(|(x, y)| format!("{}{}", (x + 'a' as u8) as char, 8 - y))
                .collect::<Vec<String>>()
                .join(",");
            println!("{moves}");
        } else {
            println!("There's no piece there!");
        }
    }

    pub(crate) fn is_player(&self, new_pos: (u8, u8), player: Player) -> bool {
        if let Some(piece) = self.get(new_pos) {
            piece.player() == player
        } else {
            false
        }
    }

    pub(crate) fn try_move_for_check(
        &mut self,
        from: (u8, u8),
        to: (u8, u8),
        player: Player,
    ) -> bool {
        let friendly = self.get(from);
        let is_king = friendly.clone().unwrap().is_type::<King>();
        let old = self.get(to);
        let enemy_pieces = self.get_pieces(player.other()).clone();
        // assumes that old isn't a friendly piece
        if old.is_some() {
            self.remove_piece(to);
        }
        self.set(to, friendly.clone());
        self.set(from, None);
        if is_king {
            self.set_king(player, to);
        }
        let still_in_check = self.in_check(player);
        self.set(from, friendly);
        self.set(to, old);
        if is_king {
            self.set_king(player, from);
        }
        self.set_player_pieces(player.other(), enemy_pieces);
        still_in_check
    }

    fn set_king(&mut self, player: Player, king: (u8, u8)) {
        match player {
            Player::One => self.king_one = king,
            Player::Two => self.king_two = king,
        }
    }

    pub fn get_pieces(&self, player: Player) -> &Vec<(u8, u8)> {
        match player {
            Player::One => &self.p1_pieces,
            Player::Two => &self.p2_pieces,
        }
    }

    fn update_piece(&mut self, from: (u8, u8), to: (u8, u8)) {
        match self.current_player {
            Player::One => {
                self.p1_pieces.retain(|x| *x != from);
                self.p1_pieces.push(to);
            }
            Player::Two => {
                self.p2_pieces.retain(|x| *x != from);
                self.p2_pieces.push(to);
            }
        }
    }

    pub(crate) fn set_player_pieces(&mut self, player: Player, pieces: Vec<(u8, u8)>) {
        match player {
            Player::One => self.p1_pieces = pieces.to_vec(),
            Player::Two => self.p2_pieces = pieces.to_vec(),
        }
    }

    fn stalemate(&mut self) -> bool {
        self.get_possible_moves(self.current_player).is_empty()
    }

    pub fn get_possible_moves(&mut self, player: Player) -> Vec<((u8, u8), Vec<(u8, u8)>)> {
        let mut moves = Vec::new();
        for position in self.get_pieces(player).clone() {
            let piece = self.get(position).expect("Piece not found!");
            let piece_moves = piece.get_legal_moves(position, self);
            if !piece_moves.is_empty() {
                moves.push((position, piece_moves));
            }
            // moves.extend(moves.iter().map(|x| (position, *x))); -> Vec<((u8,u8),(u8,u8))>>
        }
        moves
    }

    fn tick(&mut self) {
        if self.half_move_clock >= 50 {
            println!("50 moves without a capture or pawn move, it's a draw!");
            self.game_over = true;
            return;
        }
        self.half_move_clock += 1;
        self.full_move_clock += 1;
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        writeln!(f, "{}", format!("\t\tPlayer 2").red().bold())?;
        if let Some(taken2) = self.display_taken(Player::Two) {
            writeln!(f, "{}", taken2)?;
        }
        writeln!(f, "    a    b    c    d    e    f    g    h")?;
        self.board.iter().enumerate().for_each(|(i, row)| {
            writeln!(f, "  -----------------------------------------").unwrap();
            write!(f, "{} ", 8 - i).unwrap();
            row.iter().enumerate().for_each(|(_, piece)| {
                match piece {
                    // Some(piece) => write!(f, "|{}", if (i+j)%2==0 {format!(" {piece}  ").on_black()} else {format!(" {piece}  ").on_bright_black()}),
                    // None => write!(f,"|{}", if (i+j)%2==0 {format!("    ").on_black()} else {format!("    ").on_bright_black()}),
                    Some(piece) => write!(f, "|{}", format!(" {piece}  ").on_black()),
                    None => write!(f, "|{}", format!("    ").on_black()),
                }
                .unwrap();
            });
            write!(f, "|").unwrap();
            writeln!(f, " {}", 8 - i).unwrap();
        });
        // writeln!(f, "  ---------------------------------")?;
        writeln!(f, "  -----------------------------------------")?;
        // writeln!(f, "    a   b   c   d   e   f   g   h")?;
        writeln!(f, "    a    b    c    d    e    f    g    h")?;
        // writeln!(f, "{}", format!("\t      Player 2").blue().bold())?;
        if let Some(taken1) = self.display_taken(Player::One) {
            writeln!(f, "{}", taken1)?;
        }
        writeln!(f, "{}", format!("\t\tPlayer 1").blue().bold())?;
        Ok(())
    }
}

fn coord_to_pos(coord: (u8,u8)) -> String {
    format!("{}{}", (coord.0 + 'a' as u8) as char, (coord.1 + '8' as u8) as char)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkmate_no_friendlies() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Queen::new(Player::Two)));
        board[1][0] = Some(Box::new(Queen::new(Player::Two)));

        let mut game = Game::new();
        game.set_board(board);
        game.set_pieces(vec![(0, 0)], vec![(1, 0), (0, 1)]);
        game.set_king(Player::One, (0, 0));
        game.set_model(None);

        assert!(game.checkmate());
    }

    #[test]
    fn checkmate_no_friendlies2() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Queen::new(Player::Two)));
        board[1][0] = Some(Box::new(Rook::new(Player::Two)));
        board[0][2] = Some(Box::new(Queen::new(Player::Two)));

        let mut game = Game::new();
        game.set_board(board);
        game.set_pieces(vec![(0, 0)], vec![(1, 0), (0, 1), (2, 0)]);
        game.set_king(Player::One, (0, 0));
        game.set_model(None);

        print!("{game}");

        assert!(game.checkmate());
    }

    #[test]
    fn no_checkmate_no_friendlies() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Rook::new(Player::Two)));
        board[1][0] = Some(Box::new(Rook::new(Player::Two)));

        let mut game = Game::new();
        game.set_board(board);
        game.set_pieces(vec![(0, 0)], vec![(1, 0), (0, 1)]);
        game.set_king(Player::One, (0, 0));
        game.set_model(None);

        print!("{game}");

        assert!(!game.checkmate());
    }

    #[test]
    fn no_checkmate_blockable() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Knight::new(Player::One)));
        board[2][0] = Some(Box::new(Queen::new(Player::Two)));

        let mut game = Game::new();
        game.set_board(board);
        game.set_pieces(vec![(0, 0), (1, 0)], vec![(0, 2)]);
        game.set_king(Player::One, (0, 0));
        game.set_model(None);

        print!("{game}");

        assert!(!game.checkmate());
    }

    #[test]
    fn checkmate_unblockable() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Rook::new(Player::One)));
        board[1][1] = Some(Box::new(Pawn::new(Player::One)));
        board[3][0] = Some(Box::new(Queen::new(Player::Two)));

        let mut game = Game::new();
        game.set_board(board);
        game.set_pieces(vec![(0, 0), (1, 0), (1, 1)], vec![(0, 3)]);
        game.set_king(Player::One, (0, 0));
        game.set_model(None);

        assert!(game.checkmate());
    }

    #[test]
    fn checking_all_legal_moves_are_valid() {
        let mut game = Game::new();
        game.set_model(None);

        for position in game.p1_pieces.clone() {
            let piece = game.get(position).unwrap();
            for (x, y) in piece.get_legal_moves(position, &mut game) {
                assert!(piece.valid_move(position, (x, y), &mut game) != Move::Invalid);
            }
        }
    }

    #[test]
    fn stalemate() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[7][5] = Some(Box::new(King::new(Player::One)));
        board[4][4] = Some(Box::new(Queen::new(Player::Two)));
        board[4][3] = Some(Box::new(Bishop::new(Player::Two)));
        board[2][3] = Some(Box::new(King::new(Player::Two)));

        let mut game = Game::new();
        game.set_board(board);
        game.set_pieces(vec![(5, 7)], vec![(4, 4), (3, 4), (3, 2)]);
        game.set_king(Player::One, (5, 7));
        game.set_king(Player::Two, (3, 2));
        game.set_model(None);

        println!("{game}");

        assert!(game.stalemate());
        assert!(!game.checkmate());
    }
}
