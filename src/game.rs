use crate::{
    bishop::Bishop,
    king::King,
    knight::Knight,
    model::Model,
    pawn::Pawn,
    piece::{Construct, Move, Piece},
    player::Player,
    queen::Queen,
    rook::Rook,
};
use colored::Colorize;
use rand::{Rng, seq::SliceRandom};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::{
    cmp::{max, min},
    fmt::{Display, Error, Formatter},
    io, collections::HashMap, sync::{Mutex, Arc},
};

pub type Square = Option<Box<dyn Piece>>;
pub type Board = Vec<Vec<Square>>;
pub type Matrix = [[[f32; 8]; 8]; 13];
pub type Cache = Arc<Mutex<HashMap<String,(f32, u8)>>>;
// const NUM_THREADS: usize = 4;
const DEFAULT_SEARCH_DEPTH: u8 = 2;
const HALF_MOVE_LIMIT: u8 = 100;
pub const DEFAULT_EPSILON_DECAY: f64 = 0.98;

#[derive(Clone)]
pub struct Game<'a> {
    board: Vec<Vec<Square>>,
    current_player: Player,
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
    full_move_clock: u32,
    in_simulation: bool,
    two_player: bool,
    model: Option<&'a Model>,
    computer_player: Option<Player>,
    cache: Cache,
    rl_training: bool,
    search_depth: u8,
    epsilon_greedy: bool,
    epsilon: f64,
    epsilon_decay_rate: f64,
    allow_hints: bool,
    winner: Option<Player>
    // TODO: opening_book
}

impl<'a> Game<'a> {
    pub fn new(
        two_player: bool,
        computer_player: Option<Player>,
        rl_training: bool,
        model: Option<&'a Model>,
        search_depth: Option<u8>,
        epsilon_greedy: bool,
        allow_hints: bool,
        epsilon: Option<f64>,
        epsilon_decay_rate: Option<f64>,
        cache: Option<Cache>
    ) -> Self {
        Self {
            board: setup_board(),
            current_player: Player::One,
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
            in_simulation: false,
            two_player,
            model,
            computer_player,
            cache: cache.unwrap_or(Arc::new(Mutex::new(HashMap::new()))),
            rl_training,
            search_depth: search_depth.unwrap_or(DEFAULT_SEARCH_DEPTH),
            epsilon_greedy,
            epsilon: epsilon.unwrap_or(0.0),
            epsilon_decay_rate: epsilon_decay_rate.unwrap_or(DEFAULT_EPSILON_DECAY),
            allow_hints,
            winner: None,
        }
    }

    pub fn two_player_game(allow_hints: bool) -> Self {
        Self::new(true, None, false, None, Some(6),false, allow_hints, None, None, None)
    }

    pub fn single_player_game(
        player: Option<Player>,
        model: Option<&'a Model>,
        search_depth: Option<u8>,
    ) -> Self {
        Self::new(false, player, false, model, search_depth, false, true, None, None, None)
    }

    pub fn self_play(model: Option<&'a Model>, search_depth: Option<u8>, epsilon_greedy: bool, initial_epsilon: Option<f64>, epsilon_decay: Option<f64>, cache: Option<Cache>) -> Self {
        Self::new(false, None, true, model, search_depth, epsilon_greedy, false, initial_epsilon, epsilon_decay, cache)
    }

    fn get_best_move(&mut self) -> Option<((u8, u8), (u8, u8))> {
        println!("Thinking...");
        self.in_simulation = true;
        let now = std::time::SystemTime::now();
        // let possible_moves = self.get_moves_sorted(true);
        let possible_moves = self.get_possible_moves(self.current_player);
        if possible_moves.is_empty() {
            return None;
        }

        let move_evals: Vec<(((u8, u8), (u8, u8)), f32)> = possible_moves.par_iter().map(|&mov| {
            let mut game = self.clone();
            game.move_piece(mov.0, mov.1);
            let score = game.minimax_search(self.search_depth - 1, game.current_player.is_maximizing(), f32::MIN, f32::MAX);
            (mov, score)
        }).collect();

        // move_evals.iter().for_each(|(mov, score)| {
        //     print!("{} -> {} : {}\n", format_coord(mov.0), format_coord(mov.1), score);
        // });

        let best_score = if self.is_maximizing() {
            move_evals.iter()
                .max_by(|(_, score1), (_, score2)| score1.partial_cmp(score2).unwrap())
                .unwrap().1
        } else {
            move_evals.iter()
                .min_by(|(_, score1), (_, score2)| score1.partial_cmp(score2).unwrap())
                .unwrap().1
        };
        let best_moves = move_evals.par_iter()
            .filter(|(_, score)| *score == best_score)
            .map(|(mov, _)| mov)
            .collect::<Vec<&((u8, u8), (u8, u8))>>();
        let best_move = best_moves.choose(&mut rand::thread_rng()).unwrap();

        let elapsed = now.elapsed().unwrap();
        println!(
            "Time to evaluate best move to depth of {}: {:?}",
            self.search_depth, elapsed
        );
        self.in_simulation = false;
        Some(**best_move)
    }

    fn get_best_move_and_back_propagate(&mut self) -> ((u8, u8), (u8, u8)) {
        let now = std::time::SystemTime::now();
        let possible_moves = self.get_possible_moves(self.current_player);

        let games: Vec<Game> = possible_moves
            .par_iter()
            .map(|&(from, to)| {
                let mut game = self.clone();
                game.move_piece(from, to);
                game
            })
            .collect();

        let matrices = games
            .par_iter()
            .map(|game| game.clone().to_matrix())
            .collect::<Vec<Matrix>>();

        let amplified_scores = games
            .par_iter()
            .map(|game| {
                // TODO: use one hashmap the whole time?
                game.clone()
                    .minimax_search(self.search_depth - 1, game.current_player.is_maximizing(), f32::MIN, f32::MAX)
            })
            .collect::<Vec<f32>>();
        // cache amplified_scores
        let mut best_move = possible_moves[0];
        let mut best_score = if self.is_maximizing() {
            f32::MIN
        } else {
            f32::MAX
        };
        for i in 0..possible_moves.len() {
            if self.current_player == Player::One {
                if amplified_scores[i] > best_score {
                    best_score = amplified_scores[i];
                    best_move = possible_moves[i];
                }
            } else {
                if amplified_scores[i] < best_score {
                    best_score = amplified_scores[i];
                    best_move = possible_moves[i];
                }
            }
        }
        let elapsed = now.elapsed().unwrap();
        println!(
            "Time to evaluate best move to depth of {}: {:?}",
            self.search_depth, elapsed
        );

        self.model
        .as_ref()
        .unwrap()
        .back_propagate(&matrices, &amplified_scores);
        best_move
    }

    fn minimax_search(&mut self, depth: u8, maximizing: bool, mut alpha: f32, mut beta: f32) -> f32 {
        // the algorithm assumes a good heuristic or evaluation model to approximate game state evaluations
        if self.draw() {
            return 0.0;
        }
        let fen = self.to_short_fen();
        if let Some(&(score, stored_depth)) = self.cache.lock().unwrap().get(&fen) {
            if stored_depth >= depth {
                return score;
            }
        }
        if depth <= 1 && self.model.is_some() {
            return self.last_level_minimax(maximizing, alpha, beta);
        }
        if depth == 0 {
            return self.evaluate()
        }
        let moves = self.get_possible_moves(self.current_player);
        let mut best_score;
        if maximizing {
            best_score = f32::MIN;
            for (from, to) in moves {
                let mut game = self.clone();
                game.move_piece(from, to);
                best_score = f32::max(best_score, game.minimax_search(depth - 1, false, alpha, beta));
                alpha = f32::max(alpha, best_score);
                if alpha >= beta {
                    break;
                }
            }
        } else {
            best_score = f32::MAX;
            for (from, to) in moves {
                let mut game = self.clone();
                game.move_piece(from, to);
                best_score = f32::min(best_score, game.minimax_search(depth - 1, true, alpha, beta));
                beta = f32::min(beta, best_score);
                if alpha >= beta {
                    break;
                }
            }
        }
        self.cache.lock().unwrap().insert(fen, (best_score, depth));
        best_score
    }

    fn last_level_minimax(&mut self, maximizing: bool, alpha: f32, beta: f32) -> f32 {
        let moves = self.get_possible_moves(self.current_player);
        if moves.is_empty() {
            return 0.0;
        };
        let move_evals = self.evaluate_moves(&moves);
        if beta <= alpha {
            move_evals[0]
        } else if maximizing {
            move_evals
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .clone()
        } else {
            move_evals
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .clone()
        }
    }

    fn draw(&mut self) -> bool {
        self.half_move_clock_expired() || (self.stalemate() && !self.checkmate())
    }

    pub fn turn(&mut self) -> bool {
        println!("{}", self);
        println!("It's {}'s turn.", self.current_player);

        if self.rl_training {
            if self.model.is_some() {
                return self.rl_training_move();
            } else {
                return self.algorithm_move();
            }
        }
        if !self.two_player && self.current_player == self.computer_player.unwrap() {
            return self.algorithm_move();
        }
        if self.stalemate() {
            println!(
                "No possible moves for player {}!",
                self.current_player.number()
            );
            println!("Stalemate!");
            return true;
        }
        let mut in_check = false;
        if self.player_in_check() {
            println!("You're in check!");
            in_check = true;
        }
        if self.tick() {
            return true;
        }

        loop {
            let (from, to) = self.get_move();
            let mut half_move = false;
            if !self.is_current_player(from) {
                println!("You must move one of your own pieces! go again.");
                continue;
            }
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
            let (p1_pieces, p2_pieces) = (
                self.get_pieces(Player::One).clone(),
                self.get_pieces(Player::Two).clone(),
            );
            self.set_moved(piece.clone(), from, to);
            if self.player_in_check() {
                // in theory if enpassant could not be in check anymore
                if in_check {
                    println!("Invalid move while you are in check! go again");
                } else {
                    println!("Wait you can't put yourself in check! go again.");
                }
                self.set(from, self.get(to));
                self.set(to, conquered);
                self.set_pieces(p1_pieces, p2_pieces);
                continue;
            }
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
                if to.1 == 7 || to.1 == 0 {
                    self.promote_piece(to);
                }
            }
            if half_move {
                self.half_move_clock = 0;
            }
            break;
        }
        self.current_player = self.current_player.other();
        if self.checkmate() {
            println!("{self}");
            println!("Game over!");
            println!(
                "Player {} is in checkmate, {} wins!",
                self.current_player.number(),
                self.current_player.other()
            );
            return true;
        }
        return false;
    }

    #[cfg(test)]
    fn assert_pieces(&mut self) {
        for position in self.p1_pieces.clone() {
            let piece = self.get(position).expect("assertion failed: piece position is empty");
            assert_eq!(piece.player(), Player::One);
            for target in piece.get_legal_moves(position, self) {
                assert!(piece.valid_move(position, target, self).is_valid());
            }
        }
        for position in self.p2_pieces.clone() {
            let piece = self.get(position).expect("assertion failed: piece position is empty");
            assert_eq!(piece.player(), Player::Two);
            for target in piece.get_legal_moves(position, self) {
                assert!(piece.valid_move(position, target, self).is_valid());
            }
        }
        assert!(self.get(self.king_one).unwrap().is_type::<King>());
        assert!(self.get(self.king_two).unwrap().is_type::<King>());
        assert!(self.p2_taken.iter().sum::<u8>() + self.p1_pieces.len() as u8 == 16);
        assert!(self.p1_taken.iter().sum::<u8>() + self.p2_pieces.len() as u8 == 16);
    }

    fn move_piece(&mut self, from: (u8, u8), to: (u8, u8)) -> bool {
        if self.tick() {
            return true;
        }
        let piece = self.get(from);
        let conquered = self.get(to);
        let move_status = piece.clone().unwrap().valid_move(from, to, self);
        assert!(move_status.is_valid(), "Invalid move!");
        assert_eq!(
            piece.clone().unwrap().player(),
            self.current_player,
            "You must move one of your own pieces!"
        );
        assert!(self.get_pieces(self.current_player).contains(&from));
        let mut half_move = false;
        if let Some(conquered) = conquered.clone() {
            assert_ne!(
                conquered.player(),
                self.current_player,
                "You can't take your own piece!"
            );
            half_move = true;
            self.take(to, piece.clone());
            self.set(from, None);
        } else {
            self.set(to, piece.clone());
            self.set(from, None);
        }
        self.set_moved(piece.clone(), from, to);
        assert!(!self.get_pieces(self.current_player).contains(&from));
        assert!(self.get_pieces(self.current_player).contains(&to));
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
        if piece.unwrap().is_type::<Pawn>() {
            half_move = true;
            if to.1 == 7 || to.1 == 0 {
                self.set(to, Some(Box::new(Queen::new(self.current_player))));
            }
        }
        if half_move {
            self.half_move_clock = 0;
        }
        // if self.player_in_check() {
        //     println!("Wait you can't put yourself in check, something went wrong");
        //     println!("last move was {} -> {}", format_coord(from), format_coord(to));
        //     println!("{self}");
        // }
        assert!(
            !self.player_in_check(),
            "Wait you can't put yourself in check!"
        );
        self.current_player = self.current_player.other();
        false
    }

    fn evaluate(&mut self) -> f32 {
        if let Some(model) = self.model {
            model.run_inference(&vec!(self.to_matrix())).unwrap()[0]
        } else {
            self.get_piece_scores() as f32
        }
    }

    fn to_matrix(&mut self) -> Matrix {
        let mut data = [[[0.0; 8]; 8]; 13];
        for &position in self.get_pieces(Player::One) {
            let piece = self.get(position).unwrap();
            if piece.is_type::<Queen>() {
                data[0][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<King>() {
                data[2][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Rook>() {
                data[4][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Bishop>() {
                data[6][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Knight>() {
                data[8][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Pawn>() {
                data[10][position.0 as usize][position.1 as usize] = 1.0;
            }
        }
        for &position in self.get_pieces(Player::Two) {
            let piece = self.get(position).unwrap();
            if piece.is_type::<Queen>() {
                data[1][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<King>() {
                data[3][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Rook>() {
                data[5][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Bishop>() {
                data[7][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Knight>() {
                data[9][position.0 as usize][position.1 as usize] = 1.0;
            } else if piece.is_type::<Pawn>() {
                data[11][position.0 as usize][position.1 as usize] = 1.0;
            }
        }
        if let Some(last_double) = self.last_double {
            let (x, y) = last_double;
            data[12][x as usize][y as usize] = 1.0;
        }
        if self.has_p1_king_moved {
            if self.has_p1_left_rook_moved {
                data[12][7][0] = 1.0;
            }
            if self.has_p1_right_rook_moved {
                data[12][7][7] = 1.0;
            }
        }
        if self.has_p2_king_moved {
            if self.has_p2_left_rook_moved {
                data[12][0][0] = 1.0;
            }
            if self.has_p2_right_rook_moved {
                data[12][0][7] = 1.0;
            }
        }
        if self.current_player == Player::One {
            data[12][7][4] = 1.0;
        } else {
            data[12][0][4] = 1.0;
        }
        if self.half_move_clock > 0 {
            let mut half_move_clock = self.half_move_clock;
            let mut c = 7;
            while self.half_move_clock > 0 && c >= 0 {
                data[12][3][c as usize] = (half_move_clock % 2) as f32;
                half_move_clock = half_move_clock / 2;
                c -= 1;
            }
        }
        if self.full_move_clock > 0 {
            let mut full_move_clock = self.full_move_clock;
            let mut c = 7;
            while self.full_move_clock > 0 && c >= 0 {
                data[12][4][c as usize] = (full_move_clock % 2) as f32;
                full_move_clock = full_move_clock / 2;
                c -= 1;
            }
        }
        data
    }

    pub fn algorithm_move(&mut self) -> bool {
        let best_move = self.get_best_move();
        if best_move.is_none() {
            // assert!(!self.check());
            // assert!(!self.checkmate());
            println!(
                "No possible moves for player {}!",
                self.current_player.number()
            );
            println!("Stalemate!");
            return true;
        }
        let (from, to) = best_move.unwrap();
        println!(
            "Player {} moved {}, {} -> {} ",
            self.current_player.number(),
            self.get(from).unwrap().name(),
            format_coord(&from),
            format_coord(&to)
        );
        if self.move_piece(from, to) {
            println!("{self}");
            println!("Game over!");
            println!("Draw, half move clock expired");
            return true;
        }
        if self.checkmate() {
            println!("{self}");
            println!("Game over!");
            println!(
                "Player {} is in checkmate, {} wins!",
                self.current_player.number(),
                self.current_player.other()
            );
            return true;
        }
        return false;
    }

    pub fn rl_training_move(&mut self) -> bool {
        self.in_simulation = true;
        let (mut from, mut to) = self.get_best_move_and_back_propagate();
        if self.epsilon_greedy {
            let mut rng = rand::thread_rng();
            let choice = rng.gen_bool(self.epsilon);
            if choice {
                let moves = self.get_possible_moves(self.current_player);
                (from, to) = *moves.choose(&mut rng).unwrap();
            }
            self.update_epsilon();
        }
        println!(
            "Player {} moved {} -> {} ",
            self.current_player.number(),
            format_coord(&from),
            format_coord(&to)
        );
        self.in_simulation = false;
        if self.move_piece(from, to) {
            println!("{self}");
            println!("Game over!");
            println!("Draw, half move clock expired");
            return true;
        }
        if self.checkmate() {
            println!("{self}");
            println!("Game over!");
            println!(
                "Player {} is in checkmate, {} wins!",
                self.current_player.number(),
                self.current_player.other()
            );
            return true;
        }
        return false;
    }

    fn checkmate(&mut self) -> bool {
        if !self.in_check(self.current_player) {
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
        for (i, j) in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
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
        self.winner = Some(self.current_player.other());
        true
    }

    #[cfg(test)]
    pub(crate) fn set_board(&mut self, board: Board) {
        self.board = board;
    }

    pub(crate) fn set_pieces(&mut self, p1_pieces: Vec<(u8, u8)>, p2_pieces: Vec<(u8, u8)>) {
        self.p1_pieces = p1_pieces;
        self.p2_pieces = p2_pieces;
    }

    pub fn save_model(&self) {
        self.model.as_ref().unwrap().save_model()
    }

    fn get_piece_scores(&mut self) -> i32 {
        if self.checkmate() {
            return if self.is_maximizing() {
                i32::MIN
            } else {
                i32::MAX
            }
        }
        if self.stalemate() || self.half_move_clock_expired() {
            return 0;
        }
        let mut score = 0;
        for &piece in &self.p1_pieces {
            score += self.get(piece).unwrap().value();
        }
        for &piece in &self.p2_pieces {
            score -= self.get(piece).unwrap().value();
        }
        score
    }

    fn is_current_player(&self, from: (u8, u8)) -> bool {
        // if let Some(spot) = self.get(from) {
        //     self.current_player == spot.player()
        // } else {
        //     false
        // }
        self.get(from).is_some_and(|spot| self.current_player == spot.player())
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

    pub(crate) fn square_is_opponent(&self, to: (u8, u8), player: Player) -> bool {
        // if let Some(piece) = self.get(to) {
        //     piece.player() != player
        // } else {
        //     false
        // }
        self.get(to).is_some_and(|piece| piece.player() != player)
    }

    pub(crate) fn take(&mut self, to: (u8, u8), new_piece: Square) {
        let piece = self.get(to).unwrap();
        // if piece.is_type::<King>() {
        //     println!("You can't take a king, something went wrong!");
        //     println!("{self}");
        //     println!("Took {:?}", to);
        //     panic!("You can't take a king, something went wrong!");
        // }
        assert!(
            !piece.is_type::<King>(),
            "You can't take a king, something went wrong!"
        );
        if !self.in_simulation {
            println!(
                "Player {} took {}'s {}!",
                self.current_player.number(),
                self.current_player.other(),
                piece.name()
            );
        }
        self.remove_piece(to);
        self.set(to, new_piece);
        let i = match piece.name() {
            "pawn" => 0,
            "rook" => 1,
            "bishop" => 2,
            "knight" => 3,
            "queen" => 4,
            _ => unreachable!("Invalid piece name!"),
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
        let mut taken_vec = vec![];
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
                if taken[i] > 1 {
                    taken_vec.push(format!("{}×{piece} ", taken[i]));
                } else {
                    taken_vec.push(format!("{piece} "));
                }
            }
        }
        if taken_vec.is_empty() {
            None
        } else {
            Some(taken_vec.join(", "))
        }
    }

    fn remove_piece(&mut self, position: (u8, u8)) {
        match self.get(position).unwrap().player() {
            Player::One => self.p1_pieces.retain(|&x| x != position),
            Player::Two => self.p2_pieces.retain(|&x| x != position),
        }
    }

    pub(crate) fn get(&self, (x, y): (u8, u8)) -> Square {
        // should I pass the position as reference?
        self.board[y as usize][x as usize].clone()
    }

    fn set(&mut self, (x, y): (u8, u8), piece: Square) {
        self.board[y as usize][x as usize] = piece;
    }

    pub(crate) fn in_check(&mut self, player: Player) -> bool {
        let king = self.get_king(player);
        for position in self.get_pieces(player.other()).clone() {
            let piece = self.get(position).unwrap();
            if piece.valid_move(position, king, self).is_valid() {
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
        println!("Enter a move or \"help\" to see more commands");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        if input.to_ascii_lowercase().trim() == "exit" {
            std::process::exit(0);
        } else if input.to_ascii_lowercase().trim() == "resign" {
            println!(
                "Player {} resigned, {} wins!",
                self.current_player.number(),
                self.current_player.other()
            );
            std::process::exit(0);
        } else if input.to_ascii_lowercase().trim() == "hint" {
            if self.allow_hints {
                let best_move = self.get_best_move().unwrap();
                println!("Hint, your best move is: {} -> {}", format_coord(&best_move.0), format_coord(&best_move.1));
            } else {
                println!("Hints are turned off!");
            }
            return self.get_move();
        } else if input.to_ascii_lowercase().trim() == "moves" {
            for position in self.get_pieces(self.current_player).clone() {
                let piece = self.get(position).expect("Piece not found! something went wrong");
                let piece_moves = piece.get_legal_moves(position, self);
                if !piece_moves.is_empty() {
                    println!("{} {} -> {}", piece.name(), format_coord(&position), piece_moves.iter().map(format_coord).collect::<Vec<String>>().join(", "));
                }
            }
            return self.get_move();
        } else if input.to_ascii_lowercase().trim() == "help" {
            println!("Commands:");
            println!("  a2      - display all possible moves for the piece at a4");
            println!("  a2 a4   - move the piece at a2 to a4");
            println!("  moves   - see all possible moves");
            println!("  hint    - get a hint for your next move");
            println!("  resign  - resign the game");
            println!("  exit    - exit the game");
            return self.get_move();
        }
        let mut input = input.split_whitespace();
        let from = input.next();
        let to = input.next();
        if from.is_none() {
            print!("Invalid input! ");
            return self.get_move();
        } else if to.is_none() {
            let from = from.unwrap().to_ascii_lowercase();
            let from = (from.chars().nth(0), from.chars().nth(1));
            if from.0.is_none() || from.1.is_none() {
                print!("Invalid input! ");
                return self.get_move();
            }
            let from = (
                from.0.unwrap() as i8 - 'a' as i8,
                '8' as i8 - from.1.unwrap() as i8,
            );
            if from.0 < 0 || from.0 > 7 || from.1 < 0 || from.1 > 7 {
                print!("Invalid input! ");
                return self.get_move();
            }
            let from = (from.0 as u8, from.1 as u8);
            self.see_all_moves(from);
            return self.get_move();
        }
        let from = from.unwrap().to_ascii_lowercase();
        let to = to.unwrap().to_ascii_lowercase();
        let from = (from.chars().nth(0), from.chars().nth(1));
        let to = (to.chars().nth(0), to.chars().nth(1));
        if from.0.is_none() || from.1.is_none() || to.0.is_none() || to.1.is_none() {
            print!("Invalid input! ");
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
        if from.0 < 0
            || from.1 < 0
            || from.0 > 7
            || from.1 > 7
            || to.0 < 0
            || to.1 < 0
            || to.0 > 7
            || to.1 > 7
        {
            print!("Invalid input! ");
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

    pub fn set_last_double(&mut self, position: Option<(u8, u8)>) {
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
            if moves.is_empty() {
                println!(
                    "Player {}'s {} has no legal moves!",
                    piece.player().number(),
                    piece.name()
                );
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
        let friendly_piece = friendly.clone().unwrap();
        let en_passent_capture = if let Some(pawn) = friendly_piece.get_piece::<Pawn>() {
            pawn.en_passent_capture(from, to, self)
        } else {
            None
        };
        let is_king = friendly_piece.is_type::<King>();
        let old = self.get(to);
        let enemy_pieces = self.get_pieces(player.other()).clone();
        // assumes that old isn't a friendly piece
        if old.is_some() {
            self.remove_piece(to);
        }
        if let Some(captured) = en_passent_capture {
            self.remove_piece(captured);
            self.set(captured, None);
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
        if let Some(captured) = en_passent_capture {
            self.set(captured, Some(Box::new(Pawn::new(player.other()))));
        }
        self.set_player_pieces(player.other(), enemy_pieces);
        still_in_check
    }

    pub(crate) fn set_king(&mut self, player: Player, king: (u8, u8)) {
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
                self.p1_pieces.retain(|&x| x != from);
                self.p1_pieces.push(to);
                // self.p2_pieces.retain(|x| *x != to);
            }
            Player::Two => {
                self.p2_pieces.retain(|&x| x != from);
                self.p2_pieces.push(to);
                // self.p1_pieces.retain(|x| *x != to);
                // should this be here or in take
            }
        }
    }

    pub(crate) fn set_player_pieces(&mut self, player: Player, pieces: Vec<(u8, u8)>) {
        match player {
            Player::One => self.p1_pieces = pieces,
            Player::Two => self.p2_pieces = pieces,
        }
    }
    
    fn stalemate(&mut self) -> bool {
        self.get_possible_moves(self.current_player).is_empty()
    }

    pub fn get_possible_moves(&mut self, player: Player) -> Vec<((u8, u8), (u8, u8))> {
        let mut moves = Vec::new();
        for position in self.get_pieces(player).clone() {
            let piece = self.get(position).expect("Piece not found!");
            let piece_moves = piece.get_legal_moves(position, self);
            moves.extend(piece_moves.iter().map(|&x| (position, x)));
        }
        moves
    }

    fn tick(&mut self) -> bool {
        if self.is_last_halfmove() && !self.in_simulation {
            println!("The halfmove clock is nearly up! Next move must be a capture or pawn move.");
        } else if self.half_move_clock_expired() && !self.in_simulation {
            println!("{HALF_MOVE_LIMIT} moves without a capture or pawn move, it's a draw!");
            return true;
        }
        self.half_move_clock += 1;
        self.full_move_clock += 1;
        false
    }

    fn evaluate_moves(&mut self, moves: &Vec<((u8, u8), (u8, u8))>) -> Vec<f32> {
        // TODO: optimize by not cloning the entire game?
        // } if self.rl_training {
        self.model
            .clone()
            .unwrap()
            .run_inference(&moves.iter().map(|&(from, to)| {
                let mut game = self.clone();
                game.move_piece(from, to);
                game.to_matrix()
            }).collect())
            .unwrap()
        // } else if let Some(model) = self.model {
        //     // check if par_iter is actually faster
        //     let games = moves
        //         .iter()
        //         .map(|&(from, to)| {
        //             let mut game = self.clone();
        //             game.move_piece(from, to);
        //             game
        //             // self.matrix_from_move(mov)
        //     });

        //     let fens: Vec<String> = games.clone().map(|mut game| {
        //         game.to_short_fen()
        //     }).collect();

        //     let mut cached = Vec::with_capacity(moves.len());
        //     for (i, fen) in fens.iter().enumerate() {
        //         if let Some(eval) = self.cache.get(fen) {
        //             cached.push((i, eval));
        //         }
        //     }

        //     let matrices: Vec<[[[f32; 8]; 8]; 13]> = games.clone().map(|mut game| {
        //         game.to_matrix()
        //     }).collect();

        //     if cached.len() == 0 {
        //         let evals = self.model.as_ref().unwrap().run_inference(&matrices).unwrap();
        //         (0..evals.len()).for_each(|i| {
        //             self.cache.insert(fens[i].clone(), evals[i]);
        //         });
        //         return evals;
        //     }

        //     let mut j = 0;
        //     let uncached_games = matrices
        //         .into_iter()
        //         .enumerate()
        //         .filter(|(i, _)| {
        //             if j < cached.len() && cached[j].0 == *i {
        //                 j += 1;
        //                 false
        //             } else {
        //                 true
        //             }
        //         })
        //         .map(|(_, game)| game)
        //         .collect();

        //     let evals = self
        //         .model
        //         .run_inference(&uncached_games)
        //         .unwrap();

        //     j = 0;
        //     for i in 0..moves.len() {
        //         if j < cached.len() && cached[j].0 == i {
        //             j += 1;
        //             continue;
        //         }
        //         self.cache.clone().insert(fens[i].clone(), evals[i - j]);
        //     }

        //     j = 0;
        //     games
        //         .enumerate()
        //         .map(|(i, _)| {
        //             if j < cached.len() && cached[j].0 == i {
        //                 j += 1;
        //                 cached[j].1
        //             } else {
        //                 &evals[i - j]
        //             }
        //         }).cloned().collect()
        // }
        // // self.model.clone().unwrap().run_inference(games).unwrap()
    }

    // fn matrix_from_move(&mut self, &(from, to): &((u8, u8), (u8, u8))) -> Matrix {
    //     if self.player_in_check() {
    //         println!("You're in check!");
    //     }
    //     let half_move_clock = self.half_move_clock;
    //     let full_move_clock = self.full_move_clock;
    //     assert!(self.tick(), "Half move clock is up, draw!");
    //     assert!(self.is_current_player(from));
    //     let mut half_move = false;
    //     let piece = self.get(from);
    //     let conquered = self.get(to);
    //     let move_status = piece.clone().unwrap().valid_move(from, to, self);
    //     assert!(move_status.is_valid());
    //     // let board: Board = self.board.clone();
    //     if let Some(conquered) = conquered.clone() {
    //         assert!(conquered.player() != self.current_player);
    //         half_move = true;
    //         self.take(to, piece.clone());
    //         self.set(from, None);
    //     } else {
    //         self.set(to, piece.clone());
    //         self.set(from, None);
    //     }
    //     let (p1_pieces, p2_pieces) = (
    //         self.get_pieces(Player::One).clone(),
    //         self.get_pieces(Player::Two).clone(),
    //     );
    //     self.set_moved(piece.clone(), from, to);
    //     assert!(
    //         !self.player_in_check(),
    //         "Wait you can't put yourself in check! go again."
    //     );
    //     let last_double = self.last_double;
    //     self.set_last_double(None);
    //     let mut en_passant_piece = None;
    //     match move_status {
    //         Move::Normal => (),
    //         Move::Double(position) => {
    //             self.set_last_double(Some(position));
    //         }
    //         Move::Castle => {
    //             self.castle(to);
    //         }
    //         Move::EnPassant(position) => {
    //             en_passant_piece = self.get(position);
    //             self.take(position, None);
    //         }
    //         Move::Invalid => unreachable!(),
    //     }
    //     let mut promoted = false;
    //     if piece.clone().unwrap().is_type::<Pawn>() {
    //         half_move = true;
    //         if to.1 == 7 || to.1 == 0 {
    //             self.promote_piece(to);
    //             promoted = true;
    //         }
    //     }
    //     self.set_moved(piece.clone(), from, to);
    //     if half_move {
    //         self.half_move_clock = 0;
    //     }
    //     let matrix = self.to_matrix();
    //     // self.set_board(board);
    //     self.half_move_clock = half_move_clock;
    //     self.full_move_clock = full_move_clock;
    //     self.set(from, self.get(to));
    //     self.set(to, conquered);
    //     match move_status {
    //         Move::Normal => (),
    //         Move::Double(_) => {
    //             self.set_last_double(last_double);
    //         }
    //         Move::Castle => {
    //             self.castle(to);
    //             let (rook, rook_to) = match to {
    //                 (2, y) => ((3, y), (0, y)),
    //                 (6, y) => ((5, y), (7, y)),
    //                 _ => unreachable!(),
    //             };
    //             self.set(rook_to, self.get(rook));
    //             self.set(rook, None);
    //         }
    //         Move::EnPassant(position) => self.set(position, en_passant_piece),
    //         Move::Invalid => unreachable!(),
    //     }
    //     if promoted {
    //         self.set(to, piece)
    //     }
    //     self.set_pieces(p1_pieces, p2_pieces);
    //     matrix
    // }

    // fn piece_score_from_move(&mut self, &(from, to): &((u8, u8), (u8, u8))) -> i32 {
    //     if self.player_in_check() {
    //         println!("You're in check!");
    //     }
    //     let half_move_clock = self.half_move_clock;
    //     let full_move_clock = self.full_move_clock;
    //     assert!(!self.tick(), "Half move clock is up, draw!");
    //     assert!(self.is_current_player(from));
    //     let mut half_move = false;
    //     let piece = self.get(from);
    //     let conquered = self.get(to);
    //     let move_status = piece.clone().unwrap().valid_move(from, to, self);
    //     assert!(move_status.is_valid());
    //     // let board: Board = self.board.clone();
    //     if let Some(conquered) = conquered.clone() {
    //         assert!(conquered.player() != self.current_player);
    //         half_move = true;
    //         self.take(to, piece.clone());
    //         self.set(from, None);
    //     } else {
    //         self.set(to, piece.clone());
    //         self.set(from, None);
    //     }
    //     let (p1_pieces, p2_pieces) = (
    //         self.get_pieces(Player::One).clone(),
    //         self.get_pieces(Player::Two).clone(),
    //     );
    //     self.set_moved(piece.clone(), from, to);
    //     assert!(
    //         !self.player_in_check(),
    //         "Wait you can't put yourself in check! go again."
    //     );
    //     // if self.player_in_check() {
    //     //     println!("Wait you can't put yourself in check!");
    //     //     println!("{self}");
    //     //     println!("Moved from {:?} to {:?}", from, to);
    //     //     panic!("Wait you can't put yourself in check!");
    //     // }
    //     let last_double = self.last_double;
    //     self.set_last_double(None);
    //     let mut en_passant_piece = None;
    //     match move_status {
    //         Move::Normal => (),
    //         Move::Double(position) => {
    //             self.set_last_double(Some(position));
    //         }
    //         Move::Castle => {
    //             self.castle(to);
    //         }
    //         Move::EnPassant(position) => {
    //             en_passant_piece = self.get(position);
    //             self.take(position, None);
    //         }
    //         Move::Invalid => unreachable!(),
    //     }
    //     let mut promoted = false;
    //     if piece.clone().unwrap().is_type::<Pawn>() {
    //         half_move = true;
    //         if to.1 == 7 || to.1 == 0 {
    //             self.promote_piece(to);
    //             promoted = true;
    //         }
    //     }
    //     self.set_moved(piece.clone(), from, to);
    //     if half_move {
    //         self.half_move_clock = 0;
    //     }
    //     let score = self.get_piece_scores();
    //     // self.set_board(board);
    //     self.half_move_clock = half_move_clock;
    //     self.full_move_clock = full_move_clock;
    //     self.set(from, self.get(to));
    //     self.set(to, conquered);
    //     match move_status {
    //         Move::Normal => (),
    //         Move::Double(_) => {
    //             self.set_last_double(last_double);
    //         }
    //         Move::Castle => {
    //             self.castle(to);
    //             let (rook, rook_to) = match to {
    //                 (2, y) => ((3, y), (0, y)),
    //                 (6, y) => ((5, y), (7, y)),
    //                 _ => unreachable!(),
    //             };
    //             self.set(rook_to, self.get(rook));
    //             self.set(rook, None);
    //         }
    //         Move::EnPassant(position) => self.set(position, en_passant_piece),
    //         Move::Invalid => unreachable!(),
    //     }
    //     if promoted {
    //         self.set(to, piece)
    //     }
    //     self.set_pieces(p1_pieces, p2_pieces);
    //     score
    // }

    // fn get_moves_sorted(&mut self, descending: bool) -> Vec<((u8, u8), (u8, u8))> {
    //     let moves = self.get_possible_moves(self.current_player);
    //     let evals = self.evaluate_moves(&moves);
    //     let mut move_evals = moves
    //         .into_iter()
    //         .zip(evals.into_iter())
    //         .collect::<Vec<(((u8, u8), (u8, u8)), f32)>>();
    //     if descending {
    //         move_evals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    //     } else {
    //         move_evals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    //     }
    //     move_evals
    //         .iter()
    //         .map(|&(mov, _)| mov)
    //         .take(SEARCH_BREADTH)
    //         .collect::<Vec<((u8, u8), (u8, u8))>>()
    // }

    #[cfg(test)]
    pub(crate) fn set_player(&mut self, player: Player) {
        self.current_player = player;
    }

    pub(crate) fn is_last_halfmove(&self) -> bool {
        self.half_move_clock == HALF_MOVE_LIMIT - 1
    }

    fn half_move_clock_expired(&self) -> bool {
        self.half_move_clock >= HALF_MOVE_LIMIT
    }

    pub fn to_fen(&mut self) -> String {
        let mut fen = self.to_short_fen();
        fen.push(' ');

        // fen.push((self.half_move_clock + '0' as u8) as char);
        // fen.push(self.half_move_clock.to_string().chars().next().unwrap());
        fen.extend(self.half_move_clock.to_string().chars());
        fen.push(' ');
        
        // fen.push((self.full_move_clock + '0' as u8) as char);
        // self.full_move_clock.to_string().chars().for_each(|c| fen.push(c));
        fen.extend(self.full_move_clock.to_string().chars());
        fen
    }

    pub fn to_short_fen(&self) -> String {
        let mut fen = String::with_capacity(87);
        for row in &self.board {
            let mut num_empty = 0;
            for square in row {
                if let Some(piece) = square {
                    if num_empty > 0 {
                        fen.push((num_empty + '0' as u8) as char);
                    }
                    num_empty = 0;
                    match (piece.name(), piece.player()) {
                        ("rook", Player::Two) => fen.push('r'),
                        ("knight", Player::Two) => fen.push('n'),
                        ("bishop", Player::Two) => fen.push('b'),
                        ("king", Player::Two) => fen.push('k'),
                        ("queen", Player::Two) => fen.push('q'),
                        ("pawn", Player::Two) => fen.push('p'),
                        ("rook", Player::One) => fen.push('R'),
                        ("knight", Player::One) => fen.push('N'),
                        ("bishop", Player::One) => fen.push('B'),
                        ("king", Player::One) => fen.push('K'),
                        ("queen", Player::One) => fen.push('Q'),
                        ("pawn", Player::One) => fen.push('P'),
                        (_, _) => unreachable!("invalid piece player combos"),
                    }
                } else {
                    num_empty += 1
                }
            }
            if num_empty > 0 {
                fen.push((num_empty + '0' as u8) as char)
            }
            fen.push('/');
        }
        fen.pop();
        fen.push(' ');
        fen.push(match self.current_player {
            Player::One => 'w',
            Player::Two => 'b',
        });
        fen.push(' ');
        
        let mut castleable = false;
        if self.has_p1_king_moved {
            if self.has_p1_left_rook_moved {
                fen.push('K');
                castleable = true;
            }
            if self.has_p1_right_rook_moved {
                fen.push('Q')
            }
        }
        if self.has_p2_king_moved {
            if self.has_p2_left_rook_moved {
                fen.push('k');
                castleable = true;
            }
            if self.has_p2_right_rook_moved {
                fen.push('q');
                castleable = true;
            }
        }
        if !castleable {
            fen.push('-');
        }
        fen.push(' ');

        if let Some(last_double) = self.last_double {
            for c in format_coord(&last_double).bytes() {
                fen.push(c as char);
            }
        } else {
            fen.push('-')
        }
        fen
    }

    fn update_epsilon(&mut self) {
        self.epsilon *= self.epsilon_decay_rate;
    }

    pub fn winner(&mut self) -> Option<Player> {
        self.winner
    }

    pub fn is_maximizing(&self) -> bool {
        self.current_player.is_maximizing()
    }

    // pub fn from_fen(fen: String) -> Self {
    //     Self {
    //         board: vec![vec![None; 8]; 8],
    //         p1_pieces: Vec::new(),
    //         p2_pieces: Vec::new(),
    //         p1_taken: [0;5],
    //         p2_taken: [0;5],
    //         king_one: (0, 0),
    //         king_two: (0, 0),
    //         current_player: Player::One,
    //         last_double: None,
    //         has_p1_king_moved: false,
    //         has_p1_left_rook_moved: false,
    //         has_p1_right_rook_moved: false,
    //         has_p2_king_moved: false,
    //         has_p2_left_rook_moved: false,
    //         has_p2_right_rook_moved: false,
    //         half_move_clock: 0,
    //         full_move_clock: 1,
    //         in_simulation: false,
    //         two_player: false,
    //         model: None,
    //         computer_player: None
    //     }
    // }
}

impl<'a> Display for Game<'a> {
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

fn format_coord(coordinate: &(u8, u8)) -> String {
    format!(
        "{}{}",
        (coordinate.0 + 'a' as u8) as char,
        ('8' as u8 - coordinate.1) as char
    )
}

fn setup_board() -> Board {
    let mut board: Board = vec![vec![None; 8]; 8];
    board[0] = vec![
        Some(Box::new(Rook::new(Player::Two))),
        Some(Box::new(Knight::new(Player::Two))),
        Some(Box::new(Bishop::new(Player::Two))),
        Some(Box::new(Queen::new(Player::Two))),
        Some(Box::new(King::new(Player::Two))),
        Some(Box::new(Bishop::new(Player::Two))),
        Some(Box::new(Knight::new(Player::Two))),
        Some(Box::new(Rook::new(Player::Two))),
    ];
    board[1] = vec![Some(Box::new(Pawn::new(Player::Two))); 8];
    board[7] = vec![
        Some(Box::new(Rook::new(Player::One))),
        Some(Box::new(Knight::new(Player::One))),
        Some(Box::new(Bishop::new(Player::One))),
        Some(Box::new(Queen::new(Player::One))),
        Some(Box::new(King::new(Player::One))),
        Some(Box::new(Bishop::new(Player::One))),
        Some(Box::new(Knight::new(Player::One))),
        Some(Box::new(Rook::new(Player::One))),
    ];
    board[6] = vec![Some(Box::new(Pawn::new(Player::One))); 8];
    board
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

        let mut game = Game::two_player_game(false);
        game.set_board(board);
        game.set_pieces(vec![(0, 0)], vec![(1, 0), (0, 1)]);
        game.set_king(Player::One, (0, 0));

        assert!(game.checkmate());
    }

    #[test]
    fn checkmate_no_friendlies2() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Queen::new(Player::Two)));
        board[1][0] = Some(Box::new(Rook::new(Player::Two)));
        board[0][2] = Some(Box::new(Queen::new(Player::Two)));

        let mut game = Game::two_player_game(false);
        game.set_board(board);
        game.set_pieces(vec![(0, 0)], vec![(1, 0), (0, 1), (2, 0)]);
        game.set_king(Player::One, (0, 0));

        print!("{game}");

        assert!(game.checkmate());
    }

    #[test]
    fn no_checkmate_no_friendlies() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Rook::new(Player::Two)));
        board[1][0] = Some(Box::new(Rook::new(Player::Two)));

        let mut game = Game::two_player_game(false);
        game.set_board(board);
        game.set_pieces(vec![(0, 0)], vec![(1, 0), (0, 1)]);
        game.set_king(Player::One, (0, 0));

        print!("{game}");

        assert!(!game.checkmate());
    }

    #[test]
    fn no_checkmate_blockable() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[0][0] = Some(Box::new(King::new(Player::One)));
        board[0][1] = Some(Box::new(Knight::new(Player::One)));
        board[2][0] = Some(Box::new(Queen::new(Player::Two)));

        let mut game = Game::two_player_game(false);
        game.set_board(board);
        game.set_pieces(vec![(0, 0), (1, 0)], vec![(0, 2)]);
        game.set_king(Player::One, (0, 0));

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

        let mut game = Game::two_player_game(false);
        game.set_board(board);
        game.set_pieces(vec![(0, 0), (1, 0), (1, 1)], vec![(0, 3)]);
        game.set_king(Player::One, (0, 0));

        assert!(game.checkmate());
    }

    #[test]
    fn checking_all_legal_moves_are_valid() {
        let mut game = Game::two_player_game(false);

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

        let mut game = Game::two_player_game(false);
        game.set_board(board);
        game.set_pieces(vec![(5, 7)], vec![(4, 4), (3, 4), (3, 2)]);
        game.set_king(Player::One, (5, 7));
        game.set_king(Player::Two, (3, 2));

        println!("{game}");

        assert!(game.stalemate());
        assert!(!game.checkmate());
    }

    #[test]
    fn move_into_check() {
        let mut board: Board = vec![vec![None; 8]; 8];
        board[1][4] = Some(Box::new(King::new(Player::Two)));
        board[0][4] = Some(Box::new(Queen::new(Player::Two)));
        board[2][4] = Some(Box::new(Pawn::new(Player::Two)));
        board[2][5] = Some(Box::new(Pawn::new(Player::Two)));
        board[3][6] = Some(Box::new(Bishop::new(Player::One)));
        board[7][3] = Some(Box::new(King::new(Player::One)));

        let mut game = Game::two_player_game(false);
        game.set_board(board);
        game.set_pieces(vec![(3, 7), (6, 3)], vec![(4, 2), (4, 1), (5, 2), (4, 0)]);
        game.set_king(Player::Two, (4, 1));
        game.set_player(Player::Two);

        println!("{game}");

        let pawn = game.get((5, 2)).unwrap();

        println!("{:?}", pawn.get_legal_moves((5, 2), &mut game));

        assert!(!pawn.get_legal_moves((5, 2), &mut game).contains(&(5, 3)))
    }

    #[test]
    fn fen() {
        let mut game = Game::two_player_game(false);
        game.set_last_double(Some((3,3)));
        println!("{}", game.to_fen());
    }

    #[test]
    fn en_passant_into_check() {
        let mut board: Board = vec![vec![None;8];8];
        board[3][6] = Some(Box::new(King::new(Player::One)));
        board[3][2] = Some(Box::new(Pawn::new(Player::One)));
        board[5][7] = Some(Box::new(Pawn::new(Player::One)));
        board[0][6] = Some(Box::new(King::new(Player::Two)));
        board[4][7] = Some(Box::new(Pawn::new(Player::Two)));
        board[1][5] = Some(Box::new(Pawn::new(Player::Two)));
        board[4][0] = Some(Box::new(Pawn::new(Player::Two)));
        board[3][3] = Some(Box::new(Pawn::new(Player::Two)));
        board[3][0] = Some(Box::new(Rook::new(Player::Two)));
        board[5][2] = Some(Box::new(Rook::new(Player::Two)));

        let mut game = Game::self_play(None, Some(4), false, None, None, None);
        game.set_pieces(vec![(6,3), (2, 3), (7,5)], vec![(6,0),(7,4),(5,1),(0,4), (3,3), (0,3), (2,5)]);
        game.set_last_double(Some((3,3)));
        game.set_board(board);
        game.set_king(Player::One, (6,3));
        game.set_king(Player::Two, (6,0));
        game.set_player(Player::Two);

        println!("{game}");

        let pawn = game.get((2,3)).unwrap();
        let moves = pawn.get_legal_moves((2,3), &mut game);
        println!("{:?}", moves);

        assert!(!moves.contains(&(3,2)));
    }

    #[test]
    fn pieces_add_up() {
        let mut game = Game::self_play(None, Some(2), false, None, None, None);
        let mut game_over = false;
        while !game_over {
            game_over = game.turn();
            game.assert_pieces();
        }
    }
}