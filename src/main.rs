extern crate tensorflow;
mod game;
mod piece;
mod king;
mod queen;
mod rook;
mod bishop;
mod knight;
mod pawn;
mod player;
mod model;


// use std::process::Command;
use std::env::args;
use game::Game;
use model::Model;
use player::Player;

fn main() {
    let args: Vec<String> = args().collect();
    let two_player = args.contains(&"--2p".to_string());
    let computer_player = if args.contains(&String::from("--black")) {Some(Player::Two)} else {Some(Player::One)};
    let self_play = args.contains(&String::from("--self-play"));
    let heuristic = args.contains(&String::from("--heuristic"));
    let search_depth = if args.contains(&String::from("--depth")) {
        // TODO: not working
        Some(args[args.iter().position(|x| x == "--depth").unwrap() + 1].parse::<u8>().unwrap())
    } else {
        None
    };
    let save_dir = if args.contains(&String::from("--save-dir")) {
        args[args.iter().position(|x| x == "--save-dir").unwrap() + 1].as_str()
    } else {
        "model/saved_model_t"
    };
    if heuristic && self_play {
        return heuristic_self_play(search_depth);
    }
    
    let model = if two_player || heuristic { None } else { Some(Model::new(save_dir)) };
    if self_play {
        let num_games = if args.contains(&String::from("--num-games")) {args[args.iter().position(|x| x == "--num-games").unwrap() + 1].parse::<usize>().unwrap()} else {1};
        reinforcement_learning(num_games, &model, search_depth);
    } else {
        let mut play_again = true;
        while play_again {
            play_again = launch_game(two_player, computer_player, &model, search_depth);
        }
    }
}

fn launch_game(two_player: bool, computer_player: Option<Player>, model: &Option<Model>, search_depth: Option<u8>) -> bool {
    let mut game = if two_player {
        Game::two_player_game()
    } else {
        Game::single_player_game(computer_player, model, search_depth)
    };

    let mut game_over = false;
    while !game_over {
        // print!("\x1b[120S\x1b[1;1H");
        // print!("\x1B[2J\x1B[1;1H");
        // Command::new(if cfg!(target_os = "windows") {"cls"} else {"clear"}).status().unwrap();
        game_over = game.turn();
    }
    
    
    game.play_again()
}

fn reinforcement_learning(num_games: usize, model: &Option<Model>, search_depth: Option<u8>) {
    let mut game = Game::self_play(model, search_depth);

    for _ in 0..num_games {
        let mut game_over = false;
        while !game_over {
            game_over = game.turn();
        }
        game.save_model();
    }
}

fn heuristic_self_play(search_depth: Option<u8>) {
    let mut game = Game::self_play(&None, search_depth);
    let mut game_over = false;
    while !game_over {
        game_over = game.turn();
    }
}