// extern crate tensorflow;
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
use game::Game;

fn main() {
    let mut play_again = true;
    while play_again {
        play_again = launch_game();
    }
}

fn launch_game() -> bool {
    let mut game = Game::new();

    while !game.is_over() {
        // print!("\x1b[120S\x1b[1;1H");
            // print!("\x1B[2J\x1B[1;1H");
        // Command::new(if cfg!(target_os = "windows") {"cls"} else {"clear"}).status().unwrap();
        game.turn();
    }
    
    
    game.play_again()
}