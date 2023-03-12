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
// mod chess_ai;

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
        // if cfg!(target_os = "windows") {
        //     std::process::Command::new("cls").status().unwrap();
        // } else {
        //     std::process::Command::new("clear").status().unwrap();
        // }
        print!("\x1b[200S\x1b[1H");
        game.turn();
    }
    
    game.play_again()
}