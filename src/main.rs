mod game;
mod piece;
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
        game.turn();
    }

    game.play_again()
}