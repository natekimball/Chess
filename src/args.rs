use clap::{Args, Parser, Subcommand};

/// Chess game for two-player, single-player, and reinforcement learning
#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct ChessArgs {
    #[clap(subcommand)]
    pub game_type: GameType,
}

#[derive(Subcommand, Debug)]
pub enum GameType {
    /// Two-player mode
    TwoPlayer(TwoPlayerArgs),
    
    /// Single-player mode
    SinglePlayer(OnePlayerArgs),
    
    /// Self-play reinforcement learning
    SelfPlay(SelfPlayArgs),
}

#[derive(Args, Debug)]
pub struct TwoPlayerArgs {
    /// allow players to get best move hints
    #[arg(short, long, default_value_t = false)]
    pub allow_hints: bool
}

#[derive(Args, Debug)]
pub struct OnePlayerArgs {
    /// play as black 
    #[arg(short, long, default_value_t = false)]
    pub black: bool,
    
    /// use heuristic evaluation function in minimax
    #[arg(short = 'p', long, default_value_t = false)]
    pub heuristic: bool,
    
    /// search depth for minimax algorithm
    #[arg(short = 'd', long = "depth")]
    pub search_depth: Option<u8>,
    
    /// directory for evaluation model
    #[arg(long)]
    pub model_dir: Option<String>
}

#[derive(Args, Debug)]
pub struct SelfPlayArgs {
    
    // TODO: depricated
    /// use heuristic evaluation function in minimax
    #[arg(short = 'p', long, default_value_t = false)]
    pub heuristic: bool,
    
    /// directory for evaluation model
    #[arg(long)]
    pub model_dir: Option<String>,
    
    /// number of games to play in self-play mode
    #[arg(short, long, default_value_t = 1)]
    pub num_games: u16,
    
    /// search depth for minimax algorithm
    #[arg(short = 'd', long = "depth")]
    pub search_depth: Option<u8>,

    /// use epsilon-greedy algorithm for reinforcement learning
    #[arg(short, long)]
    pub epsilon_greedy: bool,
    
    /// epsilon decay rate for epsilon-greedy algorithm
    #[arg(long = "decay")]
    pub epsilon_decay: Option<f64>
}

// #[derive(Parser, Debug)]
// #[command(author, version, about, long_about = None)]
// pub struct ChessArgs {
//     /// Two-player mode
//     #[arg(short, long, default_value_t = false)]
//     pub two_player: bool,

//     #[arg(short, long, default_value_t = false)]
//     pub allow-hints: bool,

//     /// Start second in single-player mode 
//     #[arg(short, long, default_value_t = false)]
//     pub black: bool,

//     /// Use heuristic evaluation function in single-player mode
//     #[arg(long, default_value_t = false)]
//     pub heuristic: bool,

//     /// Search depth for minimax algorithm
//     #[arg(long)]
//     pub search_depth: Option<u8>,

//     /// Self-play reinforcement learning
//     #[arg(short, long)]
//     pub self_play: bool,

//     /// Save directory for model
//     #[arg(long, default_value_t = String::from("model/model_v4_w_sigs"))]
//     pub save_dir: String,

//     /// Number of games to play in self-play mode
//     #[arg(short, long, default_value_t = 1)]
//     pub num_games: u16,

//     /// Use epsilon-greedy algorithm for reinforcement learning
//     #[arg(short, long)]
//     pub epsilon_greedy: bool,

//     /// Epsilon decay rate for epsilon-greedy algorithm
//     #[arg(long)]
//     pub epsilon_decay: Option<f64>,
// }