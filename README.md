# Chess

AI Chess game written in Rust complete with all the rules of chess, including castling, en passant, pawn promotion, check/checkmate detection, and the fifty-move rule. The game allows you to play against another player or our custom AI chess algorithm.

## Play

### AI gameplay (to play against AI, you must first train the model, see below)

```shell
cargo run --release -- single-player --depth d
```

set d to be the depth of the search or number of future moves the algorithm will evaluate

### Play against heuristic algorithm

```shell
cargo run --release -- single-player --heuristic --depth d
```

### Two-player gameplay

```shell
cargo run -- two-player
```

## Help

```shell
cargo run
cargo run -- <subcommand> --help
```

## Algorithm Design

First, the model was pre-trained on stockfish evaluations, to build a model that could roughly evaluate board states and thus Q values. To make decisions, the algorithm performs a multithreaded mini-max tree search with alpha-beta pruning to a depth of \<d> moves. A higher search depth leads to a better adversary, but more compute intensive decision making. The model was further trained via the reinforcement learning technique called amplification, where the model is trained on its own output after performing a mini-max search. This guarantees convergence on game theory optimal strategy, because as the model improves, its amplified self will also improve.

## Environment set-up

```shell
git clone https://github.com/natekimball/chess # or git@github.com:natekimball/chess.git
cd model
python -m venv ENV
source ENV/bin/activate
pip install -r ../requirements.txt
```

## Pre-training

```shell
python model/train.py
```

## Reinforcement learning

```shell
cargo run -- self-play --num-games n --depth m --epsilon-greedy
```

## Training the model on Rivanna HPC

### Environment set-up (Rivanna)

```shell
sbatch model/environment.slurm
```

### Pre-training (Rivanna)
  
```shell
sbatch model/train.slurm
```

### Reinforcement learning (Rivanna)

```shell
sbatch model/rl-training.slurm
```

### Play (Rivanna)

```shell
module purge
module load gcc/11.2.0 rust/1.66.1
cargo run --release -- single-player
```
