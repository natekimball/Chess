# Chess

AI Chess game written in Rust complete with all the rules of chess, including castling, en passant, pawn promotion, check/checkmate detection, and the fifty-move rule. The game allows you to play against another player or our custom AI chess algorithm.

## Play

### AI gameplay (to play against AI, you must first train the model, see below)

```shell
cargo run --release -- --depth d
```

set d to be the depth of the search or number of future moves the algorithm will evaluate

### Two-player gameplay

```shell
cargo run -- --2p
```

## Algorithm Design

First, the model was pre=trained on stockfish evaluations, to build a model that could roughly evaluate board states and thus Q values. To make decisions, the algorithm performs an augmented mini-max tree search with alpha-beta pruning to a depth of 3 moves. For efficiency, my algorithm only searches moves that have a high probability of success as determined by the move's evaluation. The model was further trained via the reinforcement learning technique called amplification, where the model is trained on its own output after performing a mini-max search. This guarantees convergence on game theory optimal strategy, because as the model improves, its amplified self will also improve.

## Pre-training

```shell
python model/train.py
```

## Reinforcement learning

```shell
cargo run -- --self-play --num-games n --depth m --epsilon-greedy
```

## Training the model on Rivanna

### Pre-training (Rivanna)
  
```shell
sbatch model/train.slurm
```

### Reinforcement learning (Rivanna)

```shell
sbatch model/rl-training.slurm
```

