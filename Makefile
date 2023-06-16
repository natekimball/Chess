run-rivanna:
	module load gcc/11.2.0 rust/1.66.1; \
	cargo run --release

job:
	ijob -c 1 -A bii_dsc_community -p standard --time=01:00:00 --partition=bii-gpu --mem-per-cpu=256G --gres=gpu:v100

play-ai:
	cargo run --release -- --depth 2

play-heuristics:
	cargo run --release -- --heuristics --depth 4

two-player:
	cargo run --release -- --2p

train:
	python model/train.py

rl-train:
	cargo run --release -- --self-play --depth 2

test:
	cargo test

saved_model:
	scp -r tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/$@ /home/nkimball/Projects/rust_projects/chess/model

transfer:
	scp -r tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/model_v5 /home/nkimball/Projects/rust_projects/chess/model; \

# scp tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/model.* /home/nkimball/Projects/rust_projects/chess/model