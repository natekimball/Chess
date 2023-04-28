run-rivanna:
	module load gcc/11.2.0; \
	module load rust/1.66.1; \
	cargo run --release

job:
	ijob -c 1 -A bii_dsc_community -p standard --time=01:00:00 --partition=bii-gpu --mem-per-cpu=256G --gres=gpu:v100

run:
	cargo run --release

train:
	python model/train.py

test:
	cargo test

# %.pb:
# 	scp tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/$@ /home/nkimball/Projects/rust_projects/chess/model

saved_model:
	scp -r tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/$@ /home/nkimball/Projects/rust_projects/chess/model

# %.pt:
# 	scp tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/$@ /home/nkimball/Projects/rust_projects/chess/model

# %.onnx:
# 	scp tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/$@ /home/nkimball/Projects/rust_projects/chess/model

transfer:
	scp -r tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/tf_saved_model /home/nkimball/Projects/rust_projects/chess/model; \

# scp tma5gv@rivanna.hpc.virginia.edu:/scratch/tma5gv/chess/model/model.* /home/nkimball/Projects/rust_projects/chess/model
