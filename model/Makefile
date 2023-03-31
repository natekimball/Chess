train:
	sbatch train.slurm

clean:
	rm *.err *.out

onnx:
	python -m tf2onnx.convert --saved-model tf_saved_model --opset 17 --output model.onnx; \
	python convert.py

# find . -name "*.out" | while read file; do fgrep val_loss "$file" | tail -1; done
# find . -name "*.out" | while read file; do if fgrep val_loss "$file" >nul; then echo "$file"; fi; done

.PHONY: clean