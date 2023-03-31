train:
	sbatch train.slurm

clean:
	rm *.err *.out

onnx:
	python -m tf2onnx.convert --saved-model tf_saved_model --opset 17 --output model.onnx; \
	python convert.py

results:
	find . -name "*.out" | while read file; \
	do if fgrep val_loss "$$file" >nul; \
	then echo $$file; \
	fgrep "EDT 2023" $$file | head -1; \
	grep val_loss $$file; \
	fgrep Epoch $$file | tail -1; \
	tail -1 $$file; fi; \
	done; echo

# fgrep -E val_loss -E Epoch $$file
# find . -name "*.out" | while read file; do if fgrep val_loss "$file" >nul; then echo "$file" fi; done
# find . -name "*.out" | while read file; do fgrep val_loss "$file" | tail -1; done

.PHONY: train clean onnx results