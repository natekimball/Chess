train:
	sbatch train.slurm

clean:
	rm *.err *.out

results:
	find . -name "*.out" | while read file; \
		do if [[ $$(fgrep val_loss $$file) ]]; \
		then \
			echo $$file; \
			fgrep "EDT 2023" $$file | head -1; \
			grep val_loss $$file; \
			fgrep Epoch $$file | tail -1; \
			tail -1 $$file; \
			echo; \
		fi; \
	done; \
	echo

result:
	find . -name "*.out" | while read file; \
		do if [[ $$(tail -1 $$file | fgrep loss) ]]; \
		then \
			echo $$file; \
			fgrep "EDT 2023" $$file | head -1; \
			grep val_loss $$file; \
            tail -2f $$file; \
        fi; \
    done; \
    echo

squeue:
	squeue -u $$USER

scancel:
	scancel -u $$USER

onnx:
	python -m tf2onnx.convert --saved-model tf_saved_model --opset 17 --output model.onnx; \
	python convert.py

# fgrep -E val_loss -E Epoch $$file
# find . -name "*.out" | while read file; do if fgrep val_loss "$file" >nul; then echo "$file" fi; done
# find . -name "*.out" | while read file; do fgrep val_loss "$file" | tail -1; done

.PHONY: train clean onnx results squeue