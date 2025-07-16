export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
step1:

	deepspeed --num_gpus=1  /home/wyd/CITER/src/finetune/cot_lora_finetune.py   --data_path /home/wyd/CITER/datasets/cqa/train_small_model.json   --deepspeed_path /home/wyd/CITER/src/finetune/ds_config.json   --model_name /home/wyd/huggingface/Qwen/Qwen2.5-1.5B   --output_dir model
step2_remote:

	@touch training_step2.log
	@echo "Starting background job..."
	@echo "Output and errors will be saved to 'training_step2.log'"
	@nohup sh -c 'python src/pipeline/get_gt_first_iter_remote.py --cot_data /home/wyd/CITER/datasets/cqa/train_mlp.json --output_train train_iter_1.pkl --output_analysis analysis_iter_1.pkl --data_type mc --small_model_path /home/wyd/CITER/model/merged_model' > training_step2.log 2>&1 & echo $$! > step2.pid
	@echo "Job started in the background. PID saved to step2.pid."
	@echo "You can monitor progress with: tail -f training_step2.log"
	@echo "You can now safely disconnect your SSH session."

step2:
	python src/pipeline/get_gt_first_iter.py --cot_data /home/wyd/CITER/datasets/cqa/train_mlp.json --output_train train_iter_1.pkl --output_analysis analysis_iter_1.pkl --data_type mc --small_model_path /home/wyd/CITER/model/merged_model --large_model /home/wyd/huggingface/Qwen/Qwen2.5-14B-Instruct

step2_part1:
	python src/pipeline/generate_small_model_data.py \
      --cot_data /home/wyd/CITER/datasets/cqa/train_mlp.json \
      --small_model_path /home/wyd/CITER/model/merged_model  \
      --output_intermediate output_data/small_model_intermediate.pkl \
      --cuda_devices 0 \
      --data_type mc

step2_part2:
	@touch generate_large_model_data.log
	@echo "Starting background job..."
	@echo "Output and errors will be saved to 'training_step2.log'"
	@nohup sh -c 'python src/pipeline/generate_large_model_data.py \
		--intermediate_data /home/wyd/CITER/output_data/small_model_intermediate.pkl.pkl \
		--large_model /home/wyd/huggingface/Qwen/Qwen2.5-14B-Instruct \
		--output_large output_data/large_model_data.pkl \
		--cuda_devices 0'  > generate_large_model_data.log 2>&1 & echo $$! > generate_large_model_data.pid
	@echo "Job started in the background. PID saved to step2.pid."
	@echo "You can monitor progress with: tail -f generate_large_model_data.log"
	@echo "You can now safely disconnect your SSH session."

step2_part3:
	python src/pipeline/merge_model_data.py \
		--intermediate_data /home/wyd/CITER/output_data/small_model_intermediate.pkl.pkl \
		--large_model_data /home/wyd/CITER/output_data/large_model_data.pkl.pkl \
		--output_train output_data/train_iter_1.pkl \
		--output_analysis output_data/analysis_iter_1.pkl \
		--data_type mc

step3:
	python src/pipeline/mlp_training.py \
		--data_path /home/wyd/CITER/output_data/train_iter_1.pkl.pkl \
		--output_path model/mlp/ \
		--model_name /home/wyd/huggingface/Qwen/Qwen2-1.5B \
		--batch_size 64 \
		--learning_rate 1e-7
		