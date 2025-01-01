deepspeed --num_gpus=4 train.py \
  --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
  --tokenizer_name mistralai/Mixtral-8x7B-v0.1 \
  --output_dir ./output \
  --my_num_train_epochs 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_seq_length 8 \
  --train_file ./train.json

