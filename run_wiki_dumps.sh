#!/usr/bin/env bash

LARGE_DIR=models/xlnet_cased_L-24_H-1024_A-16

python run_classifier.py \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=wiki_dump \
  --output_dir=./proc_data/wiki_dump \
  --model_dir=./exp/wiki_dump \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=./${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=./${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=2 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=2e-5 \
  --train_steps=400 \
  --warmup_steps=100 \
  --save_steps=100 \
  --iterations=100