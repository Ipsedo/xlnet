#!/usr/bin/env bash

LARGE_DIR=models/xlnet_cased_L-24_H-1024_A-16

CUDA_VISIBLE_DEVICE=0 python run_classifier.py \
  --use-gpu=True \
  --use-tpu=False \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=dbpedia \
  --output_dir=./proc_data/dbpedia \
  --model_dir=./exp/dbpedia \
  --uncased=False \
  --spiece_model_file=./${LARGE_DIR}/spiece.model \
  --model_config_path=./${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=./${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --predict_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500