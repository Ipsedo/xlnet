#!/usr/bin/env bash

ROOT_DIR=/content/clouderizer/xlnet_fork/code/xlnet
LARGE_DIR=models/xlnet_cased_L-24_H-1024_A-16

python run_classifier.py \
  --use-tpu=False \
  --do_train=True \
  --do_eval=True \
  --do-predict=False \
  --eval_all_ckpt=True \
  --task_name=dbpedia_filtered \
  --output_dir=${ROOT_DIR}/proc_data/dbpedia_filtered \
  --model_dir=${ROOT_DIR}/exp/dbpedia_filtered \
  --uncased=False \
  --spiece_model_file=${ROOT_DIR}/${LARGE_DIR}/spiece.model \
  --model_config_path=${ROOT_DIR}/${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${ROOT_DIR}/${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=1 \
  --eval_batch_size=8 \
  --predict_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=2e-5 \
  --train_steps=8000 \
  --warmup_steps=500 \
  --save_steps=1000 \
  --iterations=500