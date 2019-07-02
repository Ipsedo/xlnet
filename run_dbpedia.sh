#!/usr/bin/env bash

GS_ROOT=.
LARGE_DIR=models/xlnet_cased_L-24_H-1024_A-16

python run_classifier.py \
  --use_tpu=False \
  --use-gpu=False \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=dbpedia \
  --output_dir=${GS_ROOT}/proc_data/dbpedia \
  --model_dir=${GS_ROOT}/exp/dbpedia \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${GS_ROOT}/${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=2e-5 \
  --train_steps=40 \
  --warmup_steps=10 \
  --save_steps=10 \
  --iterations=10