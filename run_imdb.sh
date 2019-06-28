#!/usr/bin/env bash

IMDB_DIR=
GS_ROOT=.
LARGE_DIR=

python run_classifier.py \
  --use_tpu=False \
  --use-gpu=False \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=${IMDB_DIR} \
  --output_dir=${GS_ROOT}/proc_data/imdb \
  --model_dir=${GS_ROOT}/exp/imdb \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${GS_ROOT}/${LARGE_DIR}/model_config.json \
  --init_checkpoint=${GS_ROOT}/${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500