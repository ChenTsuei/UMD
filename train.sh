#!/usr/bin/env bash

# ./train.sh <gpu_id> <task_name> <model_file> <output_file>
CUDA_VISIBLE_DEVICES=$1 nohup python -u train.py $2 $3 >> $4 2>&1 &
