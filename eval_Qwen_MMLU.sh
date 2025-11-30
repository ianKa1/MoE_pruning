#!/bin/bash

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="Qwen/Qwen1.5-MoE-A2.7B"
selected_subjects="all"
gpu_util=0.8



python Qwen_evaluate.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util

