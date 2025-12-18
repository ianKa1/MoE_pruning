#!/bin/bash

save_dir="eval_results4/"
global_record_file="eval_results4/eval_record_collection.csv"
model="/workspace/model/mixtral8x7B_Instruct_pruning4experts"
selected_subjects="all"
gpu_util=0.8



python evaluate_from_local.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util

