#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=4g
#SBATCH --partition=isi

file_path="output/roberta-large_roberta_attr_qa_fc_false-log/socialiqa/version_0/metrics.csv"


cols_to_check="val_acc val_acc2"


python3 find_max_val.py \
    -f $file_path \
    -c $cols_to_check
