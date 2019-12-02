#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=4g
#SBATCH --partition=isi

# directory="output/roberta-large_roberta_attr_qa_random_name_fc_true-log/socialiqa/version_0"
# directory="output/roberta-large_roberta_attr_qa_grad_accu_2-log/socialiqa/version_0"
# directory="output/roberta-large_roberta_which_one_qa-log/socialiqa/version_0"
directory="output/roberta-large_roberta_which_one_qa_fc_false-log/socialiqa/version_0"

metrics_file_name="metrics.csv"
hyperparam_file_name="meta_tags.csv"

cols_to_check="val_acc val_acc2 epoch"


python3 find_max_vals.py \
    -f $directory/$metrics_file_name \
    -c $cols_to_check

echo "experiment setting of $directory is:"
cat $directory/$hyperparam_file_name | grep accumulate_grad_batches
cat $directory/$hyperparam_file_name | grep batch_size
cat $directory/$hyperparam_file_name | grep task2_separate_fc
cat $directory/$hyperparam_file_name | grep dropout
cat $directory/$hyperparam_file_name | grep whatelse
