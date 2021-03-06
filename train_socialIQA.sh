#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=4g
#SBATCH --partition=isi

MODEL_TYPE=roberta
TASK_NAME=socialiqa

# TASK_NAME2=atomic_attr_qa
# TASK_NAME2=atomic_temporal_qa
# TASK_NAME2=atomic_which_one_qa
TASK_NAME2=atomic_attr_qa_random_name

# MODEL_WEIGHT="large_roberta_attr_qa"
# MODEL_WEIGHT="large_roberta_temporal_qa"
# MODEL_WEIGHT="large_roberta_which_one_qa"
MODEL_WEIGHT="large_roberta_attr_qa_random_name_fc_true"
# MODEL_WEIGHT="large_roberta_temporal_qa_fc_false"
# MODEL_WEIGHT="large_roberta_attr_qa_grad_accu_2"

python3 -W ignore train.py --model_type $MODEL_TYPE --model_weight $MODEL_WEIGHT \
  --task_config_file config/tasks.yaml \
  --running_config_file config/socialIQA.yaml \
  --task_name $TASK_NAME \
  --task_name2 $TASK_NAME2 \
  --task_cache_dir ./cache \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --output_dir2 output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME2-pred \
  --log_save_interval 25 --row_log_interval 25 \
  --task2_separate_fc true \
