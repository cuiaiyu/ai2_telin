#!/bin/bash
#SBATCH --account=mics
#SBATCH --partition=mics
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4g

source ~/.bashrc
conda activate mcs

MODEL_TYPE=roberta
TASK_NAME=physicaliqa-carved
#TASK_NAME2=cn_all_cs
#TASK_NAME2=cn_all_cs_10k
#TASK_NAME2=cn_all_cs_20k
#TASK_NAME2=cn_all_cs_50k
#TASK_NAME2=cn_physical_cs_narrow
#TASK_NAME2=cn_physical_cs_relaxed

# MODEL_WEIGHT="baseline_w_cn_all_cs_v1"
# MODEL_WEIGHT="baseline_w_cn_all_cs_60k"
# MODEL_WEIGHT="baseline"
# MODEL_WEIGHT="lm_finetuned_wikihow_30000_w_cn_all_cs_v1"
#MODEL_WEIGHT="comet_roberta_singletrip_98_w_cn_all_cs_v1"
MODEL_WEIGHT="large_roberta"

python3 -W ignore train.py --model_type $MODEL_TYPE --model_weight $MODEL_WEIGHT \
  --task_config_file config/tasks.yaml \
  --running_config_file config/physicalIQA.yaml \
  --task_name $TASK_NAME \
#  --task_name2 $TASK_NAME2 \
#  --task2_separate_fc true \
  --task_cache_dir ./cache \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
#  --output_dir2 output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME2-pred \
  --log_save_interval 25 --row_log_interval 25
