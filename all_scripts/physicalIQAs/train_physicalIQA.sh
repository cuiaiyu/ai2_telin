#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=4g
#SBATCH --partition=isi

MODEL_TYPE=roberta
TASK_NAME=physicaliqa

# MODEL_WEIGHT="baseline_w_cn_all_cs_v1"
# MODEL_WEIGHT="baseline_w_cn_all_cs_60k"
# MODEL_WEIGHT="baseline"
# MODEL_WEIGHT="lm_finetuned_wikihow_30000_w_cn_all_cs_v1"
# MODEL_WEIGHT="comet_roberta_singletrip_98_w_cn_all_cs_v1"
# MODEL_WEIGHT="lm_finetuned_wikihow_30000_w_cn_all_cs_v1"
MODEL_WEIGHT="large_roberta_bz4"

python3 -W ignore train.py --model_type $MODEL_TYPE --model_weight $MODEL_WEIGHT \
  --task_config_file config/tasks.yaml \
  --running_config_file config/physicalIQA.yaml \
  --task_name $TASK_NAME \
  --task_cache_dir ./cache \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --log_save_interval 25 --row_log_interval 25
