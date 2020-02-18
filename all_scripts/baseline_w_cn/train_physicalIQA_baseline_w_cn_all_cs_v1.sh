#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=4g
#SBATCH --partition=isi

MODEL_TYPE=roberta
TASK_NAME=physicaliqa
TASK_NAME2=cn_all_cs
# TASK_NAME2=cn_all_cs_30k

# MODEL_WEIGHT="baseline_w_cn_all_cs_v1"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd_v1"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_gac_4"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_v1"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd_30k"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_gac_4_20k_contd"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_maxlen256"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_30k"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_maxlen256_v2"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_maxlen256"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd_maxlen256"
MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd_maxlen256_contd2"

python3 -W ignore train.py --model_type $MODEL_TYPE --model_weight $MODEL_WEIGHT \
  --task_config_file config/tasks.yaml \
  --running_config_file config/physicalIQA.yaml \
  --task_name $TASK_NAME \
  --task_name2 $TASK_NAME2 \
  --task2_separate_fc true \
  --task_cache_dir ./cache \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --output_dir2 output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME2-pred \
  --log_save_interval 25 --row_log_interval 25
