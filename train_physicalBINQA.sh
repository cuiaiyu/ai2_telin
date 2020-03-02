#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=4g
#SBATCH --partition=isi

MODEL_TYPE=roberta
TASK_NAME=physicalbinqa
# TASK_NAME2=cn_all_cs

# MODEL_WEIGHT="large_roberta_bz4"
# MODEL_WEIGHT="large_roberta_bz8"
# MODEL_WEIGHT="large_roberta_bz4_accu2_linear2"
# MODEL_WEIGHT="binpiqa_lm_finetuned_entity_masking_nopad_nothehave_prob0p4_dc_cc_0214_iter50k"
# MODEL_WEIGHT="binpiqa_lm_finetuned_entity_masking_nopad_nothehave_prob0p4_dc_cc_0214_iter50k_nolinear2_w_cn_all_cs_single_fc"
MODEL_WEIGHT="large_roberta_bz2"

python3 -W ignore train.py --model_type $MODEL_TYPE --model_weight $MODEL_WEIGHT \
  --task_config_file config/tasks.yaml \
  --running_config_file config/physicalBINQA.yaml \
  --task_name $TASK_NAME \
  --task_cache_dir ./cache \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --log_save_interval 25 --row_log_interval 25 \
  # --task_name2 $TASK_NAME2 \
  # --output_dir2 output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME2-pred \
  # --true_percentage_test 0.25 \
  # --true_percentage_train 0.8 \
  # --kg_enhanced_finetuning true \
