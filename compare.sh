#!/bin/bash

# LABEL_PATH="dev-labels.lst"
# JSONL_PATH="dev.jsonl"
# EXP_NAME_1="attr"
# EXP_NAME_2="temp"
# PRED_1_PATH="attribute/dev-predictions.lst"
# PRED_2_PATH="temporal/dev-predictions.lst"
# PROB_1_PATH="attribute/dev-probabilities.lst"
# PROB_2_PATH="temporal/dev-probabilities.lst"

LABEL_PATH="cache/social_before_after-train-dev/social_before_after-train-dev/dev-labels.lst"
JSONL_PATH="cache/social_before_after-train-dev/social_before_after-train-dev/dev.jsonl"
EXP_NAME_1="Without_ATOMIC_attr_qa"
EXP_NAME_2="With_ATOMIC_attr_qa"
EXP_NAME_3="Baseline"
PRED_1_PATH="output/roberta-large_roberta_siqa_sent_2_12_1-social_before_after-pred/dev-predictions.lst"
PRED_2_PATH="output/roberta-large_roberta_attr_qa_social_sent_2_12_1_fc_true-social_before_after-pred/dev-predictions.lst"
PRED_3_PATH="output/roberta-baseline_2_12_1-social_before_after_still_qa-pred/predictions.lst"
PROB_1_PATH="output/roberta-large_roberta_siqa_sent_2_12_1-social_before_after-pred/dev-probabilities.lst"
PROB_2_PATH="output/roberta-large_roberta_attr_qa_social_sent_2_12_1_fc_true-social_before_after-pred/dev-probabilities.lst"
PROB_3_PATH="output/roberta-baseline_2_12_1-social_before_after_still_qa-pred/probabilities.lst"

python3 compare.py \
    --label_path $LABEL_PATH \
    --jsonl_path $JSONL_PATH \
    --exp_name_1 $EXP_NAME_1 \
    --exp_name_2 $EXP_NAME_3 \
    --pred_1_path $PRED_1_PATH \
    --pred_2_path $PRED_3_PATH \
    --prob_1_path $PROB_1_PATH \
    --prob_2_path $PROB_3_PATH 
