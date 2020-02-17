#!/bin/bash
LABEL_PATH="dev-labels.lst"
JSONL_PATH="dev.jsonl"
EXP_NAME_1="attr"
EXP_NAME_2="temp"
PRED_1_PATH="attribute/dev-predictions.lst"
PRED_2_PATH="temporal/dev-predictions.lst"
PROB_1_PATH="attribute/dev-probabilities.lst"
PROB_2_PATH="temporal/dev-probabilities.lst"

# LABEL_PATH="/dev-labels.lst"
# JSONL_PATH="/dev.jsonl"
# EXP_NAME_1=""
# EXP_NAME_2=""
# PRED_1_PATH="/dev-predictions.lst"
# PRED_2_PATH="/dev-predictions.lst"
# PROB_1_PATH="/dev-probabilities.lst"
# PROB_2_PATH="/dev-probabilities.lst"

python3 compare.py \
    --label_path $LABEL_PATH \
    --jsonl_path $JSONL_PATH \
    --exp_name_1 $EXP_NAME_1 \
    --exp_name_2 $EXP_NAME_2 \
    --pred_1_path $PRED_1_PATH \
    --pred_2_path $PRED_2_PATH \
    --prob_1_path $PROB_1_PATH \
    --prob_2_path $PROB_2_PATH 