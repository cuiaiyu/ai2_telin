PYTHON=python3
MODEL_TYPE=roberta
TASK_NAME=physicaliqa
# MODEL_WEIGHT="baseline_w_cn_all_cs_v3_single_fc"
MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_maxlen256_v2"


$PYTHON -W ignore analyze.py \
  --file_data ./cache/$TASK_NAME-test/$TASK_NAME-test/dev.jsonl \
  --file_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/predictions.lst \
  --prob_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/probabilities.lst \
  --file_gt ./cache/$TASK_NAME-test/$TASK_NAME-test/dev-labels.lst
