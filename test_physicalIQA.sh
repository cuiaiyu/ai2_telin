PYTHON=python3
MODEL_TYPE=roberta
TASK_NAME=physicaliqa
# MODEL_WEIGHT="baseline_w_cn_all_cs_v3_single_fc"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1"
MODEL_WEIGHT="baseline"

$PYTHON -W ignore test.py --model_type $MODEL_TYPE \
  --model_weight $MODEL_WEIGHT \
  --task_name $TASK_NAME \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --test_input_dir ./cache/$TASK_NAME-test/$TASK_NAME-test/ \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --weights_path output/$MODEL_TYPE-$MODEL_WEIGHT-checkpoints/$TASK_NAME/0/_ckpt_epoch_3.ckpt \
  --tags_csv output/$MODEL_TYPE-$MODEL_WEIGHT-log/$TASK_NAME/version_0/meta_tags.csv \
  # --task2_separate_fc true \


$PYTHON -W ignore analyze.py \
  --file_data ./cache/$TASK_NAME-test/$TASK_NAME-test/dev.jsonl \
  --file_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/predictions.lst \
  --prob_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/probabilities.lst \
  --file_gt ./cache/$TASK_NAME-test/$TASK_NAME-test/dev-labels.lst