PYTHON=python3
MODEL_TYPE=roberta
TASK_NAME=physicalbinqa

MODEL_WEIGHT="large_roberta_bz4"

# $PYTHON -W ignore eval.py --model_type $MODEL_TYPE \
# $PYTHON -W ignore new_eval.py --model_type $MODEL_TYPE \
$PYTHON -W ignore test.py --model_type $MODEL_TYPE \
  --model_weight $MODEL_WEIGHT \
  --task_name $TASK_NAME \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/physicalBINQA.yaml \
  --test_input_dir ./cache/$TASK_NAME-test/$TASK_NAME-test/ \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --weights_path output/$MODEL_TYPE-$MODEL_WEIGHT-checkpoints/$TASK_NAME/0/_ckpt_epoch_1.ckpt \
  --tags_csv output/$MODEL_TYPE-$MODEL_WEIGHT-log/$TASK_NAME/version_0/meta_tags.csv \
  # --task2_separate_fc true \


$PYTHON -W ignore analyze.py \
  --file_data ./cache/$TASK_NAME-test/$TASK_NAME-test/dev.jsonl \
  --file_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/predictions.lst \
  --prob_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/probabilities.lst \
  --file_gt ./cache/$TASK_NAME-test/$TASK_NAME-test/dev-labels.lst
