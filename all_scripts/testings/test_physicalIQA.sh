PYTHON=python3
MODEL_TYPE=roberta
TASK_NAME=physicaliqa
# MODEL_WEIGHT="baseline_w_cn_all_cs_v3_single_fc"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1"
# MODEL_WEIGHT="baseline"
<<<<<<< HEAD
=======
# MODEL_WEIGHT="baseline_w_cn_all_cs_v3_contd"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd_v1"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_v1"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd_30k"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_maxlen256"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_30k"
# MODEL_WEIGHT="baseline_w_cn_all_cs_v1_contd2_maxlen256_v2"
# MODEL_WEIGHT="baseline8"
>>>>>>> b421bdb6fdd7df48ccceb5e1ff60f08d8b3e977b
MODEL_WEIGHT="lm_with_dp_graphs_gtn1L_fcn_dpbatch1_multitask_beta_0p001_10000_bc"

# $PYTHON -W ignore eval.py --model_type $MODEL_TYPE \
# $PYTHON -W ignore new_eval.py --model_type $MODEL_TYPE \
$PYTHON -W ignore test.py --model_type $MODEL_TYPE \
  --model_weight $MODEL_WEIGHT \
  --task_name $TASK_NAME \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --test_input_dir ./cache/$TASK_NAME-test/$TASK_NAME-test/ \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --weights_path output/$MODEL_TYPE-$MODEL_WEIGHT-checkpoints/$TASK_NAME/0/_ckpt_epoch_4.ckpt \
  --tags_csv output/$MODEL_TYPE-$MODEL_WEIGHT-log/$TASK_NAME/version_0/meta_tags.csv \
  # --task2_separate_fc true \


$PYTHON -W ignore analyze.py \
  --file_data ./cache/$TASK_NAME-test/$TASK_NAME-test/dev.jsonl \
  --file_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/predictions.lst \
  --prob_pred output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred/probabilities.lst \
  --file_gt ./cache/$TASK_NAME-test/$TASK_NAME-test/dev-labels.lst
