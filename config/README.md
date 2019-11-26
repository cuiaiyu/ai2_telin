# Experiments
If the hyperparams are not specified, they were same as the defaults (from CH)

## Baselines
1. `baseline` (server: hpc, gpu: 2 p100)  
```bash
lr: 5e-6
batch_size: 8
dropout: 0.5
log file: /home/nlg-05/telinwu/ai2/output/roberta-baseline-log/physicaliqa/version_0/metrics.csv
```

2. `large_roberta` (server: hpc, gpu: 1 p100)  
```bash
lr: 5e-6
batch_size: 3
log file: /home/nlg-05/telinwu/ai2/output/roberta-large_roberta-log/physicaliqa/version_0/metrics.csv
```


## Ours v3
1. `multi_large_finetuned_v3_26000_2` (server: hpc, gpu: 2 p100)
```bash
lr: 5e-6
batch_size: 8
dropout: 0.5
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000-log/physicaliqa/version_0/metrics.csv
```

2. `multi_large_finetuned_v3_26000_2` (server: hpc, gpu: 1 p100)
```bash
lr: 5e-6
batch_size: 8
dropout: 0.1
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000_2-log/physicaliqa/version_0/metrics.csv
```

3. `multi_large_finetuned_v3_26000_3` (server: hpc, gpu: 1 k80)
```bash
lr: 5e-6
batch_size: 4
dropout: 0.1
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000_3-log/physicaliqa/version_0/metrics.csv
```

4. `multi_large_finetuned_v3_26000_4` (server: hpc, gpu: 1 k80)
```bash
lr: 5e-6
batch_size: 4
dropout: 0
max_nb_epochs: 4
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v4_26000_4-log/physicaliqa/version_0/metrics.csv
```

5. `multi_large_finetuned_v3_26000_5` (server: hpc, gpu: 2 k80)
```bash
lr: 5e-6
batch_size: 8
dropout: 0
max_nb_epochs: 4
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000_5-log/physicaliqa/version_0/metrics.csv
```

6. `multi_large_finetuned_v3_26000_6` (server: hpc, gpu: 2 p100)
```bash
lr: 5e-6
batch_size: 8
dropout: 0
max_nb_epochs: 4
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000_6-log/physicaliqa/version_0/metrics.csv
```

7. `multi_large_finetuned_v3_26000_7` (server: hpc, gpu: 1 k80)
```bash
lr: 5e-6
batch_size: 4
dropout: 0
max_nb_epochs: 4
accumulate_grad_batches: 2
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000_7-log/physicaliqa/version_0/metrics.csv
```

8. `multi_large_finetuned_v3_26000_8` (server: hpc, gpu: 2 p100)
```bash
lr: 5e-6
batch_size: 16
dropout: 0
max_nb_epochs: 4
accumulate_grad_batches: 2
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000_8-log/physicaliqa/version_0/metrics.csv
```

9. `multi_large_finetuned_v3_26000_9` (server: hpc, gpu: 2 p100)
```bash
lr: 5e-6
batch_size: 16
dropout: 0.1
max_nb_epochs: 4
accumulate_grad_batches: 2
log file: /home/nlg-05/telinwu/ai2/output/roberta-multi_large_finetuned_v3_26000_9-log/physicaliqa/version_0/metrics.csv
```


## Ours with graphbert
1. `lm_with_dp_graphs_gtn1L_fcn_dpbatch1_multitask_beta_0p001_10000` (server: pluslab, gpu: 1 2080ti)
```bash
lr: 5e-6
batch_size: 4
dropout: 0.1
max_nb_epochs: 4
accumulate_grad_batches: 2
log file: /lfs1/telinwu/research/ai2_telin/output/roberta-lm_with_dp_graphs_gtn1L_fcn_dpbatch1_multitask_beta_0p001_10000-log/physicaliqa/version_0/metrics.csv 
```
