import json

f = open("atomic_attr_qa.jsonl", "r")
l = open("atomic_attr_qa-labels.lst", "r")

all_lines = []
all_labs = []

for line in f:
    all_lines.append(line)

for lab in l:
    all_labs.append(lab)

assert len(all_lines) == len(all_labs)

eval_num = 1954
train_num = len(all_lines)-1955

train_lines = all_lines[eval_num:eval_num+train_num]
train_labs = all_labs[eval_num:eval_num+train_num]
eval_lines = all_lines[:eval_num]
eval_labs = all_labs[:eval_num]

assert len(train_lines) == len(train_labs)
assert len(eval_lines) == len(eval_labs)

print ("Number of data in train: {}".format(len(train_lines)))
print ("Number of data in eval:  {}".format(len(eval_lines)))

ft = open("atomic_attr_qa-train-dev/train.jsonl", "w")
lt = open("atomic_attr_qa-train-dev/train-labels.lst", "w")
fe = open("atomic_attr_qa-train-dev/dev.jsonl", "w")
le = open("atomic_attr_qa-train-dev/dev-labels.lst", "w")

for line in train_lines:
    ft.write(line)
for lab in train_labs:
    lt.write(lab)

for line in eval_lines:
    fe.write(line)
for lab in eval_labs:
    le.write(lab)

ft.close()
lt.close()
fe.close()
le.close()
f.close()
l.close()
