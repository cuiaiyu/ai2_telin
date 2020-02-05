import random as r

with open('../physicaliqa-train-dev-original/train.jsonl','r') as original_train:
   a = original_train.readlines()
with open('../physicaliqa-train-dev-original/train-labels.lst','r') as original_labels:
   b = original_labels.readlines()

c = list(zip(a, b))
r.shuffle(c)
a, b = zip(*c)

with open('train.jsonl','w') as outfile:
   for line in a[:int(len(a)*0.9)]:
      print(line.strip(), file=outfile)

with open('train-labels.lst','w') as outfile:
   for line in b[:int(len(a)*0.9)]:
      print(line.strip(), file=outfile)

with open('dev.jsonl','w') as outfile:
   for line in a[int(len(a)*0.9):]:
      print(line.strip(), file=outfile)

with open('dev-labels.lst','w') as outfile:
   for line in b[int(len(a)*0.9):]:
      print(line.strip(), file=outfile)
