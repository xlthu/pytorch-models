from torchseq.datasets.wikitext2 import WikiText2

dataset = WikiText2(root="./data", split="train")

lens = []

for i in range(len(dataset)):
    data = dataset[i]

    if len(data[0]) < 1:
        continue

    lens.append(len(data[0]))


import numpy

print("max", numpy.max(lens))
print("min", numpy.min(lens))

print("mean", numpy.mean(lens))

print("std", numpy.std(lens))