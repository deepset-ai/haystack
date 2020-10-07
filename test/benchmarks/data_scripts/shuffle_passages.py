import json
from tqdm import tqdm
import time
import random
random.seed(42)

lines = []
with open("psgs_w100_minus_gold_unshuffled.tsv") as f:
    f.readline()    # Remove column header
    lines = [l for l in tqdm(f)]

tic = time.perf_counter()
random.shuffle(lines)
toc = time.perf_counter()
t = toc - tic
print(t)
with open("psgs_w100_minus_gold.tsv", "w") as f:
    f.write("id\ttext\title\n")
    for l in tqdm(lines):
        f.write(l)
