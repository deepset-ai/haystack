import pickle
from pathlib import Path
from tqdm import tqdm

n_passages = 1_000_000
embeddings_dir = Path("embeddings")
embeddings_filenames = [f"wikipedia_passages_{i}.pkl" for i in range(50)]
passages_filename = "psgs_w100_minus_gold.tsv"

# Extract passage ids
passage_ids = []
with open(passages_filename) as f:
    f.readline()    # Ignore column headers
    for _ in range(n_passages):
        l = f.readline()
        passage_ids.append(l.split()[0])
assert len(passage_ids) == len(set(passage_ids))
passages_ids = set(passage_ids)

# Gather vectors for passages
ret = []
for ef in tqdm(embeddings_filenames):
    curr = pickle.load(open(embeddings_dir / ef, "rb"))
    for i, vec in curr:
        if i in passages_ids:
            ret.append((i, vec))
print(len(ret))

# Write vectors to file
with open(f"wikipedia_passages_{n_passages}.pkl", "wb") as f:
    pickle.dump(ret, f)



