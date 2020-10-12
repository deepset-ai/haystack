import pickle
from pathlib import Path
from tqdm import tqdm
import json

n_passages = 1_000_000
embeddings_dir = Path("embeddings")
embeddings_filenames = [f"wikipedia_passages_{i}.pkl" for i in range(50)]
neg_passages_filename = "psgs_w100_minus_gold.tsv"
gold_passages_filename = "nq2squad-dev.json"

# Extract gold passage ids
passage_ids = []
gold_data = json.load(open(gold_passages_filename))["data"]
for d in gold_data:
    for p in d["paragraphs"]:
        passage_ids.append(str(p["passage_id"]))
print("gold_ids")
print(len(passage_ids))
print()

# Extract neg passage ids
with open(neg_passages_filename) as f:
    f.readline()    # Ignore column headers
    for _ in range(n_passages - len(passage_ids)):
        l = f.readline()
        passage_ids.append(str(l.split()[0]))
assert len(passage_ids) == len(set(passage_ids))
assert set([type(x) for x in passage_ids]) == {str}
passage_ids = set(passage_ids)
print("all_ids")
print(len(passage_ids))
print()


# Gather vectors for passages
ret = []
for ef in tqdm(embeddings_filenames):
    curr = pickle.load(open(embeddings_dir / ef, "rb"))
    for i, vec in curr:
        if i in passage_ids:
            ret.append((i, vec))
print("n_vectors")
print(len(ret))
print()

# Write vectors to file
with open(f"wikipedia_passages_{n_passages}.pkl", "wb") as f:
    pickle.dump(ret, f)



