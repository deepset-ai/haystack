from collections import defaultdict, Counter
import pickle
import json
import pandas as pd
import collections
from operator import itemgetter

data = json.load(open("triples.json"))
triples = [(item["subject"],item["predicate"],item["object"]) for item in data]
df = pd.DataFrame(triples, columns=["s", "p", "o"])

relation_names = [relation.split(":")[-1].replace("_"," ") for relation in df["p"].values]
c = collections.Counter(relation_names)
c.most_common(n=10)
# filter relations that occur less than ten times
items = list(c.items())
items.sort(key=itemgetter(1), reverse=True)
# top_relations is a list of names of relations that occur more than ten times in the knowledge base sorted by frequency of occurrenc
top_relations = [relation for (relation, frequency) in items if frequency > 10]

json.dump(top_relations, open("top_relations.json", "w"))
