from haystack import Document as Document1
from haystack.nodes import FARMReader
from haystack.preview import Document as Document2
from haystack.preview.components.reader import ExtractiveReader

from time import time

doc_contents = [
    "Angela Merkel was the chancellor of Germany.",
    "Olaf Scholz is the chancellor of Germany",
    "Jerry is the head of the department.",
]
queries = ["Who is the chancellor of Germany?"]
model = "deepset/deberta-v3-large-squad2"
model = "deepset/tinyroberta-squad2"

reader1 = ExtractiveReader(model)
reader1.warm_up()
start = time()
results2 = reader1.run(documents=[[Document2(content=content) for content in doc_contents]], queries=queries, top_k=3)
end = time()
print(end - start)
print(results2)

reader2 = FARMReader(model)
start = time()
results1, _ = reader2.run(documents=[Document1(content=content) for content in doc_contents], query=queries[0])
end = time()
print(end - start)

for answer1, answer2 in zip(results1["answers"], results2["answers"][0]):
    print("FARM:", (answer1.answer, answer1.score), "ExtractiveReader:", (answer2.data, answer2.probability))
