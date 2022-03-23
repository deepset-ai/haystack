<!---
title: "Ranker"
metaTitle: "Ranker"
metaDescription: ""
slug: "/docs/ranker"
date: "2020-05-26"
id: "rankermd"
--->

# Ranker

There are pure "semantic document search" use cases that do not need question answering functionality but only document ranking.
While the [Retriever](/docs/latest/retrievermd) is a perfect fit for document retrieval, we can further improve its results with the Ranker.
For example, BM25 (sparse retriever) does not take into account semantics of the documents and the query but only their keywords.
The Ranker can re-rank the results of the retriever step by taking semantics into account.
Similar to the Reader, it is based on the latest language models.
Instead of returning answers, it returns documents in re-ranked order.

Without a Ranker and its re-ranking step, the querying process is faster but the query results might be of lower quality.
If you want to do "semantic document search" instead of a question answering, try first with a Retriever only.
In case the semantic similarity of the query and the resulting documents is low, add a Ranker.

Note that a Ranker needs to be initialised with a model trained on a text pair classification task.
You can train the model also with the train() method of the Ranker.
Alternatively, [this example](https://github.com/deepset-ai/FARM/blob/master/examples/text_pair_classification.py) shows how to train a text pair classification model in FARM.


## FARMRanker

### Description

The FARMRanker consists of a Transformer-based model for document re-ranking using the TextPairClassifier of [FARM](https://github.com/deepset-ai/FARM).
Given a text pair of query and passage, the TextPairClassifier either predicts label "1" if the pair is similar or label "0" if they are dissimilar (accompanied with a probability).
While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.
With a FARMRanker, you can:
* Directly get predictions (re-ranked version of the supplied list of Document) via predict() if supplying a pre-trained model
* Take a plain language model (e.g. `bert-base-cased`) and train it for TextPairClassification via train()

### Initialisation

```python
from haystack.document_store import ElasticsearchDocumentStore
from haystack.retriever import ElasticsearchRetriever
from haystack.ranker import FARMRanker
from haystack import Pipeline

document_store = ElasticsearchDocumentStore()
...
retriever = ElasticsearchRetriever(document_store)
ranker = FARMRanker(model_name_or_path="saved_models/roberta-base-asnq-binary")
...
p = Pipeline()
p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
```
