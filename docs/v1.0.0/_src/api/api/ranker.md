<a name="base"></a>
# Module base

<a name="base.BaseRanker"></a>
## BaseRanker

```python
class BaseRanker(BaseComponent)
```

<a name="base.BaseRanker.timing"></a>
#### timing

```python
 | timing(fn, attr_name)
```

Wrapper method used to time functions.

<a name="base.BaseRanker.eval"></a>
#### eval

```python
 | eval(label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label", top_k: int = 10, open_domain: bool = False, return_preds: bool = False) -> dict
```

Performs evaluation of the Ranker.
Ranker is evaluated in the same way as a Retriever based on whether it finds the correct document given the query string and at which
position in the ranking of documents the correct document is.

|  Returns a dict containing the following metrics:

    - "recall": Proportion of questions for which correct document is among retrieved documents
    - "mrr": Mean of reciprocal rank. Rewards retrievers that give relevant documents a higher rank.
      Only considers the highest ranked relevant document.
    - "map": Mean of average precision for each question. Rewards retrievers that give relevant
      documents a higher rank. Considers all retrieved relevant documents. If ``open_domain=True``,
      average precision is normalized by the number of retrieved relevant documents per query.
      If ``open_domain=False``, average precision is normalized by the number of all relevant documents
      per query.

**Arguments**:

- `label_index`: Index/Table in DocumentStore where labeled questions are stored
- `doc_index`: Index/Table in DocumentStore where documents that are used for evaluation are stored
- `top_k`: How many documents to return per query
- `open_domain`: If ``True``, retrieval will be evaluated by checking if the answer string to a question is
                    contained in the retrieved docs (common approach in open-domain QA).
                    If ``False``, retrieval uses a stricter evaluation that checks if the retrieved document ids
                    are within ids explicitly stated in the labels.
- `return_preds`: Whether to add predictions in the returned dictionary. If True, the returned dictionary
                     contains the keys "predictions" and "metrics".

<a name="sentence_transformers"></a>
# Module sentence\_transformers

<a name="sentence_transformers.SentenceTransformersRanker"></a>
## SentenceTransformersRanker

```python
class SentenceTransformersRanker(BaseRanker)
```

Sentence Transformer based pre-trained Cross-Encoder model for Document Re-ranking (https://huggingface.co/cross-encoder).
Re-Ranking can be used on top of a retriever to boost the performance for document search. This is particularly useful if the retriever has a high recall but is bad in sorting the documents by relevance.

SentenceTransformerRanker handles Cross-Encoder models
    - use a single logit as similarity score e.g.  cross-encoder/ms-marco-MiniLM-L-12-v2
    - use two output logits (no_answer, has_answer) e.g. deepset/gbert-base-germandpr-reranking
https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers

|  With a SentenceTransformersRanker, you can:
 - directly get predictions via predict()

Usage example:
...
retriever = ElasticsearchRetriever(document_store=document_store)
ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
p = Pipeline()
p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])

<a name="sentence_transformers.SentenceTransformersRanker.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: Union[str, Path], model_version: Optional[str] = None, top_k: int = 10, use_gpu: bool = True, devices: Optional[List[Union[int, str, torch.device]]] = None)
```

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g.
'cross-encoder/ms-marco-MiniLM-L-12-v2'.
See https://huggingface.co/cross-encoder for full list of available models
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `top_k`: The maximum number of documents to return
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `devices`: List of GPU devices to limit inference to certain GPUs and not use all available ones (e.g. ["cuda:0"]).

<a name="sentence_transformers.SentenceTransformersRanker.predict_batch"></a>
#### predict\_batch

```python
 | predict_batch(query_doc_list: List[dict], top_k: int = None, batch_size: int = None)
```

Use loaded Ranker model to, for a list of queries, rank each query's supplied list of Document.

Returns list of dictionary of query and list of document sorted by (desc.) similarity with query

**Arguments**:

- `query_doc_list`: List of dictionaries containing queries with their retrieved documents
- `top_k`: The maximum number of answers to return for each query
- `batch_size`: Number of samples the model receives in one batch for inference

**Returns**:

List of dictionaries containing query and ranked list of Document

<a name="sentence_transformers.SentenceTransformersRanker.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]
```

Use loaded ranker model to re-rank the supplied list of Document.

Returns list of Document sorted by (desc.) similarity with the query.

**Arguments**:

- `query`: Query string
- `documents`: List of Document to be re-ranked
- `top_k`: The maximum number of documents to return

**Returns**:

List of Document
