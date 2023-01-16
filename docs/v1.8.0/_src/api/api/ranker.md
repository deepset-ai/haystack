<a id="base"></a>

# Module base

<a id="base.BaseRanker"></a>

## BaseRanker

```python
class BaseRanker(BaseComponent)
```

<a id="base.BaseRanker.timing"></a>

#### BaseRanker.timing

```python
def timing(fn, attr_name)
```

Wrapper method used to time functions.

<a id="base.BaseRanker.eval"></a>

#### BaseRanker.eval

```python
def eval(label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label", top_k: int = 10, open_domain: bool = False, return_preds: bool = False) -> dict
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

<a id="sentence_transformers"></a>

# Module sentence\_transformers

<a id="sentence_transformers.SentenceTransformersRanker"></a>

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

```python
|     retriever = BM25Retriever(document_store=document_store)
|     ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
|     p = Pipeline()
|     p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
|     p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
```

<a id="sentence_transformers.SentenceTransformersRanker.__init__"></a>

#### SentenceTransformersRanker.\_\_init\_\_

```python
def __init__(model_name_or_path: Union[str, Path], model_version: Optional[str] = None, top_k: int = 10, use_gpu: bool = True, devices: Optional[List[Union[str, torch.device]]] = None, batch_size: int = 16, scale_score: bool = True, progress_bar: bool = True, use_auth_token: Optional[Union[str, bool]] = None)
```

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g.
'cross-encoder/ms-marco-MiniLM-L-12-v2'.
See https://huggingface.co/cross-encoder for full list of available models
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `top_k`: The maximum number of documents to return
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `devices`: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
The strings will be converted into pytorch devices, so use the string notation described here:
https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device
(e.g. ["cuda:0"]).
- `batch_size`: Number of documents to process at a time.
- `scale_score`: The raw predictions will be transformed using a Sigmoid activation function in case the model
only predicts a single label. For multi-label predictions, no scaling is applied. Set this
to False if you do not want any scaling of the raw predictions.
- `progress_bar`: Whether to show a progress bar while processing the documents.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a id="sentence_transformers.SentenceTransformersRanker.predict"></a>

#### SentenceTransformersRanker.predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]
```

Use loaded ranker model to re-rank the supplied list of Document.

Returns list of Document sorted by (desc.) similarity with the query.

**Arguments**:

- `query`: Query string
- `documents`: List of Document to be re-ranked
- `top_k`: The maximum number of documents to return

**Returns**:

List of Document

<a id="sentence_transformers.SentenceTransformersRanker.predict_batch"></a>

#### SentenceTransformersRanker.predict\_batch

```python
def predict_batch(queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int] = None, batch_size: Optional[int] = None) -> Union[List[Document], List[List[Document]]]
```

Use loaded ranker model to re-rank the supplied lists of Documents.

Returns lists of Documents sorted by (desc.) similarity with the corresponding queries.


- If you provide a list containing a single query...

    - ... and a single list of Documents, the single list of Documents will be re-ranked based on the
      supplied query.
    - ... and a list of lists of Documents, each list of Documents will be re-ranked individually based on the
      supplied query.


- If you provide a list of multiple queries...

    - ... you need to provide a list of lists of Documents. Each list of Documents will be re-ranked based on
      its corresponding query.

**Arguments**:

- `queries`: Single query string or list of queries
- `documents`: Single list of Documents or list of lists of Documents to be reranked.
- `top_k`: The maximum number of documents to return per Document list.
- `batch_size`: Number of Documents to process at a time.
