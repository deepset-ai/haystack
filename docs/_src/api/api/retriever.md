<a name="sparse"></a>
# sparse

<a name="sparse.ElasticsearchRetriever"></a>
## ElasticsearchRetriever

```python
class ElasticsearchRetriever(BaseRetriever)
```

<a name="sparse.ElasticsearchRetriever.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(document_store: ElasticsearchDocumentStore, custom_query: str = None)
```

**Arguments**:

- `document_store`: an instance of a DocumentStore to retrieve documents from.
- `custom_query`: query string as per Elasticsearch DSL with a mandatory question placeholder($question).

Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
names must match with the filters dict supplied in self.retrieve().
::

An example custom_query:
{
"size": 10,
"query": {
"bool": {
"should": [{"multi_match": {
"query": "${question}",                 // mandatory $question placeholder
"type": "most_fields",
"fields": ["text", "title"]}}],
"filter": [                                 // optional custom filters
{"terms": {"year": "${years}"}},
{"terms": {"quarter": "${quarters}"}},
{"range": {"date": {"gte": "${date}"}}}
],

}
},
}

For this custom_query, a sample retrieve() could be:
::
self.retrieve(query="Why did the revenue increase?",
filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})

<a name="sparse.ElasticsearchFilterOnlyRetriever"></a>
## ElasticsearchFilterOnlyRetriever

```python
class ElasticsearchFilterOnlyRetriever(ElasticsearchRetriever)
```

Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.

<a name="sparse.TfidfRetriever"></a>
## TfidfRetriever

```python
class TfidfRetriever(BaseRetriever)
```

Read all documents from a SQL backend.

Split documents into smaller units (eg, paragraphs or pages) to reduce the
computations when text is passed on to a Reader for QA.

It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.

<a name="dense"></a>
# dense

<a name="dense.DensePassageRetriever"></a>
## DensePassageRetriever

```python
class DensePassageRetriever(BaseRetriever)
```

Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
See the original paper for more details:
Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
(https://arxiv.org/abs/2004.04906).

<a name="dense.DensePassageRetriever.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(document_store: BaseDocumentStore, query_embedding_model: str = "facebook/dpr-question_encoder-single-nq-base", passage_embedding_model: str = "facebook/dpr-ctx_encoder-single-nq-base", max_seq_len: int = 256, use_gpu: bool = True, batch_size: int = 16, embed_title: bool = True, remove_sep_tok_from_untitled_passages: bool = True)
```

Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
The checkpoint format matches huggingface transformers' model format

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `query_embedding_model`: Local path or remote name of question encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models
Currently available remote names: ``"facebook/dpr-question_encoder-single-nq-base"``
- `passage_embedding_model`: Local path or remote name of passage encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models
Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
- `max_seq_len`: Longest length of each sequence
- `use_gpu`: Whether to use gpu or not
- `batch_size`: Number of questions or passages to encode at once
- `embed_title`: Whether to concatenate title and passage to a text pair that is then used to create the embedding
- `remove_sep_tok_from_untitled_passages`: If embed_title is ``True``, there are different strategies to deal with documents that don't have a title.
If this param is ``True`` => Embed passage as single text, similar to embed_title = False (i.e [CLS] passage_tok1 ... [SEP]).
If this param is ``False`` => Embed passage as text pair with empty title (i.e. [CLS] [SEP] passage_tok1 ... [SEP])

<a name="dense.DensePassageRetriever.embed_queries"></a>
#### embed\_queries

```python
 | embed_queries(texts: List[str]) -> List[np.array]
```

Create embeddings for a list of queries using the query encoder

**Arguments**:

- `texts`: Queries to embed

**Returns**:

Embeddings, one per input queries

<a name="dense.DensePassageRetriever.embed_passages"></a>
#### embed\_passages

```python
 | embed_passages(docs: List[Document]) -> List[np.array]
```

Create embeddings for a list of passages using the passage encoder

**Arguments**:

- `docs`: List of Document objects used to represent documents / passages in a standardized way within Haystack.

**Returns**:

Embeddings of documents / passages shape (batch_size, embedding_dim)

<a name="dense.EmbeddingRetriever"></a>
## EmbeddingRetriever

```python
class EmbeddingRetriever(BaseRetriever)
```

<a name="dense.EmbeddingRetriever.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(document_store: BaseDocumentStore, embedding_model: str, use_gpu: bool = True, model_format: str = "farm", pooling_strategy: str = "reduce_mean", emb_extraction_layer: int = -1)
```

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `embedding_model`: Local path or name of model in Hugging Face's model hub such as ``'deepset/sentence_bert'``
- `use_gpu`: Whether to use gpu or not
- `model_format`: Name of framework that was used for saving the model. Options:

- ``'farm'``
- ``'transformers'``
- ``'sentence_transformers'``
- `pooling_strategy`: Strategy for combining the embeddings from the model (for farm / transformers models only).
Options:

- ``'cls_token'`` (sentence vector)
- ``'reduce_mean'`` (sentence vector)
- ``'reduce_max'`` (sentence vector)
- ``'per_token'`` (individual token vectors)
- `emb_extraction_layer`: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
Default: -1 (very last layer).

<a name="dense.EmbeddingRetriever.embed"></a>
#### embed

```python
 | embed(texts: Union[List[str], str]) -> List[np.array]
```

Create embeddings for each text in a list of texts using the retrievers model (`self.embedding_model`)

**Arguments**:

- `texts`: Texts to embed

**Returns**:

List of embeddings (one per input text). Each embedding is a list of floats.

<a name="dense.EmbeddingRetriever.embed_queries"></a>
#### embed\_queries

```python
 | embed_queries(texts: List[str]) -> List[np.array]
```

Create embeddings for a list of queries. For this Retriever type: The same as calling .embed()

**Arguments**:

- `texts`: Queries to embed

**Returns**:

Embeddings, one per input queries

<a name="dense.EmbeddingRetriever.embed_passages"></a>
#### embed\_passages

```python
 | embed_passages(docs: List[Document]) -> List[np.array]
```

Create embeddings for a list of passages. For this Retriever type: The same as calling .embed()

**Arguments**:

- `docs`: List of documents to embed

**Returns**:

Embeddings, one per input passage

<a name="base"></a>
# base

<a name="base.BaseRetriever"></a>
## BaseRetriever

```python
class BaseRetriever(ABC)
```

<a name="base.BaseRetriever.retrieve"></a>
#### retrieve

```python
 | @abstractmethod
 | retrieve(query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents
that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a name="base.BaseRetriever.eval"></a>
#### eval

```python
 | eval(label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label", top_k: int = 10, open_domain: bool = False) -> dict
```

Performs evaluation on the Retriever.
Retriever is evaluated based on whether it finds the correct document given the question string and at which
position in the ranking of documents the correct document is.

|  Returns a dict containing the following metrics:

- "recall": Proportion of questions for which correct document is among retrieved documents
- "mean avg precision": Mean of average precision for each question. Rewards retrievers that give relevant
documents a higher rank.

**Arguments**:

- `label_index`: Index/Table in DocumentStore where labeled questions are stored
- `doc_index`: Index/Table in DocumentStore where documents that are used for evaluation are stored
- `top_k`: How many documents to return per question
- `open_domain`: If ``True``, retrieval will be evaluated by checking if the answer string to a question is
contained in the retrieved docs (common approach in open-domain QA).
If ``False``, retrieval uses a stricter evaluation that checks if the retrieved document ids
are within ids explicitly stated in the labels.

