<a name="base"></a>
# Module base

<a name="base.BaseRetriever"></a>
## BaseRetriever Objects

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

<a name="base.BaseRetriever.timing"></a>
#### timing

```python
 | timing(fn)
```

Wrapper method used to time functions.

<a name="base.BaseRetriever.eval"></a>
#### eval

```python
 | eval(label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label", top_k: int = 10, open_domain: bool = False, return_preds: bool = False) -> dict
```

Performs evaluation on the Retriever.
Retriever is evaluated based on whether it finds the correct document given the query string and at which
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

<a name="sparse"></a>
# Module sparse

<a name="sparse.ElasticsearchRetriever"></a>
## ElasticsearchRetriever Objects

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
- `custom_query`: query string as per Elasticsearch DSL with a mandatory query placeholder(query).

Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
names must match with the filters dict supplied in self.retrieve().
::

**An example custom_query:**
```python
|    {
|        "size": 10,
|        "query": {
|            "bool": {
|                "should": [{"multi_match": {
|                    "query": "${query}",                 // mandatory query placeholder
|                    "type": "most_fields",
|                    "fields": ["text", "title"]}}],
|                "filter": [                                 // optional custom filters
|                    {"terms": {"year": "${years}"}},
|                    {"terms": {"quarter": "${quarters}"}},
|                    {"range": {"date": {"gte": "${date}"}}}
|                    ],
|            }
|        },
|    }
```

**For this custom_query, a sample retrieve() could be:**
```python
|    self.retrieve(query="Why did the revenue increase?",
|                  filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
```

<a name="sparse.ElasticsearchRetriever.retrieve"></a>
#### retrieve

```python
 | retrieve(query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents
that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a name="sparse.ElasticsearchFilterOnlyRetriever"></a>
## ElasticsearchFilterOnlyRetriever Objects

```python
class ElasticsearchFilterOnlyRetriever(ElasticsearchRetriever)
```

Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.

<a name="sparse.ElasticsearchFilterOnlyRetriever.retrieve"></a>
#### retrieve

```python
 | retrieve(query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents
that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a name="sparse.TfidfRetriever"></a>
## TfidfRetriever Objects

```python
class TfidfRetriever(BaseRetriever)
```

Read all documents from a SQL backend.

Split documents into smaller units (eg, paragraphs or pages) to reduce the
computations when text is passed on to a Reader for QA.

It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.

<a name="sparse.TfidfRetriever.retrieve"></a>
#### retrieve

```python
 | retrieve(query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents
that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a name="sparse.TfidfRetriever.fit"></a>
#### fit

```python
 | fit()
```

Performing training on this class according to the TF-IDF algorithm.

<a name="dense"></a>
# Module dense

<a name="dense.DensePassageRetriever"></a>
## DensePassageRetriever Objects

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
 | __init__(document_store: BaseDocumentStore, query_embedding_model: Union[Path, str] = "facebook/dpr-question_encoder-single-nq-base", passage_embedding_model: Union[Path, str] = "facebook/dpr-ctx_encoder-single-nq-base", max_seq_len_query: int = 64, max_seq_len_passage: int = 256, use_gpu: bool = True, batch_size: int = 16, embed_title: bool = True, use_fast_tokenizers: bool = True, similarity_function: str = "dot_product")
```

Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
The checkpoint format matches huggingface transformers' model format

**Example:**

```python
|    # remote model from FAIR
|    DensePassageRetriever(document_store=your_doc_store,
|                          query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
|                          passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")
|    # or from local path
|    DensePassageRetriever(document_store=your_doc_store,
|                          query_embedding_model="model_directory/question-encoder",
|                          passage_embedding_model="model_directory/context-encoder")
```

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `query_embedding_model`: Local path or remote name of question encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models
Currently available remote names: ``"facebook/dpr-question_encoder-single-nq-base"``
- `passage_embedding_model`: Local path or remote name of passage encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models
Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
- `max_seq_len_query`: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
- `max_seq_len_passage`: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
- `use_gpu`: Whether to use gpu or not
- `batch_size`: Number of questions or passages to encode at once
- `embed_title`: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
This is the approach used in the original paper and is likely to improve performance if your
titles contain meaningful information for retrieval (topic, entities etc.) .
The title is expected to be present in doc.meta["name"] and can be supplied in the documents
before writing them to the DocumentStore like this:
{"text": "my text", "meta": {"name": "my title"}}.

<a name="dense.DensePassageRetriever.retrieve"></a>
#### retrieve

```python
 | retrieve(query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents
that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

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

<a name="dense.DensePassageRetriever.train"></a>
#### train

```python
 | train(data_dir: str, train_filename: str, dev_filename: str = None, test_filename: str = None, batch_size: int = 2, embed_title: bool = True, num_hard_negatives: int = 1, num_positives: int = 1, n_epochs: int = 3, evaluate_every: int = 1000, n_gpu: int = 1, learning_rate: float = 1e-5, epsilon: float = 1e-08, weight_decay: float = 0.0, num_warmup_steps: int = 100, grad_acc_steps: int = 1, optimizer_name: str = "TransformersAdamW", optimizer_correct_bias: bool = True, save_dir: str = "../saved_models/dpr", query_encoder_save_dir: str = "query_encoder", passage_encoder_save_dir: str = "passage_encoder")
```

train a DensePassageRetrieval model

**Arguments**:

- `data_dir`: Directory where training file, dev file and test file are present
- `train_filename`: training filename
- `dev_filename`: development set filename, file to be used by model in eval step of training
- `test_filename`: test set filename, file to be used by model in test step after training
- `batch_size`: total number of samples in 1 batch of data
- `embed_title`: whether to concatenate passage title with each passage. The default setting in official DPR embeds passage title with the corresponding passage
- `num_hard_negatives`: number of hard negative passages(passages which are very similar(high score by BM25) to query but do not contain the answer
- `num_positives`: number of positive passages
- `n_epochs`: number of epochs to train the model on
- `evaluate_every`: number of training steps after evaluation is run
- `n_gpu`: number of gpus to train on
- `learning_rate`: learning rate of optimizer
- `epsilon`: epsilon parameter of optimizer
- `weight_decay`: weight decay parameter of optimizer
- `grad_acc_steps`: number of steps to accumulate gradient over before back-propagation is done
- `optimizer_name`: what optimizer to use (default: TransformersAdamW)
- `num_warmup_steps`: number of warmup steps
- `optimizer_correct_bias`: Whether to correct bias in optimizer
- `save_dir`: directory where models are saved
- `query_encoder_save_dir`: directory inside save_dir where query_encoder model files are saved
- `passage_encoder_save_dir`: directory inside save_dir where passage_encoder model files are saved

<a name="dense.DensePassageRetriever.save"></a>
#### save

```python
 | save(save_dir: Union[Path, str], query_encoder_dir: str = "query_encoder", passage_encoder_dir: str = "passage_encoder")
```

Save DensePassageRetriever to the specified directory.

**Arguments**:

- `save_dir`: Directory to save to.
- `query_encoder_dir`: Directory in save_dir that contains query encoder model.
- `passage_encoder_dir`: Directory in save_dir that contains passage encoder model.

**Returns**:

None

<a name="dense.DensePassageRetriever.load"></a>
#### load

```python
 | @classmethod
 | load(cls, load_dir: Union[Path, str], document_store: BaseDocumentStore, max_seq_len_query: int = 64, max_seq_len_passage: int = 256, use_gpu: bool = True, batch_size: int = 16, embed_title: bool = True, use_fast_tokenizers: bool = True, similarity_function: str = "dot_product", query_encoder_dir: str = "query_encoder", passage_encoder_dir: str = "passage_encoder")
```

Load DensePassageRetriever from the specified directory.

<a name="dense.EmbeddingRetriever"></a>
## EmbeddingRetriever Objects

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

<a name="dense.EmbeddingRetriever.retrieve"></a>
#### retrieve

```python
 | retrieve(query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents
that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

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

