<a id="base"></a>

# Module base

<a id="base.BaseGraphRetriever"></a>

## BaseGraphRetriever

```python
class BaseGraphRetriever(BaseComponent)
```

Base classfor knowledge graph retrievers.

<a id="base.BaseRetriever"></a>

## BaseRetriever

```python
class BaseRetriever(BaseComponent)
```

Base class for regular retrievers.

<a id="base.BaseRetriever.retrieve"></a>

#### BaseRetriever.retrieve

```python
@abstractmethod
def retrieve(query: str,
             filters: Optional[Dict[str, Union[Dict, List, str, int, float,
                                               bool]]] = None,
             top_k: Optional[int] = None,
             index: str = None,
             headers: Optional[Dict[str, str]] = None,
             scale_score: bool = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="base.BaseRetriever.timing"></a>

#### BaseRetriever.timing

```python
def timing(fn, attr_name)
```

Wrapper method used to time functions.

<a id="base.BaseRetriever.eval"></a>

#### BaseRetriever.eval

```python
def eval(label_index: str = "label",
         doc_index: str = "eval_document",
         label_origin: str = "gold-label",
         top_k: int = 10,
         open_domain: bool = False,
         return_preds: bool = False,
         headers: Optional[Dict[str, str]] = None) -> dict
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
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="sparse"></a>

# Module sparse

<a id="sparse.BM25Retriever"></a>

## BM25Retriever

```python
class BM25Retriever(BaseRetriever)
```

<a id="sparse.BM25Retriever.__init__"></a>

#### BM25Retriever.\_\_init\_\_

```python
def __init__(document_store: KeywordDocumentStore,
             top_k: int = 10,
             all_terms_must_match: bool = False,
             custom_query: Optional[str] = None,
             scale_score: bool = True)
```

**Arguments**:

- `document_store`: an instance of one of the following DocumentStores to retrieve from: ElasticsearchDocumentStore, OpenSearchDocumentStore and OpenDistroElasticsearchDocumentStore
- `all_terms_must_match`: Whether all terms of the query must match the document.
If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
Defaults to False.
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
    |                    "query": ${query},                 // mandatory query placeholder
    |                    "type": "most_fields",
    |                    "fields": ["content", "title"]}}],
    |                "filter": [                                 // optional custom filters
    |                    {"terms": {"year": ${years}}},
    |                    {"terms": {"quarter": ${quarters}}},
    |                    {"range": {"date": {"gte": ${date}}}}
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

 Optionally, highlighting can be defined by specifying Elasticsearch's highlight settings.
 See https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html.
 You will find the highlighted output in the returned Document's meta field by key "highlighted".
 ::

     **Example custom_query with highlighting:**
     ```python
    |    {
    |        "size": 10,
    |        "query": {
    |            "bool": {
    |                "should": [{"multi_match": {
    |                    "query": ${query},                 // mandatory query placeholder
    |                    "type": "most_fields",
    |                    "fields": ["content", "title"]}}],
    |            }
    |        },
    |        "highlight": {             // enable highlighting
    |            "fields": {            // for fields content and title
    |                "content": {},
    |                "title": {}
    |            }
    |        },
    |    }
     ```

     **For this custom_query, highlighting info can be accessed by:**
    ```python
    |    docs = self.retrieve(query="Why did the revenue increase?")
    |    highlighted_content = docs[0].meta["highlighted"]["content"]
    |    highlighted_title = docs[0].meta["highlighted"]["title"]
    ```
- `top_k`: How many documents to return per query.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="sparse.BM25Retriever.retrieve"></a>

#### BM25Retriever.retrieve

```python
def retrieve(query: str,
             filters: Optional[Dict[str, Union[Dict, List, str, int, float,
                                               bool]]] = None,
             top_k: Optional[int] = None,
             index: str = None,
             headers: Optional[Dict[str, str]] = None,
             scale_score: bool = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="sparse.BM25Retriever.retrieve_batch"></a>

#### BM25Retriever.retrieve\_batch

```python
def retrieve_batch(queries: List[str],
                   filters: Optional[Union[Dict[str, Union[Dict, List, str,
                                                           int, float, bool]],
                                           List[Dict[str,
                                                     Union[Dict, List, str,
                                                           int, float,
                                                           bool]]], ]] = None,
                   top_k: Optional[int] = None,
                   index: str = None,
                   headers: Optional[Dict[str, str]] = None,
                   batch_size: Optional[int] = None,
                   scale_score: bool = None) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the supplied queries.

Returns a list of lists of Documents (one per query).

**Arguments**:

- `queries`: List of query strings.
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
- `batch_size`: Not applicable.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different
value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="sparse.FilterRetriever"></a>

## FilterRetriever

```python
class FilterRetriever(BM25Retriever)
```

Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.

<a id="sparse.FilterRetriever.retrieve"></a>

#### FilterRetriever.retrieve

```python
def retrieve(query: str,
             filters: dict = None,
             top_k: Optional[int] = None,
             index: str = None,
             headers: Optional[Dict[str, str]] = None,
             scale_score: bool = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: Has no effect, can pass in empty string
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: Has no effect, pass in any int or None
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="sparse.TfidfRetriever"></a>

## TfidfRetriever

```python
class TfidfRetriever(BaseRetriever)
```

Read all documents from a SQL backend.

Split documents into smaller units (eg, paragraphs or pages) to reduce the
computations when text is passed on to a Reader for QA.

It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.

<a id="sparse.TfidfRetriever.__init__"></a>

#### TfidfRetriever.\_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore,
             top_k: int = 10,
             auto_fit=True)
```

**Arguments**:

- `document_store`: an instance of a DocumentStore to retrieve documents from.
- `top_k`: How many documents to return per query.
- `auto_fit`: Whether to automatically update tf-idf matrix by calling fit() after new documents have been added

<a id="sparse.TfidfRetriever.retrieve"></a>

#### TfidfRetriever.retrieve

```python
def retrieve(query: str,
             filters: Optional[Union[Dict[str, Union[Dict, List, str, int,
                                                     float, bool]],
                                     List[Dict[str,
                                               Union[Dict, List, str, int,
                                                     float, bool]]], ]] = None,
             top_k: Optional[int] = None,
             index: str = None,
             headers: Optional[Dict[str, str]] = None,
             scale_score: bool = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="sparse.TfidfRetriever.retrieve_batch"></a>

#### TfidfRetriever.retrieve\_batch

```python
def retrieve_batch(queries: Union[str, List[str]],
                   filters: Optional[Dict[str, Union[Dict, List, str, int,
                                                     float, bool]]] = None,
                   top_k: Optional[int] = None,
                   index: str = None,
                   headers: Optional[Dict[str, str]] = None,
                   batch_size: Optional[int] = None,
                   scale_score: bool = None) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the supplied queries.

Returns a list of lists of Documents (one per query).

**Arguments**:

- `queries`: Single query string or list of queries.
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `batch_size`: Not applicable.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different
value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="sparse.TfidfRetriever.fit"></a>

#### TfidfRetriever.fit

```python
def fit()
```

Performing training on this class according to the TF-IDF algorithm.

<a id="dense"></a>

# Module dense

<a id="dense.DenseRetriever"></a>

## DenseRetriever

```python
class DenseRetriever(BaseRetriever)
```

Base class for all dense retrievers.

<a id="dense.DenseRetriever.embed_queries"></a>

#### DenseRetriever.embed\_queries

```python
@abstractmethod
def embed_queries(queries: List[str]) -> np.ndarray
```

Create embeddings for a list of queries.

**Arguments**:

- `queries`: List of queries to embed.

**Returns**:

Embeddings, one per input query, shape: (queries, embedding_dim)

<a id="dense.DenseRetriever.embed_documents"></a>

#### DenseRetriever.embed\_documents

```python
@abstractmethod
def embed_documents(documents: List[Document]) -> np.ndarray
```

Create embeddings for a list of documents.

**Arguments**:

- `documents`: List of documents to embed.

**Returns**:

Embeddings of documents, one per input document, shape: (documents, embedding_dim)

<a id="dense.DensePassageRetriever"></a>

## DensePassageRetriever

```python
class DensePassageRetriever(DenseRetriever)
```

Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
See the original paper for more details:
Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
(https://arxiv.org/abs/2004.04906).

<a id="dense.DensePassageRetriever.__init__"></a>

#### DensePassageRetriever.\_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore,
             query_embedding_model: Union[
                 Path, str] = "facebook/dpr-question_encoder-single-nq-base",
             passage_embedding_model: Union[
                 Path, str] = "facebook/dpr-ctx_encoder-single-nq-base",
             model_version: Optional[str] = None,
             max_seq_len_query: int = 64,
             max_seq_len_passage: int = 256,
             top_k: int = 10,
             use_gpu: bool = True,
             batch_size: int = 16,
             embed_title: bool = True,
             use_fast_tokenizers: bool = True,
             similarity_function: str = "dot_product",
             global_loss_buffer_size: int = 150000,
             progress_bar: bool = True,
             devices: Optional[List[Union[str, torch.device]]] = None,
             use_auth_token: Optional[Union[str, bool]] = None,
             scale_score: bool = True)
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
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `max_seq_len_query`: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
- `max_seq_len_passage`: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
- `top_k`: How many documents to return per query.
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `batch_size`: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
- `embed_title`: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
This is the approach used in the original paper and is likely to improve performance if your
titles contain meaningful information for retrieval (topic, entities etc.) .
The title is expected to be present in doc.meta["name"] and can be supplied in the documents
before writing them to the DocumentStore like this:
{"text": "my text", "meta": {"name": "my title"}}.
- `use_fast_tokenizers`: Whether to use fast Rust tokenizers
- `similarity_function`: Which function to apply for calculating the similarity of query and passage embeddings during training.
Options: `dot_product` (Default) or `cosine`
- `global_loss_buffer_size`: Buffer size for all_gather() in DDP.
Increase if errors like "encoded data exceeds max_size ..." come up
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.
Note: as multi-GPU training is currently not implemented for DPR, training
will only use the first device provided in this list.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="dense.DensePassageRetriever.retrieve"></a>

#### DensePassageRetriever.retrieve

```python
def retrieve(query: str,
             filters: Optional[Dict[str, Union[Dict, List, str, int, float,
                                               bool]]] = None,
             top_k: Optional[int] = None,
             index: str = None,
             headers: Optional[Dict[str, str]] = None,
             scale_score: bool = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="dense.DensePassageRetriever.retrieve_batch"></a>

#### DensePassageRetriever.retrieve\_batch

```python
def retrieve_batch(queries: List[str],
                   filters: Optional[Union[Dict[str, Union[Dict, List, str,
                                                           int, float, bool]],
                                           List[Dict[str,
                                                     Union[Dict, List, str,
                                                           int, float,
                                                           bool]]], ]] = None,
                   top_k: Optional[int] = None,
                   index: str = None,
                   headers: Optional[Dict[str, str]] = None,
                   batch_size: Optional[int] = None,
                   scale_score: bool = None) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the supplied queries.

Returns a list of lists of Documents (one per query).

**Arguments**:

- `queries`: List of query strings.
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions. Can be a single filter that will be applied to each query or a list of filters
(one filter per query).

Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `batch_size`: Number of queries to embed at a time.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different
value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="dense.DensePassageRetriever.embed_queries"></a>

#### DensePassageRetriever.embed\_queries

```python
def embed_queries(queries: List[str]) -> np.ndarray
```

Create embeddings for a list of queries using the query encoder.

**Arguments**:

- `queries`: List of queries to embed.

**Returns**:

Embeddings, one per input query, shape: (queries, embedding_dim)

<a id="dense.DensePassageRetriever.embed_documents"></a>

#### DensePassageRetriever.embed\_documents

```python
def embed_documents(documents: List[Document]) -> np.ndarray
```

Create embeddings for a list of documents using the passage encoder.

**Arguments**:

- `documents`: List of documents to embed.

**Returns**:

Embeddings of documents, one per input document, shape: (documents, embedding_dim)

<a id="dense.DensePassageRetriever.train"></a>

#### DensePassageRetriever.train

```python
def train(data_dir: str,
          train_filename: str,
          dev_filename: str = None,
          test_filename: str = None,
          max_samples: int = None,
          max_processes: int = 128,
          multiprocessing_strategy: Optional[str] = None,
          dev_split: float = 0,
          batch_size: int = 2,
          embed_title: bool = True,
          num_hard_negatives: int = 1,
          num_positives: int = 1,
          n_epochs: int = 3,
          evaluate_every: int = 1000,
          n_gpu: int = 1,
          learning_rate: float = 1e-5,
          epsilon: float = 1e-08,
          weight_decay: float = 0.0,
          num_warmup_steps: int = 100,
          grad_acc_steps: int = 1,
          use_amp: bool = False,
          optimizer_name: str = "AdamW",
          optimizer_correct_bias: bool = True,
          save_dir: str = "../saved_models/dpr",
          query_encoder_save_dir: str = "query_encoder",
          passage_encoder_save_dir: str = "passage_encoder",
          checkpoint_root_dir: Path = Path("model_checkpoints"),
          checkpoint_every: Optional[int] = None,
          checkpoints_to_keep: int = 3,
          early_stopping: Optional[EarlyStopping] = None)
```

train a DensePassageRetrieval model

**Arguments**:

- `data_dir`: Directory where training file, dev file and test file are present
- `train_filename`: training filename
- `dev_filename`: development set filename, file to be used by model in eval step of training
- `test_filename`: test set filename, file to be used by model in test step after training
- `max_samples`: maximum number of input samples to convert. Can be used for debugging a smaller dataset.
- `max_processes`: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
It can be set to 1 to disable the use of multiprocessing or make debugging easier.
- `multiprocessing_strategy`: Set the multiprocessing sharing strategy, this can be one of file_descriptor/file_system depending on your OS.
If your system has low limits for the number of open file descriptors, and you can’t raise them,
you should use the file_system strategy.
- `dev_split`: The proportion of the train set that will sliced. Only works if dev_filename is set to None
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
- `use_amp`: Whether to use automatic mixed precision (AMP) natively implemented in PyTorch to improve
training speed and reduce GPU memory usage.
For more information, see (Haystack Optimization)[https://haystack.deepset.ai/guides/optimization]
and (Automatic Mixed Precision Package - Torch.amp)[https://pytorch.org/docs/stable/amp.html].
- `optimizer_name`: what optimizer to use (default: AdamW)
- `num_warmup_steps`: number of warmup steps
- `optimizer_correct_bias`: Whether to correct bias in optimizer
- `save_dir`: directory where models are saved
- `query_encoder_save_dir`: directory inside save_dir where query_encoder model files are saved
- `passage_encoder_save_dir`: directory inside save_dir where passage_encoder model files are saved
- `checkpoint_root_dir`: The Path of a directory where all train checkpoints are saved. For each individual
checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
- `checkpoint_every`: Save a train checkpoint after this many steps of training.
- `checkpoints_to_keep`: The maximum number of train checkpoints to save.
- `early_stopping`: An initialized EarlyStopping object to control early stopping and saving of the best models.
Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.

<a id="dense.DensePassageRetriever.save"></a>

#### DensePassageRetriever.save

```python
def save(save_dir: Union[Path, str],
         query_encoder_dir: str = "query_encoder",
         passage_encoder_dir: str = "passage_encoder")
```

Save DensePassageRetriever to the specified directory.

**Arguments**:

- `save_dir`: Directory to save to.
- `query_encoder_dir`: Directory in save_dir that contains query encoder model.
- `passage_encoder_dir`: Directory in save_dir that contains passage encoder model.

**Returns**:

None

<a id="dense.DensePassageRetriever.load"></a>

#### DensePassageRetriever.load

```python
@classmethod
def load(cls,
         load_dir: Union[Path, str],
         document_store: BaseDocumentStore,
         max_seq_len_query: int = 64,
         max_seq_len_passage: int = 256,
         use_gpu: bool = True,
         batch_size: int = 16,
         embed_title: bool = True,
         use_fast_tokenizers: bool = True,
         similarity_function: str = "dot_product",
         query_encoder_dir: str = "query_encoder",
         passage_encoder_dir: str = "passage_encoder")
```

Load DensePassageRetriever from the specified directory.

<a id="dense.TableTextRetriever"></a>

## TableTextRetriever

```python
class TableTextRetriever(DenseRetriever)
```

Retriever that uses a tri-encoder to jointly retrieve among a database consisting of text passages and tables
(one transformer for query, one transformer for text passages, one transformer for tables).
See the original paper for more details:
Kostić, Bogdan, et al. (2021): "Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models"
(https://arxiv.org/abs/2108.04049),

<a id="dense.TableTextRetriever.__init__"></a>

#### TableTextRetriever.\_\_init\_\_

```python
def __init__(
        document_store: BaseDocumentStore,
        query_embedding_model: Union[
            Path, str] = "deepset/bert-small-mm_retrieval-question_encoder",
        passage_embedding_model: Union[
            Path, str] = "deepset/bert-small-mm_retrieval-passage_encoder",
        table_embedding_model: Union[
            Path, str] = "deepset/bert-small-mm_retrieval-table_encoder",
        model_version: Optional[str] = None,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        max_seq_len_table: int = 256,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name", "section_title", "caption"],
        use_fast_tokenizers: bool = True,
        similarity_function: str = "dot_product",
        global_loss_buffer_size: int = 150000,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
        use_fast: bool = True)
```

Init the Retriever incl. the two encoder models from a local or remote model checkpoint.

The checkpoint format matches huggingface transformers' model format

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `query_embedding_model`: Local path or remote name of question encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models.
- `passage_embedding_model`: Local path or remote name of passage encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models.
- `table_embedding_model`: Local path or remote name of table encoder checkpoint. The format equala the
one used by hugging-face transformers' modelhub models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `max_seq_len_query`: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
- `max_seq_len_passage`: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
- `top_k`: How many documents to return per query.
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `batch_size`: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
- `embed_meta_fields`: Concatenate the provided meta fields and text passage / table to a text pair that is
then used to create the embedding.
This is the approach used in the original paper and is likely to improve
performance if your titles contain meaningful information for retrieval
(topic, entities etc.).
- `use_fast_tokenizers`: Whether to use fast Rust tokenizers
- `similarity_function`: Which function to apply for calculating the similarity of query and passage embeddings during training.
Options: `dot_product` (Default) or `cosine`
- `global_loss_buffer_size`: Buffer size for all_gather() in DDP.
Increase if errors like "encoded data exceeds max_size ..." come up
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.
Note: as multi-GPU training is currently not implemented for TableTextRetriever,
training will only use the first device provided in this list.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
- `use_fast`: Whether to use the fast version of DPR tokenizers or fallback to the standard version. Defaults to True.

<a id="dense.TableTextRetriever.retrieve_batch"></a>

#### TableTextRetriever.retrieve\_batch

```python
def retrieve_batch(queries: List[str],
                   filters: Optional[Union[Dict[str, Union[Dict, List, str,
                                                           int, float, bool]],
                                           List[Dict[str,
                                                     Union[Dict, List, str,
                                                           int, float,
                                                           bool]]], ]] = None,
                   top_k: Optional[int] = None,
                   index: str = None,
                   headers: Optional[Dict[str, str]] = None,
                   batch_size: Optional[int] = None,
                   scale_score: bool = None) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the supplied queries.

Returns a list of lists of Documents (one per query).

**Arguments**:

- `queries`: List of query strings.
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions. Can be a single filter that will be applied to each query or a list of filters
(one filter per query).

Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `batch_size`: Number of queries to embed at a time.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different
value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="dense.TableTextRetriever.embed_queries"></a>

#### TableTextRetriever.embed\_queries

```python
def embed_queries(queries: List[str]) -> np.ndarray
```

Create embeddings for a list of queries using the query encoder.

**Arguments**:

- `queries`: List of queries to embed.

**Returns**:

Embeddings, one per input query, shape: (queries, embedding_dim)

<a id="dense.TableTextRetriever.embed_documents"></a>

#### TableTextRetriever.embed\_documents

```python
def embed_documents(documents: List[Document]) -> np.ndarray
```

Create embeddings for a list of text documents and / or tables using the text passage encoder and

the table encoder.

**Arguments**:

- `documents`: List of documents to embed.

**Returns**:

Embeddings of documents, one per input document, shape: (documents, embedding_dim)

<a id="dense.TableTextRetriever.train"></a>

#### TableTextRetriever.train

```python
def train(data_dir: str,
          train_filename: str,
          dev_filename: str = None,
          test_filename: str = None,
          max_samples: int = None,
          max_processes: int = 128,
          dev_split: float = 0,
          batch_size: int = 2,
          embed_meta_fields: List[str] = [
              "page_title", "section_title", "caption"
          ],
          num_hard_negatives: int = 1,
          num_positives: int = 1,
          n_epochs: int = 3,
          evaluate_every: int = 1000,
          n_gpu: int = 1,
          learning_rate: float = 1e-5,
          epsilon: float = 1e-08,
          weight_decay: float = 0.0,
          num_warmup_steps: int = 100,
          grad_acc_steps: int = 1,
          use_amp: bool = False,
          optimizer_name: str = "AdamW",
          optimizer_correct_bias: bool = True,
          save_dir: str = "../saved_models/mm_retrieval",
          query_encoder_save_dir: str = "query_encoder",
          passage_encoder_save_dir: str = "passage_encoder",
          table_encoder_save_dir: str = "table_encoder",
          checkpoint_root_dir: Path = Path("model_checkpoints"),
          checkpoint_every: Optional[int] = None,
          checkpoints_to_keep: int = 3,
          early_stopping: Optional[EarlyStopping] = None)
```

Train a TableTextRetrieval model.

**Arguments**:

- `data_dir`: Directory where training file, dev file and test file are present.
- `train_filename`: Training filename.
- `dev_filename`: Development set filename, file to be used by model in eval step of training.
- `test_filename`: Test set filename, file to be used by model in test step after training.
- `max_samples`: Maximum number of input samples to convert. Can be used for debugging a smaller dataset.
- `max_processes`: The maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
It can be set to 1 to disable the use of multiprocessing or make debugging easier.
- `dev_split`: The proportion of the train set that will sliced. Only works if dev_filename is set to None.
- `batch_size`: Total number of samples in 1 batch of data.
- `embed_meta_fields`: Concatenate meta fields with each passage and table.
The default setting in official MMRetrieval embeds page title,
section title and caption with the corresponding table and title with
corresponding text passage.
- `num_hard_negatives`: Number of hard negative passages (passages which are
very similar (high score by BM25) to query but do not contain the answer)-
- `num_positives`: Number of positive passages.
- `n_epochs`: Number of epochs to train the model on.
- `evaluate_every`: Number of training steps after evaluation is run.
- `n_gpu`: Number of gpus to train on.
- `learning_rate`: Learning rate of optimizer.
- `epsilon`: Epsilon parameter of optimizer.
- `weight_decay`: Weight decay parameter of optimizer.
- `grad_acc_steps`: Number of steps to accumulate gradient over before back-propagation is done.
- `use_amp`: Whether to use automatic mixed precision (AMP) natively implemented in PyTorch to improve
training speed and reduce GPU memory usage.
For more information, see (Haystack Optimization)[https://haystack.deepset.ai/guides/optimization]
and (Automatic Mixed Precision Package - Torch.amp)[https://pytorch.org/docs/stable/amp.html].
- `optimizer_name`: What optimizer to use (default: TransformersAdamW).
- `num_warmup_steps`: Number of warmup steps.
- `optimizer_correct_bias`: Whether to correct bias in optimizer.
- `save_dir`: Directory where models are saved.
- `query_encoder_save_dir`: Directory inside save_dir where query_encoder model files are saved.
- `passage_encoder_save_dir`: Directory inside save_dir where passage_encoder model files are saved.
- `table_encoder_save_dir`: Directory inside save_dir where table_encoder model files are saved.
- `checkpoint_root_dir`: The Path of a directory where all train checkpoints are saved. For each individual
checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
- `checkpoint_every`: Save a train checkpoint after this many steps of training.
- `checkpoints_to_keep`: The maximum number of train checkpoints to save.
- `early_stopping`: An initialized EarlyStopping object to control early stopping and saving of the best models.

<a id="dense.TableTextRetriever.save"></a>

#### TableTextRetriever.save

```python
def save(save_dir: Union[Path, str],
         query_encoder_dir: str = "query_encoder",
         passage_encoder_dir: str = "passage_encoder",
         table_encoder_dir: str = "table_encoder")
```

Save TableTextRetriever to the specified directory.

**Arguments**:

- `save_dir`: Directory to save to.
- `query_encoder_dir`: Directory in save_dir that contains query encoder model.
- `passage_encoder_dir`: Directory in save_dir that contains passage encoder model.
- `table_encoder_dir`: Directory in save_dir that contains table encoder model.

**Returns**:

None

<a id="dense.TableTextRetriever.load"></a>

#### TableTextRetriever.load

```python
@classmethod
def load(cls,
         load_dir: Union[Path, str],
         document_store: BaseDocumentStore,
         max_seq_len_query: int = 64,
         max_seq_len_passage: int = 256,
         max_seq_len_table: int = 256,
         use_gpu: bool = True,
         batch_size: int = 16,
         embed_meta_fields: List[str] = ["name", "section_title", "caption"],
         use_fast_tokenizers: bool = True,
         similarity_function: str = "dot_product",
         query_encoder_dir: str = "query_encoder",
         passage_encoder_dir: str = "passage_encoder",
         table_encoder_dir: str = "table_encoder")
```

Load TableTextRetriever from the specified directory.

<a id="dense.EmbeddingRetriever"></a>

## EmbeddingRetriever

```python
class EmbeddingRetriever(DenseRetriever)
```

<a id="dense.EmbeddingRetriever.__init__"></a>

#### EmbeddingRetriever.\_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore,
             embedding_model: str,
             model_version: Optional[str] = None,
             use_gpu: bool = True,
             batch_size: int = 32,
             max_seq_len: int = 512,
             model_format: Optional[str] = None,
             pooling_strategy: str = "reduce_mean",
             emb_extraction_layer: int = -1,
             top_k: int = 10,
             progress_bar: bool = True,
             devices: Optional[List[Union[str, torch.device]]] = None,
             use_auth_token: Optional[Union[str, bool]] = None,
             scale_score: bool = True,
             embed_meta_fields: List[str] = [])
```

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `embedding_model`: Local path or name of model in Hugging Face's model hub such as ``'sentence-transformers/all-MiniLM-L6-v2'``
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `batch_size`: Number of documents to encode at once.
- `max_seq_len`: Longest length of each document sequence. Maximum number of tokens for the document text. Longer ones will be cut down.
- `model_format`: Name of framework that was used for saving the model or model type. If no model_format is
provided, it will be inferred automatically from the model configuration files.
Options:

- ``'farm'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
- ``'transformers'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
- ``'sentence_transformers'`` (will use `_SentenceTransformersEmbeddingEncoder` as embedding encoder)
- ``'retribert'`` (will use `_RetribertEmbeddingEncoder` as embedding encoder)
- `pooling_strategy`: Strategy for combining the embeddings from the model (for farm / transformers models only).
Options:

- ``'cls_token'`` (sentence vector)
- ``'reduce_mean'`` (sentence vector)
- ``'reduce_max'`` (sentence vector)
- ``'per_token'`` (individual token vectors)
- `emb_extraction_layer`: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
Default: -1 (very last layer).
- `top_k`: How many documents to return per query.
- `progress_bar`: If true displays progress bar during embedding.
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.
Note: As multi-GPU training is currently not implemented for EmbeddingRetriever,
training will only use the first device provided in this list.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
- `embed_meta_fields`: Concatenate the provided meta fields and text passage / table to a text pair that is
then used to create the embedding.
This approach is also used in the TableTextRetriever paper and is likely to improve
performance if your titles contain meaningful information for retrieval
(topic, entities etc.).

<a id="dense.EmbeddingRetriever.retrieve"></a>

#### EmbeddingRetriever.retrieve

```python
def retrieve(query: str,
             filters: Optional[Dict[str, Union[Dict, List, str, int, float,
                                               bool]]] = None,
             top_k: Optional[int] = None,
             index: str = None,
             headers: Optional[Dict[str, str]] = None,
             scale_score: bool = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="dense.EmbeddingRetriever.retrieve_batch"></a>

#### EmbeddingRetriever.retrieve\_batch

```python
def retrieve_batch(queries: List[str],
                   filters: Optional[Union[Dict[str, Union[Dict, List, str,
                                                           int, float, bool]],
                                           List[Dict[str,
                                                     Union[Dict, List, str,
                                                           int, float,
                                                           bool]]], ]] = None,
                   top_k: Optional[int] = None,
                   index: str = None,
                   headers: Optional[Dict[str, str]] = None,
                   batch_size: Optional[int] = None,
                   scale_score: bool = None) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the supplied queries.

Returns a list of lists of Documents (one per query).

**Arguments**:

- `queries`: List of query strings.
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions. Can be a single filter that will be applied to each query or a list of filters
(one filter per query).

Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `batch_size`: Number of queries to embed at a time.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different
value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="dense.EmbeddingRetriever.embed_queries"></a>

#### EmbeddingRetriever.embed\_queries

```python
def embed_queries(queries: List[str]) -> np.ndarray
```

Create embeddings for a list of queries.

**Arguments**:

- `queries`: List of queries to embed.

**Returns**:

Embeddings, one per input query, shape: (queries, embedding_dim)

<a id="dense.EmbeddingRetriever.embed_documents"></a>

#### EmbeddingRetriever.embed\_documents

```python
def embed_documents(documents: List[Document]) -> np.ndarray
```

Create embeddings for a list of documents.

**Arguments**:

- `documents`: List of documents to embed.

**Returns**:

Embeddings, one per input document, shape: (docs, embedding_dim)

<a id="dense.EmbeddingRetriever.train"></a>

#### EmbeddingRetriever.train

```python
def train(training_data: List[Dict[str, Any]],
          learning_rate: float = 2e-5,
          n_epochs: int = 1,
          num_warmup_steps: int = None,
          batch_size: int = 16,
          train_loss: str = "mnrl") -> None
```

Trains/adapts the underlying embedding model.

Each training data example is a dictionary with the following keys:

* question: the question string
* pos_doc: the positive document string
* neg_doc: the negative document string
* score: the score margin

**Arguments**:

- `training_data` (`List[Dict[str, Any]]`): The training data
- `learning_rate` (`float`): The learning rate
- `n_epochs` (`int`): The number of epochs
- `num_warmup_steps` (`int`): The number of warmup steps
- `batch_size` (`int (optional)`): The batch size to use for the training, defaults to 16
- `train_loss` (`str (optional)`): The loss to use for training.
If you're using sentence-transformers as embedding_model (which are the only ones that currently support training),
possible values are 'mnrl' (Multiple Negatives Ranking Loss) or 'margin_mse' (MarginMSE).

<a id="dense.EmbeddingRetriever.save"></a>

#### EmbeddingRetriever.save

```python
def save(save_dir: Union[Path, str]) -> None
```

Save the model to the given directory

**Arguments**:

- `save_dir` (`Union[Path, str]`): The directory where the model will be saved

<a id="dense.MultihopEmbeddingRetriever"></a>

## MultihopEmbeddingRetriever

```python
class MultihopEmbeddingRetriever(EmbeddingRetriever)
```

Retriever that applies iterative retrieval using a shared encoder for query and passage.
See original paper for more details:

Xiong, Wenhan, et. al. (2020): "Answering complex open-domain questions with multi-hop dense retrieval"
(https://arxiv.org/abs/2009.12756)

<a id="dense.MultihopEmbeddingRetriever.__init__"></a>

#### MultihopEmbeddingRetriever.\_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore,
             embedding_model: str,
             model_version: Optional[str] = None,
             num_iterations: int = 2,
             use_gpu: bool = True,
             batch_size: int = 32,
             max_seq_len: int = 512,
             model_format: str = "farm",
             pooling_strategy: str = "reduce_mean",
             emb_extraction_layer: int = -1,
             top_k: int = 10,
             progress_bar: bool = True,
             devices: Optional[List[Union[str, torch.device]]] = None,
             use_auth_token: Optional[Union[str, bool]] = None,
             scale_score: bool = True,
             embed_meta_fields: List[str] = [])
```

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `embedding_model`: Local path or name of model in Hugging Face's model hub such as ``'sentence-transformers/all-MiniLM-L6-v2'``
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `num_iterations`: The number of times passages are retrieved, i.e., the number of hops (Defaults to 2.)
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `batch_size`: Number of documents to encode at once.
- `max_seq_len`: Longest length of each document sequence. Maximum number of tokens for the document text. Longer ones will be cut down.
- `model_format`: Name of framework that was used for saving the model or model type. If no model_format is
provided, it will be inferred automatically from the model configuration files.
Options:

- ``'farm'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
- ``'transformers'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
- ``'sentence_transformers'`` (will use `_SentenceTransformersEmbeddingEncoder` as embedding encoder)
- ``'retribert'`` (will use `_RetribertEmbeddingEncoder` as embedding encoder)
- `pooling_strategy`: Strategy for combining the embeddings from the model (for farm / transformers models only).
Options:

- ``'cls_token'`` (sentence vector)
- ``'reduce_mean'`` (sentence vector)
- ``'reduce_max'`` (sentence vector)
- ``'per_token'`` (individual token vectors)
- `emb_extraction_layer`: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
Default: -1 (very last layer).
- `top_k`: How many documents to return per query.
- `progress_bar`: If true displays progress bar during embedding.
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.
Note: As multi-GPU training is currently not implemented for EmbeddingRetriever,
training will only use the first device provided in this list.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
- `embed_meta_fields`: Concatenate the provided meta fields and text passage / table to a text pair that is
then used to create the embedding.
This approach is also used in the TableTextRetriever paper and is likely to improve
performance if your titles contain meaningful information for retrieval
(topic, entities etc.).

<a id="dense.MultihopEmbeddingRetriever.retrieve"></a>

#### MultihopEmbeddingRetriever.retrieve

```python
def retrieve(query: str,
             filters: Optional[Dict[str, Union[Dict, List, str, int, float,
                                               bool]]] = None,
             top_k: Optional[int] = None,
             index: str = None,
             headers: Optional[Dict[str, str]] = None,
             scale_score: bool = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="dense.MultihopEmbeddingRetriever.retrieve_batch"></a>

#### MultihopEmbeddingRetriever.retrieve\_batch

```python
def retrieve_batch(queries: List[str],
                   filters: Optional[Union[Dict[str, Union[Dict, List, str,
                                                           int, float, bool]],
                                           List[Dict[str,
                                                     Union[Dict, List, str,
                                                           int, float,
                                                           bool]]], ]] = None,
                   top_k: Optional[int] = None,
                   index: str = None,
                   headers: Optional[Dict[str, str]] = None,
                   batch_size: Optional[int] = None,
                   scale_score: bool = None) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the supplied queries.

If you supply a single query, a single list of Documents is returned. If you supply a list of queries, a list of
lists of Documents (one per query) is returned.

**Arguments**:

- `queries`: Single query string or list of queries.
- `filters`: Optional filters to narrow down the search space to documents whose metadata fulfill certain
conditions. Can be a single filter that will be applied to each query or a list of filters
(one filter per query).

Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.

    __Example__:
    ```python
    filters = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": {"$in": ["economy", "politics"]},
                "publisher": {"$eq": "nytimes"}
            }
        }
    }
    # or simpler using default operators
    filters = {
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
    }
    ```

    To use the same logical operator multiple times on the same level, logical operators take
    optionally a list of dictionaries as value.

    __Example__:
    ```python
    filters = {
        "$or": [
            {
                "$and": {
                    "Type": "News Paper",
                    "Date": {
                        "$lt": "2019-01-01"
                    }
                }
            },
            {
                "$and": {
                    "Type": "Blog Post",
                    "Date": {
                        "$gte": "2019-01-01"
                    }
                }
            }
        ]
    }
    ```
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `batch_size`: Number of queries to embed at a time.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true similarity scores (e.g. cosine or dot_product) which naturally have a different
value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="text2sparql"></a>

# Module text2sparql

<a id="text2sparql.Text2SparqlRetriever"></a>

## Text2SparqlRetriever

```python
class Text2SparqlRetriever(BaseGraphRetriever)
```

Graph retriever that uses a pre-trained Bart model to translate natural language questions
given in text form to queries in SPARQL format.
The generated SPARQL query is executed on a knowledge graph.

<a id="text2sparql.Text2SparqlRetriever.__init__"></a>

#### Text2SparqlRetriever.\_\_init\_\_

```python
def __init__(knowledge_graph: BaseKnowledgeGraph,
             model_name_or_path: str = None,
             model_version: Optional[str] = None,
             top_k: int = 1,
             use_auth_token: Optional[Union[str, bool]] = None)
```

Init the Retriever by providing a knowledge graph and a pre-trained BART model

**Arguments**:

- `knowledge_graph`: An instance of BaseKnowledgeGraph on which to execute SPARQL queries.
- `model_name_or_path`: Name of or path to a pre-trained BartForConditionalGeneration model.
- `model_version`: The version of the model to use for entity extraction.
- `top_k`: How many SPARQL queries to generate per text query.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a id="text2sparql.Text2SparqlRetriever.retrieve"></a>

#### Text2SparqlRetriever.retrieve

```python
def retrieve(query: str, top_k: Optional[int] = None)
```

Translate a text query to SPARQL and execute it on the knowledge graph to retrieve a list of answers

**Arguments**:

- `query`: Text query that shall be translated to SPARQL and then executed on the knowledge graph
- `top_k`: How many SPARQL queries to generate per text query.

<a id="text2sparql.Text2SparqlRetriever.retrieve_batch"></a>

#### Text2SparqlRetriever.retrieve\_batch

```python
def retrieve_batch(queries: List[str], top_k: Optional[int] = None)
```

Translate a list of queries to SPARQL and execute it on the knowledge graph to retrieve

a list of lists of answers (one per query).

**Arguments**:

- `queries`: List of queries that shall be translated to SPARQL and then executed on the
knowledge graph.
- `top_k`: How many SPARQL queries to generate per text query.

<a id="text2sparql.Text2SparqlRetriever.format_result"></a>

#### Text2SparqlRetriever.format\_result

```python
def format_result(result)
```

Generate formatted dictionary output with text answer and additional info

**Arguments**:

- `result`: The result of a SPARQL query as retrieved from the knowledge graph
