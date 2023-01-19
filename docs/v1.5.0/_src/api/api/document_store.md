<a id="base"></a>

# Module base

<a id="base.BaseKnowledgeGraph"></a>

## BaseKnowledgeGraph

```python
class BaseKnowledgeGraph(BaseComponent)
```

Base class for implementing Knowledge Graphs.

<a id="base.BaseDocumentStore"></a>

## BaseDocumentStore

```python
class BaseDocumentStore(BaseComponent)
```

Base class for implementing Document Stores.

<a id="base.BaseDocumentStore.write_documents"></a>

#### BaseDocumentStore.write\_documents

```python
@abstractmethod
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Indexes documents for later queries.

**Arguments**:

- `documents`: a list of Python dictionaries or a list of Haystack Document objects.
For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
Optionally: Include meta data via {"text": "<the-actual-text>",
"meta":{"name": "<some-document-name>, "author": "somebody", ...}}
It can be used for filtering and is accessible in the responses of the Finder.
- `index`: Optional name of index where the documents shall be written to.
If None, the DocumentStore's default index (self.index) will be used.
- `batch_size`: Number of documents that are passed to bulk function at a time.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

**Returns**:

None

<a id="base.BaseDocumentStore.get_all_documents"></a>

#### BaseDocumentStore.get\_all\_documents

```python
@abstractmethod
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: Number of documents that are passed to bulk function at a time.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="base.BaseDocumentStore.get_all_documents_generator"></a>

#### BaseDocumentStore.get\_all\_documents\_generator

```python
@abstractmethod
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
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
```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="base.BaseDocumentStore.get_all_labels_aggregated"></a>

#### BaseDocumentStore.get\_all\_labels\_aggregated

```python
def get_all_labels_aggregated(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, open_domain: bool = True, drop_negative_labels: bool = False, drop_no_answers: bool = False, aggregate_by_meta: Optional[Union[str, list]] = None, headers: Optional[Dict[str, str]] = None) -> List[MultiLabel]
```

Return all labels in the DocumentStore, aggregated into MultiLabel objects.

This aggregation step helps, for example, if you collected multiple possible answers for one question and you
want now all answers bundled together in one place for evaluation.
How they are aggregated is defined by the open_domain and aggregate_by_meta parameters.
If the questions are being asked to a single document (i.e. SQuAD style), you should set open_domain=False to aggregate by question and document.
If the questions are being asked to your full collection of documents, you should set open_domain=True to aggregate just by question.
If the questions are being asked to a subslice of your document set (e.g. product review use cases),
you should set open_domain=True and populate aggregate_by_meta with the names of Label meta fields to aggregate by question and your custom meta fields.
For example, in a product review use case, you might set aggregate_by_meta=["product_id"] so that Labels
with the same question but different answers from different documents are aggregated into the one MultiLabel
object, provided that they have the same product_id (to be found in Label.meta["product_id"])

**Arguments**:

- `index`: Name of the index to get the labels from. If None, the
DocumentStore's default index (self.index) will be used.
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
    ```
- `open_domain`: When True, labels are aggregated purely based on the question text alone.
When False, labels are aggregated in a closed domain fashion based on the question text
and also the id of the document that the label is tied to. In this setting, this function
might return multiple MultiLabel objects with the same question string.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
- `aggregate_by_meta`: The names of the Label meta fields by which to aggregate. For example: ["product_id"]
TODO drop params

<a id="base.BaseDocumentStore.normalize_embedding"></a>

#### BaseDocumentStore.normalize\_embedding

```python
def normalize_embedding(emb: np.ndarray) -> None
```

Performs L2 normalization of embeddings vector inplace. Input can be a single vector (1D array) or a matrix
(2D array).

<a id="base.BaseDocumentStore.add_eval_data"></a>

#### BaseDocumentStore.add\_eval\_data

```python
def add_eval_data(filename: str, doc_index: str = "eval_document", label_index: str = "label", batch_size: Optional[int] = None, preprocessor: Optional[PreProcessor] = None, max_docs: Union[int, bool] = None, open_domain: bool = False, headers: Optional[Dict[str, str]] = None)
```

Adds a SQuAD-formatted file to the DocumentStore in order to be able to perform evaluation on it.

If a jsonl file and a batch_size is passed to the function, documents are loaded batchwise
from disk and also indexed batchwise to the DocumentStore in order to prevent out of memory errors.

**Arguments**:

- `filename`: Name of the file containing evaluation data (json or jsonl)
- `doc_index`: Elasticsearch index where evaluation documents should be stored
- `label_index`: Elasticsearch index where labeled questions should be stored
- `batch_size`: Optional number of documents that are loaded and processed at a time.
When set to None (default) all documents are processed at once.
- `preprocessor`: Optional PreProcessor to preprocess evaluation documents.
It can be used for splitting documents into passages (and assigning labels to corresponding passages).
Currently the PreProcessor does not support split_by sentence, cleaning nor split_overlap != 0.
When set to None (default) preprocessing is disabled.
- `max_docs`: Optional number of documents that will be loaded.
When set to None (default) all available eval documents are used.
- `open_domain`: Set this to True if your file is an open domain dataset where two different answers to the
same question might be found in different contexts.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="base.BaseDocumentStore.delete_index"></a>

#### BaseDocumentStore.delete\_index

```python
@abstractmethod
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="base.BaseDocumentStore.run"></a>

#### BaseDocumentStore.run

```python
def run(documents: List[Union[dict, Document]], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, id_hash_keys: Optional[List[str]] = None)
```

Run requests of document stores

Comment: We will gradually introduce the primitives. The doument stores also accept dicts and parse them to documents.
In the future, however, only documents themselves will be accepted. Parsing the dictionaries in the run function
is therefore only an interim solution until the run function also accepts documents.

**Arguments**:

- `documents`: A list of dicts that are documents.
- `headers`: A list of headers.
- `index`: Optional name of index where the documents shall be written to.
If None, the DocumentStore's default index (self.index) will be used.
- `id_hash_keys`: List of the fields that the hashes of the ids are generated from.

<a id="base.BaseDocumentStore.describe_documents"></a>

#### BaseDocumentStore.describe\_documents

```python
def describe_documents(index=None)
```

Return a summary of the documents in the document store

<a id="base.KeywordDocumentStore"></a>

## KeywordDocumentStore

```python
class KeywordDocumentStore(BaseDocumentStore)
```

Base class for implementing Document Stores that support keyword searches.

<a id="base.KeywordDocumentStore.query"></a>

#### KeywordDocumentStore.query

```python
@abstractmethod
def query(query: Optional[str], filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, all_terms_must_match: bool = False, scale_score: bool = True) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query as defined by keyword matching algorithms like BM25.

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
- `custom_query`: Custom query to be executed.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
- `all_terms_must_match`: Whether all terms of the query must match the document.
If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
Defaults to False.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="base.KeywordDocumentStore.query_batch"></a>

#### KeywordDocumentStore.query\_batch

```python
@abstractmethod
def query_batch(queries: List[str], filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, all_terms_must_match: bool = False, scale_score: bool = True) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the provided queries as defined by keyword matching algorithms like BM25.

This method lets you find relevant documents for a single query string (output: List of Documents), or a
a list of query strings (output: List of Lists of Documents).

**Arguments**:

- `queries`: Single query or list of queries.
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
- `custom_query`: Custom query to be executed.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
- `all_terms_must_match`: Whether all terms of the query must match the document.
If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
Defaults to False.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="base.get_batches_from_generator"></a>

#### get\_batches\_from\_generator

```python
def get_batches_from_generator(iterable, n)
```

Batch elements of an iterable into fixed-length chunks or blocks.

<a id="elasticsearch"></a>

# Module elasticsearch

<a id="elasticsearch.ElasticsearchDocumentStore"></a>

## ElasticsearchDocumentStore

```python
class ElasticsearchDocumentStore(KeywordDocumentStore)
```

<a id="elasticsearch.ElasticsearchDocumentStore.__init__"></a>

#### ElasticsearchDocumentStore.\_\_init\_\_

```python
def __init__(host: Union[str, List[str]] = "localhost", port: Union[int, List[int]] = 9200, username: str = "", password: str = "", api_key_id: Optional[str] = None, api_key: Optional[str] = None, aws4auth=None, index: str = "document", label_index: str = "label", search_fields: Union[str, list] = "content", content_field: str = "content", name_field: str = "name", embedding_field: str = "embedding", embedding_dim: int = 768, custom_mapping: Optional[dict] = None, excluded_meta_data: Optional[list] = None, analyzer: str = "standard", scheme: str = "http", ca_certs: Optional[str] = None, verify_certs: bool = True, recreate_index: bool = False, create_index: bool = True, refresh_type: str = "wait_for", similarity: str = "dot_product", timeout: int = 30, return_embedding: bool = False, duplicate_documents: str = "overwrite", index_type: str = "flat", scroll: str = "1d", skip_missing_embeddings: bool = True, synonyms: Optional[List] = None, synonym_type: str = "synonym", use_system_proxy: bool = False)
```

A DocumentStore using Elasticsearch to store and query the documents for our search.

* Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or boosts to your queries, and storing embeddings
    * You can either use an existing Elasticsearch index or create a new one via haystack
    * Retrievers operate on top of this DocumentStore to find the relevant documents for a query

**Arguments**:

- `host`: url(s) of elasticsearch nodes
- `port`: port(s) of elasticsearch nodes
- `username`: username (standard authentication via http_auth)
- `password`: password (standard authentication via http_auth)
- `api_key_id`: ID of the API key (altenative authentication mode to the above http_auth)
- `api_key`: Secret value of the API key (altenative authentication mode to the above http_auth)
- `aws4auth`: Authentication for usage with aws elasticsearch (can be generated with the requests-aws4auth package)
- `index`: Name of index in elasticsearch to use for storing the documents that we want to search. If not existing yet, we will create one.
- `label_index`: Name of index in elasticsearch to use for storing labels. If not existing yet, we will create one.
- `search_fields`: Name of fields used by BM25Retriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
- `content_field`: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
- `name_field`: Name of field that contains the title of the the doc
- `embedding_field`: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
- `embedding_dim`: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
- `custom_mapping`: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
- `analyzer`: Specify the default analyzer from one of the built-ins when creating a new Elasticsearch Index.
Elasticsearch also has built-in analyzers for different languages (e.g. impacting tokenization). More info at:
https://www.elastic.co/guide/en/elasticsearch/reference/7.9/analysis-analyzers.html
- `excluded_meta_data`: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
- `scheme`: 'https' or 'http', protocol used to connect to your elasticsearch instance
- `ca_certs`: Root certificates for SSL: it is a path to certificate authority (CA) certs on disk. You can use certifi package with certifi.where() to find where the CA certs file is located in your machine.
- `verify_certs`: Whether to be strict about ca certificates
- `recreate_index`: If set to True, an existing elasticsearch index will be deleted and a new one will be
created using the config you are using for initialization. Be aware that all data in the old index will be
lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
be recreated.
- `create_index`: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case)
..deprecated:: 2.0
This param is deprecated. In the next major version we will always try to create an index if there is no
existing index (the current behaviour when create_index=True). If you are looking to recreate an
existing index by deleting it first if it already exist use param recreate_index.
- `refresh_type`: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
If set to 'wait_for', continue only after changes are visible (slow, but safe).
If set to 'false', continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion).
More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default since it is
more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
- `timeout`: Number of seconds after which an ElasticSearch request times out.
- `return_embedding`: To return document embedding
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `index_type`: The type of index to be created. Choose from 'flat' and 'hnsw'. Currently the
ElasticsearchDocumentStore does not support HNSW but OpenDistroElasticsearchDocumentStore does.
- `scroll`: Determines how long the current index is fixed, e.g. during updating all documents with embeddings.
Defaults to "1d" and should not be larger than this. Can also be in minutes "5m" or hours "15h"
For details, see https://www.elastic.co/guide/en/elasticsearch/reference/current/scroll-api.html
- `skip_missing_embeddings`: Parameter to control queries based on vector similarity when indexed documents miss embeddings.
Parameter options: (True, False)
False: Raises exception if one or more documents do not have embeddings at query time
True: Query will ignore all documents without embeddings (recommended if you concurrently index and query)
- `synonyms`: List of synonyms can be passed while elasticsearch initialization.
For example: [ "foo, bar => baz",
               "foozball , foosball" ]
More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
- `synonym_type`: Synonym filter type can be passed.
Synonym or Synonym_graph to handle synonyms, including multi-word synonyms correctly during the analysis process.
More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-graph-tokenfilter.html
- `use_system_proxy`: Whether to use system proxy.

<a id="elasticsearch.ElasticsearchDocumentStore.get_document_by_id"></a>

#### ElasticsearchDocumentStore.get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a id="elasticsearch.ElasticsearchDocumentStore.get_documents_by_id"></a>

#### ElasticsearchDocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings. Be aware that passing a large number of ids might lead
to performance issues. Note that Elasticsearch limits the number of results to 10,000 documents by default.

<a id="elasticsearch.ElasticsearchDocumentStore.get_metadata_values_by_key"></a>

#### ElasticsearchDocumentStore.get\_metadata\_values\_by\_key

```python
def get_metadata_values_by_key(key: str, query: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> List[dict]
```

Get values associated with a metadata key. The output is in the format:

[{"value": "my-value-1", "count": 23}, {"value": "my-value-2", "count": 12}, ... ]

**Arguments**:

- `key`: the meta key name to get the values for.
- `query`: narrow down the scope to documents matching the query string.
- `filters`: Narrow down the scope to documents that match the given filters.
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
    ```
- `index`: Elasticsearch index where the meta values should be searched. If not supplied,
self.index will be used.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.write_documents"></a>

#### ElasticsearchDocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Indexes documents for later queries in Elasticsearch.

Behaviour if a document with the same ID already exists in ElasticSearch:
a) (Default) Throw Elastic's standard error message for duplicate IDs.
b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
(This is only relevant if you pass your own ID when initializing a `Document`.
If don't set custom IDs for your Documents or just pass a list of dictionaries here,
they will automatically get UUIDs assigned. See the `Document` class for details)

**Arguments**:

- `documents`: a list of Python dictionaries or a list of Haystack Document objects.
For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
Optionally: Include meta data via {"content": "<the-actual-text>",
"meta":{"name": "<some-document-name>, "author": "somebody", ...}}
It can be used for filtering and is accessible in the responses of the Finder.
Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
should be changed to what you have set for self.content_field and self.name_field.
- `index`: Elasticsearch index where the documents should be indexed. If not supplied, self.index will be used.
- `batch_size`: Number of documents that are passed to Elasticsearch's bulk function at a time.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Raises**:

- `DuplicateDocumentError`: Exception trigger on duplicate document

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.write_labels"></a>

#### ElasticsearchDocumentStore.write\_labels

```python
def write_labels(labels: Union[List[Label], List[dict]], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, batch_size: int = 10_000)
```

Write annotation labels into document store.

**Arguments**:

- `labels`: A list of Python dictionaries or a list of Haystack Label objects.
- `index`: Elasticsearch index where the labels should be stored. If not supplied, self.label_index will be used.
- `batch_size`: Number of labels that are passed to Elasticsearch's bulk function at a time.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.update_document_meta"></a>

#### ElasticsearchDocumentStore.update\_document\_meta

```python
def update_document_meta(id: str, meta: Dict[str, str], headers: Optional[Dict[str, str]] = None, index: str = None)
```

Update the metadata dictionary of a document by specifying its string id

<a id="elasticsearch.ElasticsearchDocumentStore.get_document_count"></a>

#### ElasticsearchDocumentStore.get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="elasticsearch.ElasticsearchDocumentStore.get_label_count"></a>

#### ElasticsearchDocumentStore.get\_label\_count

```python
def get_label_count(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of labels in the document store

<a id="elasticsearch.ElasticsearchDocumentStore.get_embedding_count"></a>

#### ElasticsearchDocumentStore.get\_embedding\_count

```python
def get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the count of embeddings in the document store.

<a id="elasticsearch.ElasticsearchDocumentStore.get_all_documents"></a>

#### ElasticsearchDocumentStore.get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.get_all_documents_generator"></a>

#### ElasticsearchDocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.get_all_labels"></a>

#### ElasticsearchDocumentStore.get\_all\_labels

```python
def get_all_labels(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None, batch_size: int = 10_000) -> List[Label]
```

Return all labels in the document store

<a id="elasticsearch.ElasticsearchDocumentStore.query"></a>

#### ElasticsearchDocumentStore.query

```python
def query(query: Optional[str], filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, all_terms_must_match: bool = False, scale_score: bool = True) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query as defined by the BM25 algorithm.

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
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
- `all_terms_must_match`: Whether all terms of the query must match the document.
If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
Defaults to false.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="elasticsearch.ElasticsearchDocumentStore.query_batch"></a>

#### ElasticsearchDocumentStore.query\_batch

```python
def query_batch(queries: List[str], filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, all_terms_must_match: bool = False, scale_score: bool = True) -> List[List[Document]]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the provided queries as defined by keyword matching algorithms like BM25.

This method lets you find relevant documents for list of query strings (output: List of Lists of Documents).

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
- `custom_query`: Custom query to be executed.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
- `all_terms_must_match`: Whether all terms of the query must match the document.
If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
Defaults to False.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="elasticsearch.ElasticsearchDocumentStore.query_by_embedding"></a>

#### ElasticsearchDocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
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
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="elasticsearch.ElasticsearchDocumentStore.update_embeddings"></a>

#### ElasticsearchDocumentStore.update\_embeddings

```python
def update_embeddings(retriever, index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.

This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to update the embeddings.
- `index`: Index name to update
- `update_existing_embeddings`: Whether to update existing embeddings of the documents. If set to False,
only documents without embeddings are processed. This mode can be used for
incremental updating of embeddings, wherein, only newly indexed documents
get processed.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
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
    ```
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_all_documents"></a>

#### ElasticsearchDocumentStore.delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.
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
    ```
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_documents"></a>

#### ElasticsearchDocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the documents from. If None, the
DocumentStore's default index (self.index) will be used
- `ids`: Optional list of IDs to narrow down the documents to be deleted.
- `filters`: Optional filters to narrow down the documents to be deleted.
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
    ```

    If filters are provided along with a list of IDs, this method deletes the
    intersection of the two query results (documents that match the filters and
    have their ID in the list).
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_labels"></a>

#### ElasticsearchDocumentStore.delete\_labels

```python
def delete_labels(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete labels in an index. All labels are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the labels from. If None, the
DocumentStore's default label index (self.label_index) will be used
- `ids`: Optional list of IDs to narrow down the labels to be deleted.
- `filters`: Optional filters to narrow down the labels to be deleted.
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
    ```
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_index"></a>

#### ElasticsearchDocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing elasticsearch index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="elasticsearch.OpenSearchDocumentStore"></a>

## OpenSearchDocumentStore

```python
class OpenSearchDocumentStore(ElasticsearchDocumentStore)
```

<a id="elasticsearch.OpenSearchDocumentStore.__init__"></a>

#### OpenSearchDocumentStore.\_\_init\_\_

```python
def __init__(scheme: str = "https", username: str = "admin", password: str = "admin", host: Union[str, List[str]] = "localhost", port: Union[int, List[int]] = 9200, api_key_id: Optional[str] = None, api_key: Optional[str] = None, aws4auth=None, index: str = "document", label_index: str = "label", search_fields: Union[str, list] = "content", content_field: str = "content", name_field: str = "name", embedding_field: str = "embedding", embedding_dim: int = 768, custom_mapping: Optional[dict] = None, excluded_meta_data: Optional[list] = None, analyzer: str = "standard", ca_certs: Optional[str] = None, verify_certs: bool = False, recreate_index: bool = False, create_index: bool = True, refresh_type: str = "wait_for", similarity: str = "dot_product", timeout: int = 30, return_embedding: bool = False, duplicate_documents: str = "overwrite", index_type: str = "flat", scroll: str = "1d", skip_missing_embeddings: bool = True, synonyms: Optional[List] = None, synonym_type: str = "synonym", use_system_proxy: bool = False)
```

Document Store using OpenSearch (https://opensearch.org/). It is compatible with the AWS Elasticsearch Service.

In addition to native Elasticsearch query & filtering, it provides efficient vector similarity search using
the KNN plugin that can scale to a large number of documents.

**Arguments**:

- `host`: url(s) of elasticsearch nodes
- `port`: port(s) of elasticsearch nodes
- `username`: username (standard authentication via http_auth)
- `password`: password (standard authentication via http_auth)
- `api_key_id`: ID of the API key (altenative authentication mode to the above http_auth)
- `api_key`: Secret value of the API key (altenative authentication mode to the above http_auth)
- `aws4auth`: Authentication for usage with aws elasticsearch (can be generated with the requests-aws4auth package)
- `index`: Name of index in elasticsearch to use for storing the documents that we want to search. If not existing yet, we will create one.
- `label_index`: Name of index in elasticsearch to use for storing labels. If not existing yet, we will create one.
- `search_fields`: Name of fields used by BM25Retriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
- `content_field`: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
- `name_field`: Name of field that contains the title of the the doc
- `embedding_field`: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
Note, that in OpenSearch the similarity type for efficient approximate vector similarity calculations is tied to the embedding field's data type which cannot be changed after creation.
- `embedding_dim`: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
- `custom_mapping`: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
- `analyzer`: Specify the default analyzer from one of the built-ins when creating a new Elasticsearch Index.
Elasticsearch also has built-in analyzers for different languages (e.g. impacting tokenization). More info at:
https://www.elastic.co/guide/en/elasticsearch/reference/7.9/analysis-analyzers.html
- `excluded_meta_data`: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
- `scheme`: 'https' or 'http', protocol used to connect to your elasticsearch instance
- `ca_certs`: Root certificates for SSL: it is a path to certificate authority (CA) certs on disk. You can use certifi package with certifi.where() to find where the CA certs file is located in your machine.
- `verify_certs`: Whether to be strict about ca certificates
- `create_index`: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case
- `refresh_type`: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
If set to 'wait_for', continue only after changes are visible (slow, but safe).
If set to 'false', continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion).
More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default since it is
more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
Note, that the use of efficient approximate vector calculations in OpenSearch is tied to embedding_field's data type which cannot be changed after creation.
You won't be able to use approximate vector calculations on an embedding_field which was created with a different similarity value.
In such cases a fallback to exact but slow vector calculations will happen and a warning will be displayed.
- `timeout`: Number of seconds after which an ElasticSearch request times out.
- `return_embedding`: To return document embedding
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `index_type`: The type of index to be created. Choose from 'flat' and 'hnsw'.
As OpenSearch currently does not support all similarity functions (e.g. dot_product) in exact vector similarity calculations,
we don't make use of exact vector similarity when index_type='flat'. Instead we use the same approximate vector similarity calculations like in 'hnsw', but further optimized for accuracy.
Exact vector similarity is only used as fallback when there's a mismatch between certain requested and indexed similarity types.
In these cases however, a warning will be displayed. See similarity param for more information.
- `scroll`: Determines how long the current index is fixed, e.g. during updating all documents with embeddings.
Defaults to "1d" and should not be larger than this. Can also be in minutes "5m" or hours "15h"
For details, see https://www.elastic.co/guide/en/elasticsearch/reference/current/scroll-api.html
- `skip_missing_embeddings`: Parameter to control queries based on vector similarity when indexed documents miss embeddings.
Parameter options: (True, False)
False: Raises exception if one or more documents do not have embeddings at query time
True: Query will ignore all documents without embeddings (recommended if you concurrently index and query)
- `synonyms`: List of synonyms can be passed while elasticsearch initialization.
For example: [ "foo, bar => baz",
               "foozball , foosball" ]
More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
- `synonym_type`: Synonym filter type can be passed.
Synonym or Synonym_graph to handle synonyms, including multi-word synonyms correctly during the analysis process.
More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-graph-tokenfilter.html

<a id="elasticsearch.OpenSearchDocumentStore.query_by_embedding"></a>

#### OpenSearchDocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
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
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="elasticsearch.OpenDistroElasticsearchDocumentStore"></a>

## OpenDistroElasticsearchDocumentStore

```python
class OpenDistroElasticsearchDocumentStore(OpenSearchDocumentStore)
```

A DocumentStore which has an Open Distro for Elasticsearch service behind it.

<a id="memory"></a>

# Module memory

<a id="memory.InMemoryDocumentStore"></a>

## InMemoryDocumentStore

```python
class InMemoryDocumentStore(BaseDocumentStore)
```

In-memory document store

<a id="memory.InMemoryDocumentStore.__init__"></a>

#### InMemoryDocumentStore.\_\_init\_\_

```python
def __init__(index: str = "document", label_index: str = "label", embedding_field: Optional[str] = "embedding", embedding_dim: int = 768, return_embedding: bool = False, similarity: str = "dot_product", progress_bar: bool = True, duplicate_documents: str = "overwrite", use_gpu: bool = True, scoring_batch_size: int = 500000)
```

**Arguments**:

- `index`: The documents are scoped to an index attribute that can be used when writing, querying,
or deleting documents. This parameter sets the default value for document index.
- `label_index`: The default value of index attribute for the labels.
- `embedding_field`: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
- `embedding_dim`: The size of the embedding vector.
- `return_embedding`: To return document embedding
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default sine it is
more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `use_gpu`: Whether to use a GPU or the CPU for calculating embedding similarity.
Falls back to CPU if no GPU is available.
- `scoring_batch_size`: Batch size of documents to calculate similarity for. Very small batch sizes are inefficent.
Very large batch sizes can overrun GPU memory. In general you want to make sure
you have at least `embedding_dim`*`scoring_batch_size`*4 bytes available in GPU memory.
Since the data is originally stored in CPU memory there is little risk of overruning memory
when running on CPU.

<a id="memory.InMemoryDocumentStore.write_documents"></a>

#### InMemoryDocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Indexes documents for later queries.

**Arguments**:

- `documents`: a list of Python dictionaries or a list of Haystack Document objects.
For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                  Optionally: Include meta data via {"text": "<the-actual-text>",
                  "meta": {"name": "<some-document-name>, "author": "somebody", ...}}
                  It can be used for filtering and is accessible in the responses of the Finder.
:param index: write documents to a custom namespace. For instance, documents for evaluation can be indexed in a
              separate index than the documents for search.
:param duplicate_documents: Handle duplicates document based on parameter options.
                            Parameter options : ( 'skip','overwrite','fail')
                            skip: Ignore the duplicates documents
                            overwrite: Update any existing documents with the same ID when adding documents.
                            fail: an error is raised if the document ID of the document being added already
                            exists.
:raises DuplicateDocumentError: Exception trigger on duplicate document
:return: None

<a id="memory.InMemoryDocumentStore.write_labels"></a>

#### InMemoryDocumentStore.write\_labels

```python
def write_labels(labels: Union[List[dict], List[Label]], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Write annotation labels into document store.

<a id="memory.InMemoryDocumentStore.get_document_by_id"></a>

#### InMemoryDocumentStore.get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string.

<a id="memory.InMemoryDocumentStore.get_documents_by_id"></a>

#### InMemoryDocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings.

<a id="memory.InMemoryDocumentStore.get_scores_torch"></a>

#### InMemoryDocumentStore.get\_scores\_torch

```python
def get_scores_torch(query_emb: np.ndarray, document_to_search: List[Document]) -> List[float]
```

Calculate similarity scores between query embedding and a list of documents using torch.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `document_to_search`: List of documents to compare `query_emb` against.

<a id="memory.InMemoryDocumentStore.get_scores_numpy"></a>

#### InMemoryDocumentStore.get\_scores\_numpy

```python
def get_scores_numpy(query_emb: np.ndarray, document_to_search: List[Document]) -> List[float]
```

Calculate similarity scores between query embedding and a list of documents using numpy.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `document_to_search`: List of documents to compare `query_emb` against.

<a id="memory.InMemoryDocumentStore.query_by_embedding"></a>

#### InMemoryDocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
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
Example:
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
Example:
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
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="memory.InMemoryDocumentStore.update_embeddings"></a>

#### InMemoryDocumentStore.update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.

This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to get embeddings for text
- `index`: Index name for which embeddings are to be updated. If set to None, the default self.index is used.
- `update_existing_embeddings`: Whether to update existing embeddings of the documents. If set to False,
only documents without embeddings are processed. This mode can be used for
incremental updating of embeddings, wherein, only newly indexed documents
get processed.
- `filters`: Narrow down the scope to documents that match the given filters.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.
Example:
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
    ```
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

**Returns**:

None

<a id="memory.InMemoryDocumentStore.get_document_count"></a>

#### InMemoryDocumentStore.get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, Any]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="memory.InMemoryDocumentStore.get_embedding_count"></a>

#### InMemoryDocumentStore.get\_embedding\_count

```python
def get_embedding_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int
```

Return the count of embeddings in the document store.

<a id="memory.InMemoryDocumentStore.get_label_count"></a>

#### InMemoryDocumentStore.get\_label\_count

```python
def get_label_count(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of labels in the document store.

<a id="memory.InMemoryDocumentStore.get_all_documents"></a>

#### InMemoryDocumentStore.get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get all documents from the document store as a list.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Narrow down the scope to documents that match the given filters.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.
Example:
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
    ```
- `return_embedding`: Whether to return the document embeddings.

<a id="memory.InMemoryDocumentStore.get_all_documents_generator"></a>

#### InMemoryDocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get all documents from the document store. The methods returns a Python Generator that yields individual

documents.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Narrow down the scope to documents that match the given filters.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.
Example:
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
    ```
- `return_embedding`: Whether to return the document embeddings.

<a id="memory.InMemoryDocumentStore.get_all_labels"></a>

#### InMemoryDocumentStore.get\_all\_labels

```python
def get_all_labels(index: str = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> List[Label]
```

Return all labels in the document store.

<a id="memory.InMemoryDocumentStore.delete_all_documents"></a>

#### InMemoryDocumentStore.delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Narrow down the scope to documents that match the given filters.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.
Example:
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
    ```

**Returns**:

None

<a id="memory.InMemoryDocumentStore.delete_documents"></a>

#### InMemoryDocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `ids`: Optional list of IDs to narrow down the documents to be deleted.
- `filters`: Narrow down the scope to documents that match the given filters.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.
Example:
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
    ```

**Returns**:

None

<a id="memory.InMemoryDocumentStore.delete_index"></a>

#### InMemoryDocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="memory.InMemoryDocumentStore.delete_labels"></a>

#### InMemoryDocumentStore.delete\_labels

```python
def delete_labels(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete labels in an index. All labels are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the labels from. If None, the
DocumentStore's default label index (self.label_index) will be used.
- `ids`: Optional list of IDs to narrow down the labels to be deleted.
- `filters`: Narrow down the scope to documents that match the given filters.
Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
`"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
Logical operator keys take a dictionary of metadata field names and/or logical operators as
value. Metadata field names take a dictionary of comparison operators as value. Comparison
operator keys take a single value or (in case of `"$in"`) a list of values as value.
If no logical operator is provided, `"$and"` is used as default operation. If no comparison
operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
operation.
Example:
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
    ```

**Returns**:

None

<a id="sql"></a>

# Module sql

<a id="sql.SQLDocumentStore"></a>

## SQLDocumentStore

```python
class SQLDocumentStore(BaseDocumentStore)
```

<a id="sql.SQLDocumentStore.__init__"></a>

#### SQLDocumentStore.\_\_init\_\_

```python
def __init__(url: str = "sqlite://", index: str = "document", label_index: str = "label", duplicate_documents: str = "overwrite", check_same_thread: bool = False, isolation_level: str = None)
```

An SQL backed DocumentStore. Currently supports SQLite, PostgreSQL and MySQL backends.

**Arguments**:

- `url`: URL for SQL database as expected by SQLAlchemy. More info here: https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
- `index`: The documents are scoped to an index attribute that can be used when writing, querying, or deleting documents.
This parameter sets the default value for document index.
- `label_index`: The default value of index attribute for the labels.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `check_same_thread`: Set to False to mitigate multithreading issues in older SQLite versions (see https://docs.sqlalchemy.org/en/14/dialects/sqlite.html?highlight=check_same_thread#threading-pooling-behavior)
- `isolation_level`: see SQLAlchemy's `isolation_level` parameter for `create_engine()` (https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level)

<a id="sql.SQLDocumentStore.get_document_by_id"></a>

#### SQLDocumentStore.get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a id="sql.SQLDocumentStore.get_documents_by_id"></a>

#### SQLDocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings

<a id="sql.SQLDocumentStore.get_documents_by_vector_ids"></a>

#### SQLDocumentStore.get\_documents\_by\_vector\_ids

```python
def get_documents_by_vector_ids(vector_ids: List[str], index: Optional[str] = None, batch_size: int = 10_000)
```

Fetch documents by specifying a list of text vector id strings

<a id="sql.SQLDocumentStore.get_all_documents_generator"></a>

#### SQLDocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="sql.SQLDocumentStore.get_all_labels"></a>

#### SQLDocumentStore.get\_all\_labels

```python
def get_all_labels(index=None, filters: Optional[dict] = None, headers: Optional[Dict[str, str]] = None)
```

Return all labels in the document store

<a id="sql.SQLDocumentStore.write_documents"></a>

#### SQLDocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None
```

Indexes documents for later queries.

**Arguments**:

- `documents`: a list of Python dictionaries or a list of Haystack Document objects.
For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
Optionally: Include meta data via {"text": "<the-actual-text>",
"meta":{"name": "<some-document-name>, "author": "somebody", ...}}
It can be used for filtering and is accessible in the responses of the Finder.
- `index`: add an optional index attribute to documents. It can be later used for filtering. For instance,
documents for evaluation can be indexed in a separate index than the documents for search.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents
but is considerably slower (default).
fail: an error is raised if the document ID of the document being added already
exists.

**Returns**:

None

<a id="sql.SQLDocumentStore.write_labels"></a>

#### SQLDocumentStore.write\_labels

```python
def write_labels(labels, index=None, headers: Optional[Dict[str, str]] = None)
```

Write annotation labels into document store.

<a id="sql.SQLDocumentStore.update_vector_ids"></a>

#### SQLDocumentStore.update\_vector\_ids

```python
def update_vector_ids(vector_id_map: Dict[str, str], index: Optional[str] = None, batch_size: int = 10_000)
```

Update vector_ids for given document_ids.

**Arguments**:

- `vector_id_map`: dict containing mapping of document_id -> vector_id.
- `index`: filter documents by the optional index attribute for documents in database.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="sql.SQLDocumentStore.reset_vector_ids"></a>

#### SQLDocumentStore.reset\_vector\_ids

```python
def reset_vector_ids(index: Optional[str] = None)
```

Set vector IDs for all documents as None

<a id="sql.SQLDocumentStore.update_document_meta"></a>

#### SQLDocumentStore.update\_document\_meta

```python
def update_document_meta(id: str, meta: Dict[str, str], index: str = None)
```

Update the metadata dictionary of a document by specifying its string id

<a id="sql.SQLDocumentStore.get_document_count"></a>

#### SQLDocumentStore.get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, Any]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="sql.SQLDocumentStore.get_label_count"></a>

#### SQLDocumentStore.get\_label\_count

```python
def get_label_count(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of labels in the document store

<a id="sql.SQLDocumentStore.delete_all_documents"></a>

#### SQLDocumentStore.delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a id="sql.SQLDocumentStore.delete_documents"></a>

#### SQLDocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from. If None, the
DocumentStore's default index (self.index) will be used.
- `ids`: Optional list of IDs to narrow down the documents to be deleted.
- `filters`: Optional filters to narrow down the documents to be deleted.
Example filters: {"name": ["some", "more"], "category": ["only_one"]}.
If filters are provided along with a list of IDs, this method deletes the
intersection of the two query results (documents that match the filters and
have their ID in the list).

**Returns**:

None

<a id="sql.SQLDocumentStore.delete_index"></a>

#### SQLDocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="sql.SQLDocumentStore.delete_labels"></a>

#### SQLDocumentStore.delete\_labels

```python
def delete_labels(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete labels from the document store. All labels are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the labels from. If None, the
DocumentStore's default label index (self.label_index) will be used.
- `ids`: Optional list of IDs to narrow down the labels to be deleted.
- `filters`: Optional filters to narrow down the labels to be deleted.
Example filters: {"id": ["9a196e41-f7b5-45b4-bd19-5feb7501c159", "9a196e41-f7b5-45b4-bd19-5feb7501c159"]} or {"query": ["question2"]}

**Returns**:

None

<a id="faiss"></a>

# Module faiss

<a id="faiss.FAISSDocumentStore"></a>

## FAISSDocumentStore

```python
class FAISSDocumentStore(SQLDocumentStore)
```

Document store for very large scale embedding based dense retrievers like the DPR.

It implements the FAISS library(https://github.com/facebookresearch/faiss)
to perform similarity search on vectors.

The document text and meta-data (for filtering) are stored using the SQLDocumentStore, while
the vector embeddings are indexed in a FAISS Index.

<a id="faiss.FAISSDocumentStore.__init__"></a>

#### FAISSDocumentStore.\_\_init\_\_

```python
def __init__(sql_url: str = "sqlite:///faiss_document_store.db", vector_dim: int = None, embedding_dim: int = 768, faiss_index_factory_str: str = "Flat", faiss_index: Optional[faiss.swigfaiss.Index] = None, return_embedding: bool = False, index: str = "document", similarity: str = "dot_product", embedding_field: str = "embedding", progress_bar: bool = True, duplicate_documents: str = "overwrite", faiss_index_path: Union[str, Path] = None, faiss_config_path: Union[str, Path] = None, isolation_level: str = None, n_links: int = 64, ef_search: int = 20, ef_construction: int = 80)
```

**Arguments**:

- `sql_url`: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
deployment, Postgres is recommended.
- `vector_dim`: Deprecated. Use embedding_dim instead.
- `embedding_dim`: The embedding vector size. Default: 768.
- `faiss_index_factory_str`: Create a new FAISS index of the specified type.
The type is determined from the given string following the conventions
of the original FAISS index factory.
Recommended options:
- "Flat" (default): Best accuracy (= exact). Becomes slow and RAM intense for > 1 Mio docs.
- "HNSW": Graph-based heuristic. If not further specified,
          we use the following config:
          HNSW64, efConstruction=80 and efSearch=20
- "IVFx,Flat": Inverted Index. Replace x with the number of centroids aka nlist.
                  Rule of thumb: nlist = 10 * sqrt (num_docs) is a good starting point.
For more details see:
- Overview of indices https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- Guideline for choosing an index https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- FAISS Index factory https://github.com/facebookresearch/faiss/wiki/The-index-factory
Benchmarks: XXX
- `faiss_index`: Pass an existing FAISS Index, i.e. an empty one that you configured manually
or one with docs that you used in Haystack before and want to load again.
- `return_embedding`: To return document embedding. Unlike other document stores, FAISS will return normalized embeddings
- `index`: Name of index in document store to use.
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default since it is
more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence-Transformer model.
In both cases, the returned values in Document.score are normalized to be in range [0,1]:
For `dot_product`: expit(np.asarray(raw_score / 100))
FOr `cosine`: (raw_score + 1) / 2
- `embedding_field`: Name of field containing an embedding vector.
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `faiss_index_path`: Stored FAISS index file. Can be created via calling `save()`.
If specified no other params besides faiss_config_path must be specified.
- `faiss_config_path`: Stored FAISS initial configuration parameters.
Can be created via calling `save()`
- `isolation_level`: see SQLAlchemy's `isolation_level` parameter for `create_engine()` (https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level)
- `n_links`: used only if index_factory == "HNSW"
- `ef_search`: used only if index_factory == "HNSW"
- `ef_construction`: used only if index_factory == "HNSW"

<a id="faiss.FAISSDocumentStore.write_documents"></a>

#### FAISSDocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None
```

Add new documents to the DocumentStore.

**Arguments**:

- `documents`: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
them right away in FAISS. If not, you can later call update_embeddings() to create & index them.
- `index`: (SQL) index name for storing the docs and metadata
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.

**Raises**:

- `DuplicateDocumentError`: Exception trigger on duplicate document

**Returns**:

None

<a id="faiss.FAISSDocumentStore.update_embeddings"></a>

#### FAISSDocumentStore.update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, update_existing_embeddings: bool = True, filters: Optional[Dict[str, Any]] = None, batch_size: int = 10_000)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.

This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to get embeddings for text
- `index`: Index name for which embeddings are to be updated. If set to None, the default self.index is used.
- `update_existing_embeddings`: Whether to update existing embeddings of the documents. If set to False,
only documents without embeddings are processed. This mode can be used for
incremental updating of embeddings, wherein, only newly indexed documents
get processed.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

**Returns**:

None

<a id="faiss.FAISSDocumentStore.get_all_documents_generator"></a>

#### FAISSDocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get all documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings. Unlike other document stores, FAISS will return normalized embeddings
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="faiss.FAISSDocumentStore.get_embedding_count"></a>

#### FAISSDocumentStore.get\_embedding\_count

```python
def get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> int
```

Return the count of embeddings in the document store.

<a id="faiss.FAISSDocumentStore.train_index"></a>

#### FAISSDocumentStore.train\_index

```python
def train_index(documents: Optional[Union[List[dict], List[Document]]], embeddings: Optional[np.ndarray] = None, index: Optional[str] = None)
```

Some FAISS indices (e.g. IVF) require initial "training" on a sample of vectors before you can add your final vectors.

The train vectors should come from the same distribution as your final ones.
You can pass either documents (incl. embeddings) or just the plain embeddings that the index shall be trained on.

**Arguments**:

- `documents`: Documents (incl. the embeddings)
- `embeddings`: Plain embeddings
- `index`: Name of the index to train. If None, the DocumentStore's default index (self.index) will be used.

**Returns**:

None

<a id="faiss.FAISSDocumentStore.delete_all_documents"></a>

#### FAISSDocumentStore.delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete all documents from the document store.

<a id="faiss.FAISSDocumentStore.delete_documents"></a>

#### FAISSDocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents from the document store. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `ids`: Optional list of IDs to narrow down the documents to be deleted.
- `filters`: Optional filters to narrow down the documents to be deleted.
Example filters: {"name": ["some", "more"], "category": ["only_one"]}.
If filters are provided along with a list of IDs, this method deletes the
intersection of the two query results (documents that match the filters and
have their ID in the list).

**Returns**:

None

<a id="faiss.FAISSDocumentStore.delete_index"></a>

#### FAISSDocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="faiss.FAISSDocumentStore.query_by_embedding"></a>

#### FAISSDocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: Index name to query the document from.
- `return_embedding`: To return document embedding. Unlike other document stores, FAISS will return normalized embeddings
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="faiss.FAISSDocumentStore.save"></a>

#### FAISSDocumentStore.save

```python
def save(index_path: Union[str, Path], config_path: Optional[Union[str, Path]] = None)
```

Save FAISS Index to the specified file.

**Arguments**:

- `index_path`: Path to save the FAISS index to.
- `config_path`: Path to save the initial configuration parameters to.
Defaults to the same as the file path, save the extension (.json).
This file contains all the parameters passed to FAISSDocumentStore()
at creation time (for example the SQL path, embedding_dim, etc), and will be
used by the `load` method to restore the index with the appropriate configuration.

**Returns**:

None

<a id="faiss.FAISSDocumentStore.load"></a>

#### FAISSDocumentStore.load

```python
@classmethod
def load(cls, index_path: Union[str, Path], config_path: Optional[Union[str, Path]] = None)
```

Load a saved FAISS index from a file and connect to the SQL database.

Note: In order to have a correct mapping from FAISS to SQL,
      make sure to use the same SQL DB that you used when calling `save()`.

**Arguments**:

- `index_path`: Stored FAISS index file. Can be created via calling `save()`
- `config_path`: Stored FAISS initial configuration parameters.
Can be created via calling `save()`

<a id="milvus1"></a>

# Module milvus1

<a id="milvus1.Milvus1DocumentStore"></a>

## Milvus1DocumentStore

```python
class Milvus1DocumentStore(SQLDocumentStore)
```

Milvus (https://milvus.io/) is a highly reliable, scalable Document Store specialized on storing and processing vectors.
Therefore, it is particularly suited for Haystack users that work with dense retrieval methods (like DPR).
In contrast to FAISS, Milvus ...
 - runs as a separate service (e.g. a Docker container) and can scale easily in a distributed environment
 - allows dynamic data management (i.e. you can insert/delete vectors without recreating the whole index)
 - encapsulates multiple ANN libraries (FAISS, ANNOY ...)

This class uses Milvus for all vector related storage, processing and querying.
The meta-data (e.g. for filtering) and the document text are however stored in a separate SQL Database as Milvus
does not allow these data types (yet).

Usage:
1. Start a Milvus server (see https://milvus.io/docs/v1.0.0/install_milvus.md)
2. Run pip install farm-haystack[milvus1]
3. Init a MilvusDocumentStore in Haystack

<a id="milvus1.Milvus1DocumentStore.__init__"></a>

#### Milvus1DocumentStore.\_\_init\_\_

```python
def __init__(sql_url: str = "sqlite:///", milvus_url: str = "tcp://localhost:19530", connection_pool: str = "SingletonThread", index: str = "document", vector_dim: int = None, embedding_dim: int = 768, index_file_size: int = 1024, similarity: str = "dot_product", index_type: IndexType = IndexType.FLAT, index_param: Optional[Dict[str, Any]] = None, search_param: Optional[Dict[str, Any]] = None, return_embedding: bool = False, embedding_field: str = "embedding", progress_bar: bool = True, duplicate_documents: str = "overwrite", isolation_level: str = None)
```

**WARNING:** Milvus1DocumentStore is deprecated and will be removed in a future version. Please switch to Milvus2

or consider using another DocumentStore.

**Arguments**:

- `sql_url`: SQL connection URL for storing document texts and metadata. It defaults to a local, file based SQLite DB. For large scale
deployment, Postgres is recommended. If using MySQL then same server can also be used for
Milvus metadata. For more details see https://milvus.io/docs/v1.0.0/data_manage.md.
- `milvus_url`: Milvus server connection URL for storing and processing vectors.
Protocol, host and port will automatically be inferred from the URL.
See https://milvus.io/docs/v1.0.0/install_milvus.md for instructions to start a Milvus instance.
- `connection_pool`: Connection pool type to connect with Milvus server. Default: "SingletonThread".
- `index`: Index name for text, embedding and metadata (in Milvus terms, this is the "collection name").
- `vector_dim`: Deprecated. Use embedding_dim instead.
- `embedding_dim`: The embedding vector size. Default: 768.
- `index_file_size`: Specifies the size of each segment file that is stored by Milvus and its default value is 1024 MB.
When the size of newly inserted vectors reaches the specified volume, Milvus packs these vectors into a new segment.
Milvus creates one index file for each segment. When conducting a vector search, Milvus searches all index files one by one.
As a rule of thumb, we would see a 30% ~ 50% increase in the search performance after changing the value of index_file_size from 1024 to 2048.
Note that an overly large index_file_size value may cause failure to load a segment into the memory or graphics memory.
(From https://milvus.io/docs/v1.0.0/performance_faq.md#How-can-I-get-the-best-performance-from-Milvus-through-setting-index_file_size)
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default and recommended for DPR embeddings.
'cosine' is recommended for Sentence Transformers.
- `index_type`: Type of approximate nearest neighbour (ANN) index used. The choice here determines your tradeoff between speed and accuracy.
Some popular options:
- FLAT (default): Exact method, slow
- IVF_FLAT, inverted file based heuristic, fast
- HSNW: Graph based, fast
- ANNOY: Tree based, fast
See: https://milvus.io/docs/v1.0.0/index.md
- `index_param`: Configuration parameters for the chose index_type needed at indexing time.
For example: {"nlist": 16384} as the number of cluster units to create for index_type IVF_FLAT.
See https://milvus.io/docs/v1.0.0/index.md
- `search_param`: Configuration parameters for the chose index_type needed at query time
For example: {"nprobe": 10} as the number of cluster units to query for index_type IVF_FLAT.
See https://milvus.io/docs/v1.0.0/index.md
- `return_embedding`: To return document embedding.
- `embedding_field`: Name of field containing an embedding vector.
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `isolation_level`: see SQLAlchemy's `isolation_level` parameter for `create_engine()` (https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level)

<a id="milvus1.Milvus1DocumentStore.write_documents"></a>

#### Milvus1DocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None, index_param: Optional[Dict[str, Any]] = None)
```

Add new documents to the DocumentStore.

**Arguments**:

- `documents`: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
them right away in Milvus. If not, you can later call update_embeddings() to create & index them.
- `index`: (SQL) index name for storing the docs and metadata
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.

**Raises**:

- `DuplicateDocumentError`: Exception trigger on duplicate document

**Returns**:

None

<a id="milvus1.Milvus1DocumentStore.update_embeddings"></a>

#### Milvus1DocumentStore.update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, batch_size: int = 10_000, update_existing_embeddings: bool = True, filters: Optional[Dict[str, Any]] = None)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.

This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to get embeddings for text
- `index`: (SQL) index name for storing the docs and metadata
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `update_existing_embeddings`: Whether to update existing embeddings of the documents. If set to False,
only documents without embeddings are processed. This mode can be used for
incremental updating of embeddings, wherein, only newly indexed documents
get processed.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
Example: {"name": ["some", "more"], "category": ["only_one"]}

**Returns**:

None

<a id="milvus1.Milvus1DocumentStore.query_by_embedding"></a>

#### Milvus1DocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: (SQL) index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

**Returns**:

list of Documents that are the most similar to `query_emb`

<a id="milvus1.Milvus1DocumentStore.delete_all_documents"></a>

#### Milvus1DocumentStore.delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete all documents (from SQL AND Milvus).

**Arguments**:

- `index`: (SQL) index name for storing the docs and metadata
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}

**Returns**:

None

<a id="milvus1.Milvus1DocumentStore.delete_documents"></a>

#### Milvus1DocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from. If None, the
DocumentStore's default index (self.index) will be used.
- `ids`: Optional list of IDs to narrow down the documents to be deleted.
- `filters`: Optional filters to narrow down the documents to be deleted.
Example filters: {"name": ["some", "more"], "category": ["only_one"]}.
If filters are provided along with a list of IDs, this method deletes the
intersection of the two query results (documents that match the filters and
have their ID in the list).

**Returns**:

None

<a id="milvus1.Milvus1DocumentStore.delete_index"></a>

#### Milvus1DocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="milvus1.Milvus1DocumentStore.get_all_documents_generator"></a>

#### Milvus1DocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get all documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="milvus1.Milvus1DocumentStore.get_all_documents"></a>

#### Milvus1DocumentStore.get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store (optionally using filter criteria).

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="milvus1.Milvus1DocumentStore.get_document_by_id"></a>

#### Milvus1DocumentStore.get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

**Arguments**:

- `id`: ID of the document
- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.

<a id="milvus1.Milvus1DocumentStore.get_documents_by_id"></a>

#### Milvus1DocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch multiple documents by specifying their IDs (strings)

**Arguments**:

- `ids`: List of IDs of the documents
- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `batch_size`: is currently not used

<a id="milvus1.Milvus1DocumentStore.get_all_vectors"></a>

#### Milvus1DocumentStore.get\_all\_vectors

```python
def get_all_vectors(index: Optional[str] = None) -> List[np.ndarray]
```

Helper function to dump all vectors stored in Milvus server.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.

**Returns**:

List[np.array]: List of vectors.

<a id="milvus1.Milvus1DocumentStore.get_embedding_count"></a>

#### Milvus1DocumentStore.get\_embedding\_count

```python
def get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> int
```

Return the count of embeddings in the document store.

<a id="milvus2"></a>

# Module milvus2

<a id="milvus2.Milvus2DocumentStore"></a>

## Milvus2DocumentStore

```python
class Milvus2DocumentStore(SQLDocumentStore)
```

Limitations:
Milvus 2.0 so far doesn't support the deletion of documents (https://github.com/milvus-io/milvus/issues/7130).
Therefore, delete_documents() and update_embeddings() won't work yet.

Differences to 1.x:
Besides big architectural changes that impact performance and reliability 2.0 supports the filtering by scalar data types.
For Haystack users this means you can now run a query using vector similarity and filter for some meta data at the same time!
(See https://milvus.io/docs/v2.0.x/comparison.md for more details)

Usage:
1. Start a Milvus service via docker (see https://milvus.io/docs/v2.0.x/install_standalone-docker.md)
2. Run pip install farm-haystack[milvus]
3. Init a MilvusDocumentStore() in Haystack

Overview:
Milvus (https://milvus.io/) is a highly reliable, scalable Document Store specialized on storing and processing vectors.
Therefore, it is particularly suited for Haystack users that work with dense retrieval methods (like DPR).

In contrast to FAISS, Milvus ...
 - runs as a separate service (e.g. a Docker container) and can scale easily in a distributed environment
 - allows dynamic data management (i.e. you can insert/delete vectors without recreating the whole index)
 - encapsulates multiple ANN libraries (FAISS, ANNOY ...)

This class uses Milvus for all vector related storage, processing and querying.
The meta-data (e.g. for filtering) and the document text are however stored in a separate SQL Database as Milvus
does not allow these data types (yet).

<a id="milvus2.Milvus2DocumentStore.__init__"></a>

#### Milvus2DocumentStore.\_\_init\_\_

```python
def __init__(sql_url: str = "sqlite:///", host: str = "localhost", port: str = "19530", connection_pool: str = "SingletonThread", index: str = "document", vector_dim: int = None, embedding_dim: int = 768, index_file_size: int = 1024, similarity: str = "dot_product", index_type: str = "IVF_FLAT", index_param: Optional[Dict[str, Any]] = None, search_param: Optional[Dict[str, Any]] = None, return_embedding: bool = False, embedding_field: str = "embedding", id_field: str = "id", custom_fields: Optional[List[Any]] = None, progress_bar: bool = True, duplicate_documents: str = "overwrite", isolation_level: str = None, consistency_level: int = 0, recreate_index: bool = False)
```

**Arguments**:

- `sql_url`: SQL connection URL for storing document texts and metadata. It defaults to a local, file based SQLite DB. For large scale
deployment, Postgres is recommended. If using MySQL then same server can also be used for
Milvus metadata. For more details see https://milvus.io/docs/v1.1.0/data_manage.md.
- `milvus_url`: Milvus server connection URL for storing and processing vectors.
Protocol, host and port will automatically be inferred from the URL.
See https://milvus.io/docs/v2.0.x/install_standalone-docker.md for instructions to start a Milvus instance.
- `connection_pool`: Connection pool type to connect with Milvus server. Default: "SingletonThread".
- `index`: Index name for text, embedding and metadata (in Milvus terms, this is the "collection name").
- `vector_dim`: Deprecated. Use embedding_dim instead.
- `embedding_dim`: The embedding vector size. Default: 768.
- `index_file_size`: Specifies the size of each segment file that is stored by Milvus and its default value is 1024 MB.
When the size of newly inserted vectors reaches the specified volume, Milvus packs these vectors into a new segment.
Milvus creates one index file for each segment. When conducting a vector search, Milvus searches all index files one by one.
As a rule of thumb, we would see a 30% ~ 50% increase in the search performance after changing the value of index_file_size from 1024 to 2048.
Note that an overly large index_file_size value may cause failure to load a segment into the memory or graphics memory.
(From https://milvus.io/docs/v2.0.x/performance_faq.md)
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default and recommended for DPR embeddings.
'cosine' is recommended for Sentence Transformers, but is not directly supported by Milvus.
However, you can normalize your embeddings and use `dot_product` to get the same results.
See https://milvus.io/docs/v2.0.x/metric.md.
- `index_type`: Type of approximate nearest neighbour (ANN) index used. The choice here determines your tradeoff between speed and accuracy.
Some popular options:
- FLAT (default): Exact method, slow
- IVF_FLAT, inverted file based heuristic, fast
- HSNW: Graph based, fast
- ANNOY: Tree based, fast
See: https://milvus.io/docs/v2.0.x/index.md
- `index_param`: Configuration parameters for the chose index_type needed at indexing time.
For example: {"nlist": 16384} as the number of cluster units to create for index_type IVF_FLAT.
See https://milvus.io/docs/v2.0.x/index.md
- `search_param`: Configuration parameters for the chose index_type needed at query time
For example: {"nprobe": 10} as the number of cluster units to query for index_type IVF_FLAT.
See https://milvus.io/docs/v2.0.x/index.md
- `return_embedding`: To return document embedding.
- `embedding_field`: Name of field containing an embedding vector.
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `isolation_level`: see SQLAlchemy's `isolation_level` parameter for `create_engine()` (https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level)
- `recreate_index`: If set to True, an existing Milvus index will be deleted and a new one will be
created using the config you are using for initialization. Be aware that all data in the old index will be
lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
be recreated.

<a id="milvus2.Milvus2DocumentStore.write_documents"></a>

#### Milvus2DocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None, index_param: Optional[Dict[str, Any]] = None)
```

Add new documents to the DocumentStore.

**Arguments**:

- `documents`: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
them right away in Milvus. If not, you can later call `update_embeddings()` to create & index them.
- `index`: (SQL) index name for storing the docs and metadata
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.

**Raises**:

- `DuplicateDocumentError`: Exception trigger on duplicate document

<a id="milvus2.Milvus2DocumentStore.update_embeddings"></a>

#### Milvus2DocumentStore.update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, batch_size: int = 10_000, update_existing_embeddings: bool = True, filters: Optional[Dict[str, Any]] = None)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.

This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to get embeddings for text
- `index`: (SQL) index name for storing the docs and metadata
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `update_existing_embeddings`: Whether to update existing embeddings of the documents. If set to False,
only documents without embeddings are processed. This mode can be used for
incremental updating of embeddings, wherein, only newly indexed documents
get processed.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
Example: {"name": ["some", "more"], "category": ["only_one"]}

**Returns**:

None

<a id="milvus2.Milvus2DocumentStore.query_by_embedding"></a>

#### Milvus2DocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: (SQL) index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="milvus2.Milvus2DocumentStore.delete_documents"></a>

#### Milvus2DocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, batch_size: int = 10_000)
```

Delete all documents (from SQL AND Milvus).

**Arguments**:

- `index`: (SQL) index name for storing the docs and metadata
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}

**Returns**:

None

<a id="milvus2.Milvus2DocumentStore.delete_index"></a>

#### Milvus2DocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="milvus2.Milvus2DocumentStore.get_all_documents_generator"></a>

#### Milvus2DocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get all documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="milvus2.Milvus2DocumentStore.get_all_documents"></a>

#### Milvus2DocumentStore.get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store (optionally using filter criteria).

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="milvus2.Milvus2DocumentStore.get_document_by_id"></a>

#### Milvus2DocumentStore.get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

**Arguments**:

- `id`: ID of the document
- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.

<a id="milvus2.Milvus2DocumentStore.get_documents_by_id"></a>

#### Milvus2DocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch multiple documents by specifying their IDs (strings)

**Arguments**:

- `ids`: List of IDs of the documents
- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="milvus2.Milvus2DocumentStore.get_embedding_count"></a>

#### Milvus2DocumentStore.get\_embedding\_count

```python
def get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int
```

Return the count of embeddings in the document store.

<a id="weaviate"></a>

# Module weaviate

<a id="weaviate.WeaviateDocumentStore"></a>

## WeaviateDocumentStore

```python
class WeaviateDocumentStore(BaseDocumentStore)
```

Weaviate is a cloud-native, modular, real-time vector search engine built to scale your machine learning models.
(See https://weaviate.io/developers/weaviate/current/index.html#what-is-weaviate)

Some of the key differences in contrast to FAISS & Milvus:
1. Stores everything in one place: documents, meta data and vectors - so less network overhead when scaling this up
2. Allows combination of vector search and scalar filtering, i.e. you can filter for a certain tag and do dense retrieval on that subset
3. Has less variety of ANN algorithms, as of now only HNSW.
4. Requires document ids to be in uuid-format. If wrongly formatted ids are provided at indexing time they will be replaced with uuids automatically.
5. Only support cosine similarity.

Weaviate python client is used to connect to the server, more details are here
https://weaviate-python-client.readthedocs.io/en/docs/weaviate.html

Usage:
1. Start a Weaviate server (see https://weaviate.io/developers/weaviate/current/getting-started/installation.html)
2. Init a WeaviateDocumentStore in Haystack

Limitations:
The current implementation is not supporting the storage of labels, so you cannot run any evaluation workflows.

<a id="weaviate.WeaviateDocumentStore.__init__"></a>

#### WeaviateDocumentStore.\_\_init\_\_

```python
def __init__(host: Union[str, List[str]] = "http://localhost", port: Union[int, List[int]] = 8080, timeout_config: tuple = (5, 15), username: str = None, password: str = None, index: str = "Document", embedding_dim: int = 768, content_field: str = "content", name_field: str = "name", similarity: str = "cosine", index_type: str = "hnsw", custom_schema: Optional[dict] = None, return_embedding: bool = False, embedding_field: str = "embedding", progress_bar: bool = True, duplicate_documents: str = "overwrite", recreate_index: bool = False)
```

**Arguments**:

- `host`: Weaviate server connection URL for storing and processing documents and vectors.
For more details, refer "https://weaviate.io/developers/weaviate/current/getting-started/installation.html"
- `port`: port of Weaviate instance
- `timeout_config`: Weaviate Timeout config as a tuple of (retries, time out seconds).
- `username`: username (standard authentication via http_auth)
- `password`: password (standard authentication via http_auth)
- `index`: Index name for document text, embedding and metadata (in Weaviate terminology, this is a "Class" in Weaviate schema).
- `embedding_dim`: The embedding vector size. Default: 768.
- `content_field`: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
- `name_field`: Name of field that contains the title of the the doc
- `similarity`: The similarity function used to compare document vectors. 'cosine' is the only currently supported option and default.
'cosine' is recommended for Sentence Transformers.
- `index_type`: Index type of any vector object defined in weaviate schema. The vector index type is pluggable.
Currently, HSNW is only supported.
See: https://weaviate.io/developers/weaviate/current/more-resources/performance.html
- `custom_schema`: Allows to create custom schema in Weaviate, for more details
See https://weaviate.io/developers/weaviate/current/data-schema/schema-configuration.html
- `module_name`: Vectorization module to convert data into vectors. Default is "text2vec-trasnformers"
For more details, See https://weaviate.io/developers/weaviate/current/modules/
- `return_embedding`: To return document embedding.
- `embedding_field`: Name of field containing an embedding vector.
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already exists.
- `recreate_index`: If set to True, an existing Weaviate index will be deleted and a new one will be
created using the config you are using for initialization. Be aware that all data in the old index will be
lost if you choose to recreate the index.

<a id="weaviate.WeaviateDocumentStore.get_document_by_id"></a>

#### WeaviateDocumentStore.get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its uuid string

<a id="weaviate.WeaviateDocumentStore.get_documents_by_id"></a>

#### WeaviateDocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch documents by specifying a list of uuid strings.

<a id="weaviate.WeaviateDocumentStore.write_documents"></a>

#### WeaviateDocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Add new documents to the DocumentStore.

**Arguments**:

- `documents`: List of `Dicts` or List of `Documents`. A dummy embedding vector for each document is automatically generated if it is not provided. The document id needs to be in uuid format. Otherwise a correctly formatted uuid will be automatically generated based on the provided id.
- `index`: index name for storing the docs and metadata
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.

**Raises**:

- `DuplicateDocumentError`: Exception trigger on duplicate document

**Returns**:

None

<a id="weaviate.WeaviateDocumentStore.update_document_meta"></a>

#### WeaviateDocumentStore.update\_document\_meta

```python
def update_document_meta(id: str, meta: Dict[str, Union[List, str, int, float, bool]], index: str = None)
```

Update the metadata dictionary of a document by specifying its string id.
Overwrites only the specified fields, the unspecified ones remain unchanged.

<a id="weaviate.WeaviateDocumentStore.get_embedding_count"></a>

#### WeaviateDocumentStore.get\_embedding\_count

```python
def get_embedding_count(filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, index: Optional[str] = None) -> int
```

Return the number of embeddings in the document store, which is the same as the number of documents since
every document has a default embedding.

<a id="weaviate.WeaviateDocumentStore.get_document_count"></a>

#### WeaviateDocumentStore.get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="weaviate.WeaviateDocumentStore.get_all_documents"></a>

#### WeaviateDocumentStore.get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store.

Note this limitation from the changelog of Weaviate 1.8.0:

.. quote::
    Due to the increasing cost of each page outlined above, there is a limit to
    how many objects can be retrieved using pagination. By default setting the sum
    of offset and limit to higher than 10,000 objects, will lead to an error.
    If you must retrieve more than 10,000 objects, you can increase this limit by
    setting the environment variable `QUERY_MAXIMUM_RESULTS=<desired-value>`.

    Warning: Setting this to arbitrarily high values can make the memory consumption
    of a single query explode and single queries can slow down the entire cluster.
    We recommend setting this value to the lowest possible value that does not
    interfere with your users' expectations.

(https://github.com/semi-technologies/weaviate/releases/tag/v1.8.0)

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="weaviate.WeaviateDocumentStore.get_all_documents_generator"></a>

#### WeaviateDocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

Note this limitation from the changelog of Weaviate 1.8.0:

.. quote::
    Due to the increasing cost of each page outlined above, there is a limit to
    how many objects can be retrieved using pagination. By default setting the sum
    of offset and limit to higher than 10,000 objects, will lead to an error.
    If you must retrieve more than 10,000 objects, you can increase this limit by
    setting the environment variable `QUERY_MAXIMUM_RESULTS=<desired-value>`.

    Warning: Setting this to arbitrarily high values can make the memory consumption
    of a single query explode and single queries can slow down the entire cluster.
    We recommend setting this value to the lowest possible value that does not
    interfere with your users' expectations.

(https://github.com/semi-technologies/weaviate/releases/tag/v1.8.0)

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="weaviate.WeaviateDocumentStore.query"></a>

#### WeaviateDocumentStore.query

```python
def query(query: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, scale_score: bool = True) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query as defined by Weaviate semantic search.

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
- `custom_query`: Custom query that will executed using query.raw method, for more details refer
https://weaviate.io/developers/weaviate/current/graphql-references/filters.html
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="weaviate.WeaviateDocumentStore.query_by_embedding"></a>

#### WeaviateDocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
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
- `top_k`: How many documents to return
- `index`: index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="weaviate.WeaviateDocumentStore.update_embeddings"></a>

#### WeaviateDocumentStore.update\_embeddings

```python
def update_embeddings(retriever, index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.

This can be useful if want to change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to update the embeddings.
- `index`: Index name to update
- `update_existing_embeddings`: Weaviate mandates an embedding while creating the document itself.
This option must be always true for weaviate and it will update the embeddings for all the documents.
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
    ```
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

**Returns**:

None

<a id="weaviate.WeaviateDocumentStore.delete_all_documents"></a>

#### WeaviateDocumentStore.delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
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
    ```

**Returns**:

None

<a id="weaviate.WeaviateDocumentStore.delete_documents"></a>

#### WeaviateDocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from. If None, the
DocumentStore's default index (self.index) will be used.
- `ids`: Optional list of IDs to narrow down the documents to be deleted.
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
    ```
    If filters are provided along with a list of IDs, this method deletes the
    intersection of the two query results (documents that match the filters and
    have their ID in the list).

**Returns**:

None

<a id="weaviate.WeaviateDocumentStore.delete_index"></a>

#### WeaviateDocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="weaviate.WeaviateDocumentStore.delete_labels"></a>

#### WeaviateDocumentStore.delete\_labels

```python
def delete_labels()
```

Implemented to respect BaseDocumentStore's contract.

Weaviate does not support labels (yet).

<a id="weaviate.WeaviateDocumentStore.get_all_labels"></a>

#### WeaviateDocumentStore.get\_all\_labels

```python
def get_all_labels()
```

Implemented to respect BaseDocumentStore's contract.

Weaviate does not support labels (yet).

<a id="weaviate.WeaviateDocumentStore.get_label_count"></a>

#### WeaviateDocumentStore.get\_label\_count

```python
def get_label_count()
```

Implemented to respect BaseDocumentStore's contract.

Weaviate does not support labels (yet).

<a id="weaviate.WeaviateDocumentStore.write_labels"></a>

#### WeaviateDocumentStore.write\_labels

```python
def write_labels()
```

Implemented to respect BaseDocumentStore's contract.

Weaviate does not support labels (yet).

<a id="graphdb"></a>

# Module graphdb

<a id="graphdb.GraphDBKnowledgeGraph"></a>

## GraphDBKnowledgeGraph

```python
class GraphDBKnowledgeGraph(BaseKnowledgeGraph)
```

Knowledge graph store that runs on a GraphDB instance.

<a id="graphdb.GraphDBKnowledgeGraph.__init__"></a>

#### GraphDBKnowledgeGraph.\_\_init\_\_

```python
def __init__(host: str = "localhost", port: int = 7200, username: str = "", password: str = "", index: Optional[str] = None, prefixes: str = "")
```

Init the knowledge graph by defining the settings to connect with a GraphDB instance

**Arguments**:

- `host`: address of server where the GraphDB instance is running
- `port`: port where the GraphDB instance is running
- `username`: username to login to the GraphDB instance (if any)
- `password`: password to login to the GraphDB instance (if any)
- `index`: name of the index (also called repository) stored in the GraphDB instance
- `prefixes`: definitions of namespaces with a new line after each namespace, e.g., PREFIX hp: <https://deepset.ai/harry_potter/>

<a id="graphdb.GraphDBKnowledgeGraph.create_index"></a>

#### GraphDBKnowledgeGraph.create\_index

```python
def create_index(config_path: Path, headers: Optional[Dict[str, str]] = None)
```

Create a new index (also called repository) stored in the GraphDB instance

**Arguments**:

- `config_path`: path to a .ttl file with configuration settings, details:
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
https://graphdb.ontotext.com/documentation/free/configuring-a-repository.html#configure-a-repository-programmatically

<a id="graphdb.GraphDBKnowledgeGraph.delete_index"></a>

#### GraphDBKnowledgeGraph.delete\_index

```python
def delete_index(headers: Optional[Dict[str, str]] = None)
```

Delete the index that GraphDBKnowledgeGraph is connected to. This method deletes all data stored in the index.

**Arguments**:

- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

<a id="graphdb.GraphDBKnowledgeGraph.import_from_ttl_file"></a>

#### GraphDBKnowledgeGraph.import\_from\_ttl\_file

```python
def import_from_ttl_file(index: str, path: Path, headers: Optional[Dict[str, str]] = None)
```

Load an existing knowledge graph represented in the form of triples of subject, predicate, and object from a .ttl file into an index of GraphDB

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance where the imported triples shall be stored
- `path`: path to a .ttl containing a knowledge graph
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

<a id="graphdb.GraphDBKnowledgeGraph.get_all_triples"></a>

#### GraphDBKnowledgeGraph.get\_all\_triples

```python
def get_all_triples(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Query the given index in the GraphDB instance for all its stored triples. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

**Returns**:

all triples stored in the index

<a id="graphdb.GraphDBKnowledgeGraph.get_all_subjects"></a>

#### GraphDBKnowledgeGraph.get\_all\_subjects

```python
def get_all_subjects(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Query the given index in the GraphDB instance for all its stored subjects. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

**Returns**:

all subjects stored in the index

<a id="graphdb.GraphDBKnowledgeGraph.get_all_predicates"></a>

#### GraphDBKnowledgeGraph.get\_all\_predicates

```python
def get_all_predicates(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Query the given index in the GraphDB instance for all its stored predicates. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

**Returns**:

all predicates stored in the index

<a id="graphdb.GraphDBKnowledgeGraph.get_all_objects"></a>

#### GraphDBKnowledgeGraph.get\_all\_objects

```python
def get_all_objects(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Query the given index in the GraphDB instance for all its stored objects. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

**Returns**:

all objects stored in the index

<a id="graphdb.GraphDBKnowledgeGraph.query"></a>

#### GraphDBKnowledgeGraph.query

```python
def query(sparql_query: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Execute a SPARQL query on the given index in the GraphDB instance

**Arguments**:

- `sparql_query`: SPARQL query that shall be executed
- `index`: name of the index (also called repository) in the GraphDB instance
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

**Returns**:

query result

<a id="deepsetcloud"></a>

# Module deepsetcloud

<a id="deepsetcloud.disable_and_log"></a>

#### disable\_and\_log

```python
def disable_and_log(func)
```

Decorator to disable write operation, shows warning and inputs instead.

<a id="deepsetcloud.DeepsetCloudDocumentStore"></a>

## DeepsetCloudDocumentStore

```python
class DeepsetCloudDocumentStore(KeywordDocumentStore)
```

<a id="deepsetcloud.DeepsetCloudDocumentStore.__init__"></a>

#### DeepsetCloudDocumentStore.\_\_init\_\_

```python
def __init__(api_key: str = None, workspace: str = "default", index: Optional[str] = None, duplicate_documents: str = "overwrite", api_endpoint: Optional[str] = None, similarity: str = "dot_product", return_embedding: bool = False, label_index: str = "default")
```

A DocumentStore facade enabling you to interact with the documents stored in deepset Cloud.

Thus you can run experiments like trying new nodes, pipelines, etc. without having to index your data again.

You can also use this DocumentStore to create new pipelines on deepset Cloud. To do that, take the following
steps:

- create a new DeepsetCloudDocumentStore without an index (e.g. `DeepsetCloudDocumentStore()`)
- create query and indexing pipelines using this DocumentStore
- call `Pipeline.save_to_deepset_cloud()` passing the pipelines and a `pipeline_config_name`
- call `Pipeline.deploy_on_deepset_cloud()` passing the `pipeline_config_name`

DeepsetCloudDocumentStore is not intended for use in production-like scenarios.
See [https://haystack.deepset.ai/components/v1.5.0/document-store](https://haystack.deepset.ai/components/v1.5.0/document-store)
for more information.

**Arguments**:

- `api_key`: Secret value of the API key.
If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
See docs on how to generate an API key for your workspace: https://docs.cloud.deepset.ai/docs/connect-deepset-cloud-to-your-application
- `workspace`: workspace name in deepset Cloud
- `index`: name of the index to access within the deepset Cloud workspace. This equals typically the name of
your pipeline. You can run Pipeline.list_pipelines_on_deepset_cloud() to see all available ones.
If you set index to `None`, this DocumentStore will always return empty results.
This is especially useful if you want to create a new Pipeline within deepset Cloud
(see Pipeline.save_to_deepset_cloud()` and `Pipeline.deploy_on_deepset_cloud()`).
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `api_endpoint`: The URL of the deepset Cloud API.
If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
If DEEPSET_CLOUD_API_ENDPOINT environment variable is not specified either, defaults to "https://api.cloud.deepset.ai/api/v1".
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default since it is
more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence Transformer model.
- `label_index`: index for the evaluation set interface
- `return_embedding`: To return document embedding.

<a id="deepsetcloud.DeepsetCloudDocumentStore.get_all_documents"></a>

#### DeepsetCloudDocumentStore.get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: Number of documents that are passed to bulk function at a time.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="deepsetcloud.DeepsetCloudDocumentStore.get_all_documents_generator"></a>

#### DeepsetCloudDocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="deepsetcloud.DeepsetCloudDocumentStore.query_by_embedding"></a>

#### DeepsetCloudDocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
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
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `headers`: Custom HTTP headers to pass to requests
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="deepsetcloud.DeepsetCloudDocumentStore.query"></a>

#### DeepsetCloudDocumentStore.query

```python
def query(query: Optional[str], filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, all_terms_must_match: bool = False, scale_score: bool = True) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query as defined by the BM25 algorithm.

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
- `custom_query`: Custom query to be executed.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to requests
- `all_terms_must_match`: Whether all terms of the query must match the document.
If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
Defaults to False.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="deepsetcloud.DeepsetCloudDocumentStore.write_documents"></a>

#### DeepsetCloudDocumentStore.write\_documents

```python
@disable_and_log
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Indexes documents for later queries.

**Arguments**:

- `documents`: a list of Python dictionaries or a list of Haystack Document objects.
For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
Optionally: Include meta data via {"text": "<the-actual-text>",
"meta":{"name": "<some-document-name>, "author": "somebody", ...}}
It can be used for filtering and is accessible in the responses of the Finder.
- `index`: Optional name of index where the documents shall be written to.
If None, the DocumentStore's default index (self.index) will be used.
- `batch_size`: Number of documents that are passed to bulk function at a time.
- `duplicate_documents`: Handle duplicates document based on parameter options.
Parameter options : ( 'skip','overwrite','fail')
skip: Ignore the duplicates documents
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

**Returns**:

None

<a id="deepsetcloud.DeepsetCloudDocumentStore.get_evaluation_sets"></a>

#### DeepsetCloudDocumentStore.get\_evaluation\_sets

```python
def get_evaluation_sets() -> List[dict]
```

Returns a list of uploaded evaluation sets to deepset cloud.

**Returns**:

list of evaluation sets as dicts
These contain ("name", "evaluation_set_id", "created_at", "matched_labels", "total_labels") as fields.

<a id="deepsetcloud.DeepsetCloudDocumentStore.get_all_labels"></a>

#### DeepsetCloudDocumentStore.get\_all\_labels

```python
def get_all_labels(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None) -> List[Label]
```

Returns a list of labels for the given index name.

**Arguments**:

- `index`: Optional name of evaluation set for which labels should be searched.
If None, the DocumentStore's default label_index (self.label_index) will be used.
- `headers`: Not supported.

**Returns**:

list of Labels.

<a id="deepsetcloud.DeepsetCloudDocumentStore.get_label_count"></a>

#### DeepsetCloudDocumentStore.get\_label\_count

```python
def get_label_count(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Counts the number of labels for the given index and returns the value.

**Arguments**:

- `index`: Optional evaluation set name for which the labels should be counted.
If None, the DocumentStore's default label_index (self.label_index) will be used.
- `headers`: Not supported.

**Returns**:

number of labels for the given index

<a id="pinecone"></a>

# Module pinecone

<a id="pinecone.PineconeDocumentStore"></a>

## PineconeDocumentStore

```python
class PineconeDocumentStore(SQLDocumentStore)
```

Document store for very large scale embedding based dense retrievers like the DPR. This is a hosted document store,
this means that your vectors will not be stored locally but in the cloud. This means that the similarity
search will be run on the cloud as well.

It implements the Pinecone vector database ([https://www.pinecone.io](https://www.pinecone.io))
to perform similarity search on vectors. In order to use this document store, you need an API key that you can
obtain by creating an account on the [Pinecone website](https://www.pinecone.io).

The document text is stored using the SQLDocumentStore, while
the vector embeddings and metadata (for filtering) are indexed in a Pinecone Index.

<a id="pinecone.PineconeDocumentStore.__init__"></a>

#### PineconeDocumentStore.\_\_init\_\_

```python
def __init__(api_key: str, environment: str = "us-west1-gcp", sql_url: str = "sqlite:///pinecone_document_store.db", pinecone_index: Optional[pinecone.Index] = None, embedding_dim: int = 768, return_embedding: bool = False, index: str = "document", similarity: str = "cosine", replicas: int = 1, shards: int = 1, embedding_field: str = "embedding", progress_bar: bool = True, duplicate_documents: str = "overwrite", recreate_index: bool = False)
```

**Arguments**:

- `api_key`: Pinecone vector database API key ([https://app.pinecone.io](https://app.pinecone.io)).
- `environment`: Pinecone cloud environment uses `"us-west1-gcp"` by default. Other GCP and AWS regions are
supported, contact Pinecone [here](https://www.pinecone.io/contact/) if required.
- `sql_url`: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
deployment, Postgres is recommended.
- `pinecone_index`: pinecone-client Index object, an index will be initialized or loaded if not specified.
- `embedding_dim`: The embedding vector size.
- `return_embedding`: Whether to return document embeddings.
- `index`: Name of index in document store to use.
- `similarity`: The similarity function used to compare document vectors. `"cosine"` is the default
and is recommended if you are using a Sentence-Transformer model. `"dot_product"` is more performant
with DPR embeddings.
In both cases, the returned values in Document.score are normalized to be in range [0,1]:
    - For `"dot_product"`: `expit(np.asarray(raw_score / 100))`
    - For `"cosine"`: `(raw_score + 1) / 2`
- `replicas`: The number of replicas. Replicas duplicate the index. They provide higher availability and
throughput.
- `shards`: The number of shards to be used in the index. We recommend to use 1 shard per 1GB of data.
- `embedding_field`: Name of field containing an embedding vector.
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicate documents based on parameter options.\
Parameter options:
    - `"skip"`: Ignore the duplicate documents.
    - `"overwrite"`: Update any existing documents with the same ID when adding documents.
    - `"fail"`: An error is raised if the document ID of the document being added already exists.
- `recreate_index`: If set to True, an existing Pinecone index will be deleted and a new one will be
created using the config you are using for initialization. Be aware that all data in the old index will be
lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
be recreated.

<a id="pinecone.PineconeDocumentStore.write_documents"></a>

#### PineconeDocumentStore.write\_documents

```python
def write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 32, duplicate_documents: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Add new documents to the DocumentStore.

**Arguments**:

- `documents`: List of `Dicts` or list of `Documents`. If they already contain embeddings, we'll index them
right away in Pinecone. If not, you can later call `update_embeddings()` to create & index them.
- `index`: Index name for storing the docs and metadata.
- `batch_size`: Number of documents to process at a time. When working with large number of documents,
batching can help to reduce the memory footprint.
- `duplicate_documents`: handle duplicate documents based on parameter options.
Parameter options:
    - `"skip"`: Ignore the duplicate documents.
    - `"overwrite"`: Update any existing documents with the same ID when adding documents.
    - `"fail"`: An error is raised if the document ID of the document being added already exists.
- `headers`: PineconeDocumentStore does not support headers.

**Raises**:

- `DuplicateDocumentError`: Exception trigger on duplicate document.

<a id="pinecone.PineconeDocumentStore.update_embeddings"></a>

#### PineconeDocumentStore.update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, update_existing_embeddings: bool = True, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, batch_size: int = 32)
```

Updates the embeddings in the document store using the encoding model specified in the retriever.

This can be useful if you want to add or change the embeddings for your documents (e.g. after changing the
retriever config).

**Arguments**:

- `retriever`: Retriever to use to get embeddings for text.
- `index`: Index name for which embeddings are to be updated. If set to `None`, the default `self.index` is
used.
- `update_existing_embeddings`: Whether to update existing embeddings of the documents. If set to `False`,
only documents without embeddings are processed. This mode can be used for incremental updating of
embeddings, wherein, only newly indexed documents get processed.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
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
    ```
- `batch_size`: Number of documents to process at a time. When working with large number of documents,
batching can help reduce memory footprint.

<a id="pinecone.PineconeDocumentStore.get_all_documents_generator"></a>

#### PineconeDocumentStore.get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 32, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get all documents from the document store. Under-the-hood, documents are fetched in batches from the

document store and yielded as individual documents. This method can be used to iteratively process
a large number of documents without having to load all documents in memory.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
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
    ```
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: PineconeDocumentStore does not support headers.

<a id="pinecone.PineconeDocumentStore.get_embedding_count"></a>

#### PineconeDocumentStore.get\_embedding\_count

```python
def get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None) -> int
```

Return the count of embeddings in the document store.

<a id="pinecone.PineconeDocumentStore.update_document_meta"></a>

#### PineconeDocumentStore.update\_document\_meta

```python
def update_document_meta(id: str, meta: Dict[str, str], index: str = None)
```

Update the metadata dictionary of a document by specifying its string id

<a id="pinecone.PineconeDocumentStore.delete_documents"></a>

#### PineconeDocumentStore.delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents from the document store.

**Arguments**:

- `index`: Index name to delete the documents from. If `None`, the DocumentStore's default index
(`self.index`) will be used.
- `ids`: Optional list of IDs to narrow down the documents to be deleted.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
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
    ```
- `headers`: PineconeDocumentStore does not support headers.

<a id="pinecone.PineconeDocumentStore.delete_index"></a>

#### PineconeDocumentStore.delete\_index

```python
def delete_index(index: str)
```

Delete an existing index. The index including all data will be removed.

**Arguments**:

- `index`: The name of the index to delete.

**Returns**:

None

<a id="pinecone.PineconeDocumentStore.query_by_embedding"></a>

#### PineconeDocumentStore.query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None, scale_score: bool = True) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR).
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
- `top_k`: How many documents to return.
- `index`: The name of the index from which to retrieve documents.
- `return_embedding`: Whether to return document embedding.
- `headers`: PineconeDocumentStore does not support headers.
- `scale_score`: Whether to scale the similarity score to the unit interval (range of [0,1]).
If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

<a id="pinecone.PineconeDocumentStore.load"></a>

#### PineconeDocumentStore.load

```python
@classmethod
def load(cls)
```

Default class method used for loading indexes. Not applicable to the PineconeDocumentStore.

<a id="utils"></a>

# Module utils

<a id="utils.eval_data_from_json"></a>

#### eval\_data\_from\_json

```python
def eval_data_from_json(filename: str, max_docs: Union[int, bool] = None, preprocessor: PreProcessor = None, open_domain: bool = False) -> Tuple[List[Document], List[Label]]
```

Read Documents + Labels from a SQuAD-style file.

Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

**Arguments**:

- `filename`: Path to file in SQuAD format
- `max_docs`: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
- `open_domain`: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.

<a id="utils.eval_data_from_jsonl"></a>

#### eval\_data\_from\_jsonl

```python
def eval_data_from_jsonl(filename: str, batch_size: Optional[int] = None, max_docs: Union[int, bool] = None, preprocessor: PreProcessor = None, open_domain: bool = False) -> Generator[Tuple[List[Document], List[Label]], None, None]
```

Read Documents + Labels from a SQuAD-style file in jsonl format, i.e. one document per line.

Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

This is a generator which will yield one tuple per iteration containing a list
of batch_size documents and a list with the documents' labels.
If batch_size is set to None, this method will yield all documents and labels.

**Arguments**:

- `filename`: Path to file in SQuAD format
- `max_docs`: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
- `open_domain`: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.

<a id="utils.squad_json_to_jsonl"></a>

#### squad\_json\_to\_jsonl

```python
def squad_json_to_jsonl(squad_file: str, output_file: str)
```

Converts a SQuAD-json-file into jsonl format with one document per line.

**Arguments**:

- `squad_file`: SQuAD-file in json format.
- `output_file`: Name of output file (SQuAD in jsonl format)

<a id="utils.convert_date_to_rfc3339"></a>

#### convert\_date\_to\_rfc3339

```python
def convert_date_to_rfc3339(date: str) -> str
```

Converts a date to RFC3339 format, as Weaviate requires dates to be in RFC3339 format including the time and
timezone.

If the provided date string does not contain a time and/or timezone, we use 00:00 as default time
and UTC as default time zone.

This method cannot be part of WeaviateDocumentStore, as this would result in a circular import between weaviate.py
and filter_utils.py.
