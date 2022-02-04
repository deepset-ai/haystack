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

#### write\_documents

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

#### get\_all\_documents

```python
@abstractmethod
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: Number of documents that are passed to bulk function at a time.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="base.BaseDocumentStore.get_all_documents_generator"></a>

#### get\_all\_documents\_generator

```python
@abstractmethod
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
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
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="base.BaseDocumentStore.get_all_labels_aggregated"></a>

#### get\_all\_labels\_aggregated

```python
def get_all_labels_aggregated(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, open_domain: bool = True, drop_negative_labels: bool = False, drop_no_answers: bool = False, aggregate_by_meta: Optional[Union[str, list]] = None, headers: Optional[Dict[str, str]] = None) -> List[MultiLabel]
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
- `filters`: Optional filters to narrow down the labels to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `open_domain`: When True, labels are aggregated purely based on the question text alone.
When False, labels are aggregated in a closed domain fashion based on the question text
and also the id of the document that the label is tied to. In this setting, this function
might return multiple MultiLabel objects with the same question string.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
- `aggregate_by_meta`: The names of the Label meta fields by which to aggregate. For example: ["product_id"]
TODO drop params

<a id="base.BaseDocumentStore.normalize_embedding"></a>

#### normalize\_embedding

```python
@staticmethod
@njit
def normalize_embedding(emb: np.ndarray) -> None
```

Performs L2 normalization of embeddings vector inplace. Input can be a single vector (1D array) or a matrix
(2D array).

<a id="base.BaseDocumentStore.add_eval_data"></a>

#### add\_eval\_data

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

<a id="base.BaseDocumentStore.run"></a>

#### run

```python
def run(documents: List[dict], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None, id_hash_keys: Optional[List[str]] = None)
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

<a id="base.KeywordDocumentStore"></a>

## KeywordDocumentStore

```python
class KeywordDocumentStore(BaseDocumentStore)
```

Base class for implementing Document Stores that support keyword searches.

<a id="base.KeywordDocumentStore.query"></a>

#### query

```python
@abstractmethod
def query(query: Optional[str], filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query as defined by keyword matching algorithms like BM25.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `custom_query`: Custom query to be executed.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

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

<a id="elasticsearch.ElasticsearchDocumentStore.get_document_by_id"></a>

#### get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a id="elasticsearch.ElasticsearchDocumentStore.get_documents_by_id"></a>

#### get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings. Be aware that passing a large number of ids might lead
to performance issues. Note that Elasticsearch limits the number of results to 10,000 documents by default.

<a id="elasticsearch.ElasticsearchDocumentStore.get_metadata_values_by_key"></a>

#### get\_metadata\_values\_by\_key

```python
def get_metadata_values_by_key(key: str, query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> List[dict]
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
- `index`: Elasticsearch index where the meta values should be searched. If not supplied,
self.index will be used.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.write_documents"></a>

#### write\_documents

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

#### write\_labels

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

#### update\_document\_meta

```python
def update_document_meta(id: str, meta: Dict[str, str], headers: Optional[Dict[str, str]] = None, index: str = None)
```

Update the metadata dictionary of a document by specifying its string id

<a id="elasticsearch.ElasticsearchDocumentStore.get_document_count"></a>

#### get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, Any]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="elasticsearch.ElasticsearchDocumentStore.get_label_count"></a>

#### get\_label\_count

```python
def get_label_count(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of labels in the document store

<a id="elasticsearch.ElasticsearchDocumentStore.get_embedding_count"></a>

#### get\_embedding\_count

```python
def get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the count of embeddings in the document store.

<a id="elasticsearch.ElasticsearchDocumentStore.get_all_documents"></a>

#### get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
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
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.get_all_documents_generator"></a>

#### get\_all\_documents\_generator

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
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.get_all_labels"></a>

#### get\_all\_labels

```python
def get_all_labels(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, batch_size: int = 10_000) -> List[Label]
```

Return all labels in the document store

<a id="elasticsearch.ElasticsearchDocumentStore.query"></a>

#### query

```python
def query(query: Optional[str], filters: Optional[Dict[str, Any]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
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

<a id="elasticsearch.ElasticsearchDocumentStore.query_by_embedding"></a>

#### query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
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
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="elasticsearch.ElasticsearchDocumentStore.describe_documents"></a>

#### describe\_documents

```python
def describe_documents(index=None)
```

Return a summary of the documents in the document store

<a id="elasticsearch.ElasticsearchDocumentStore.update_embeddings"></a>

#### update\_embeddings

```python
def update_embeddings(retriever, index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None)
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
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_all_documents"></a>

#### delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
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
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_documents"></a>

#### delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
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

If filters are provided along with a list of IDs, this method deletes the
intersection of the two query results (documents that match the filters and
have their ID in the list).
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_labels"></a>

#### delete\_labels

```python
def delete_labels(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None)
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
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

**Returns**:

None

<a id="elasticsearch.ElasticsearchDocumentStore.delete_index"></a>

#### delete\_index

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

<a id="elasticsearch.OpenSearchDocumentStore.query_by_embedding"></a>

#### query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
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
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

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

<a id="memory.InMemoryDocumentStore.write_documents"></a>

#### write\_documents

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

#### write\_labels

```python
def write_labels(labels: Union[List[dict], List[Label]], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

Write annotation labels into document store.

<a id="memory.InMemoryDocumentStore.get_document_by_id"></a>

#### get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string.

<a id="memory.InMemoryDocumentStore.get_documents_by_id"></a>

#### get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings.

<a id="memory.InMemoryDocumentStore.query_by_embedding"></a>

#### query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding

<a id="memory.InMemoryDocumentStore.update_embeddings"></a>

#### update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000)
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

<a id="memory.InMemoryDocumentStore.get_document_count"></a>

#### get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="memory.InMemoryDocumentStore.get_embedding_count"></a>

#### get\_embedding\_count

```python
def get_embedding_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int
```

Return the count of embeddings in the document store.

<a id="memory.InMemoryDocumentStore.get_label_count"></a>

#### get\_label\_count

```python
def get_label_count(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of labels in the document store.

<a id="memory.InMemoryDocumentStore.get_all_documents"></a>

#### get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get all documents from the document store as a list.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.

<a id="memory.InMemoryDocumentStore.get_all_documents_generator"></a>

#### get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
```

Get all documents from the document store. The methods returns a Python Generator that yields individual

documents.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.

<a id="memory.InMemoryDocumentStore.get_all_labels"></a>

#### get\_all\_labels

```python
def get_all_labels(index: str = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None) -> List[Label]
```

Return all labels in the document store.

<a id="memory.InMemoryDocumentStore.delete_all_documents"></a>

#### delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a id="memory.InMemoryDocumentStore.delete_documents"></a>

#### delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

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

<a id="memory.InMemoryDocumentStore.delete_labels"></a>

#### delete\_labels

```python
def delete_labels(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete labels in an index. All labels are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the labels from. If None, the
DocumentStore's default label index (self.label_index) will be used.
- `ids`: Optional list of IDs to narrow down the labels to be deleted.
- `filters`: Optional filters to narrow down the labels to be deleted.
Example filters: {"id": ["9a196e41-f7b5-45b4-bd19-5feb7501c159", "9a196e41-f7b5-45b4-bd19-5feb7501c159"]} or {"query": ["question2"]}

**Returns**:

None

<a id="sql"></a>

# Module sql

<a id="sql.SQLDocumentStore"></a>

## SQLDocumentStore

```python
class SQLDocumentStore(BaseDocumentStore)
```

<a id="sql.SQLDocumentStore.get_document_by_id"></a>

#### get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a id="sql.SQLDocumentStore.get_documents_by_id"></a>

#### get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings

<a id="sql.SQLDocumentStore.get_documents_by_vector_ids"></a>

#### get\_documents\_by\_vector\_ids

```python
def get_documents_by_vector_ids(vector_ids: List[str], index: Optional[str] = None, batch_size: int = 10_000)
```

Fetch documents by specifying a list of text vector id strings

<a id="sql.SQLDocumentStore.get_all_documents_generator"></a>

#### get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
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

#### get\_all\_labels

```python
def get_all_labels(index=None, filters: Optional[dict] = None, headers: Optional[Dict[str, str]] = None)
```

Return all labels in the document store

<a id="sql.SQLDocumentStore.write_documents"></a>

#### write\_documents

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
overwrite: Update any existing documents with the same ID when adding documents.
fail: an error is raised if the document ID of the document being added already
exists.

**Returns**:

None

<a id="sql.SQLDocumentStore.write_labels"></a>

#### write\_labels

```python
def write_labels(labels, index=None, headers: Optional[Dict[str, str]] = None)
```

Write annotation labels into document store.

<a id="sql.SQLDocumentStore.update_vector_ids"></a>

#### update\_vector\_ids

```python
def update_vector_ids(vector_id_map: Dict[str, str], index: Optional[str] = None, batch_size: int = 10_000)
```

Update vector_ids for given document_ids.

**Arguments**:

- `vector_id_map`: dict containing mapping of document_id -> vector_id.
- `index`: filter documents by the optional index attribute for documents in database.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="sql.SQLDocumentStore.reset_vector_ids"></a>

#### reset\_vector\_ids

```python
def reset_vector_ids(index: Optional[str] = None)
```

Set vector IDs for all documents as None

<a id="sql.SQLDocumentStore.update_document_meta"></a>

#### update\_document\_meta

```python
def update_document_meta(id: str, meta: Dict[str, str], index: str = None)
```

Update the metadata dictionary of a document by specifying its string id

<a id="sql.SQLDocumentStore.get_document_count"></a>

#### get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="sql.SQLDocumentStore.get_label_count"></a>

#### get\_label\_count

```python
def get_label_count(index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of labels in the document store

<a id="sql.SQLDocumentStore.delete_all_documents"></a>

#### delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a id="sql.SQLDocumentStore.delete_documents"></a>

#### delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
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

<a id="sql.SQLDocumentStore.delete_labels"></a>

#### delete\_labels

```python
def delete_labels(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
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

<a id="faiss.FAISSDocumentStore.write_documents"></a>

#### write\_documents

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

#### update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, update_existing_embeddings: bool = True, filters: Optional[Dict[str, List[str]]] = None, batch_size: int = 10_000)
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

#### get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
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

#### get\_embedding\_count

```python
def get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int
```

Return the count of embeddings in the document store.

<a id="faiss.FAISSDocumentStore.train_index"></a>

#### train\_index

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

#### delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete all documents from the document store.

<a id="faiss.FAISSDocumentStore.delete_documents"></a>

#### delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
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

<a id="faiss.FAISSDocumentStore.query_by_embedding"></a>

#### query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: Index name to query the document from.
- `return_embedding`: To return document embedding. Unlike other document stores, FAISS will return normalized embeddings

<a id="faiss.FAISSDocumentStore.save"></a>

#### save

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

#### load

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

<a id="milvus"></a>

# Module milvus

<a id="milvus.MilvusDocumentStore"></a>

## MilvusDocumentStore

```python
class MilvusDocumentStore(SQLDocumentStore)
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
2. Init a MilvusDocumentStore in Haystack

<a id="milvus.MilvusDocumentStore.write_documents"></a>

#### write\_documents

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

<a id="milvus.MilvusDocumentStore.update_embeddings"></a>

#### update\_embeddings

```python
def update_embeddings(retriever: "BaseRetriever", index: Optional[str] = None, batch_size: int = 10_000, update_existing_embeddings: bool = True, filters: Optional[Dict[str, List[str]]] = None)
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

<a id="milvus.MilvusDocumentStore.query_by_embedding"></a>

#### query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[dict] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: (SQL) index name for storing the docs and metadata
- `return_embedding`: To return document embedding

**Returns**:

list of Documents that are the most similar to `query_emb`

<a id="milvus.MilvusDocumentStore.delete_all_documents"></a>

#### delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete all documents (from SQL AND Milvus).

**Arguments**:

- `index`: (SQL) index name for storing the docs and metadata
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}

**Returns**:

None

<a id="milvus.MilvusDocumentStore.delete_documents"></a>

#### delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
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

<a id="milvus.MilvusDocumentStore.get_all_documents_generator"></a>

#### get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
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

<a id="milvus.MilvusDocumentStore.get_all_documents"></a>

#### get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store (optionally using filter criteria).

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="milvus.MilvusDocumentStore.get_document_by_id"></a>

#### get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

**Arguments**:

- `id`: ID of the document
- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.

<a id="milvus.MilvusDocumentStore.get_documents_by_id"></a>

#### get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch multiple documents by specifying their IDs (strings)

**Arguments**:

- `ids`: List of IDs of the documents
- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `batch_size`: is currently not used

<a id="milvus.MilvusDocumentStore.get_all_vectors"></a>

#### get\_all\_vectors

```python
def get_all_vectors(index: Optional[str] = None) -> List[np.ndarray]
```

Helper function to dump all vectors stored in Milvus server.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.

**Returns**:

List[np.array]: List of vectors.

<a id="milvus.MilvusDocumentStore.get_embedding_count"></a>

#### get\_embedding\_count

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
(See https://www.semi.technology/developers/weaviate/current/index.html#what-is-weaviate)

Some of the key differences in contrast to FAISS & Milvus:
1. Stores everything in one place: documents, meta data and vectors - so less network overhead when scaling this up
2. Allows combination of vector search and scalar filtering, i.e. you can filter for a certain tag and do dense retrieval on that subset
3. Has less variety of ANN algorithms, as of now only HNSW.
4. Requires document ids to be in uuid-format. If wrongly formatted ids are provided at indexing time they will be replaced with uuids automatically.
5. Only support cosine similarity.

Weaviate python client is used to connect to the server, more details are here
https://weaviate-python-client.readthedocs.io/en/docs/weaviate.html

Usage:
1. Start a Weaviate server (see https://www.semi.technology/developers/weaviate/current/getting-started/installation.html)
2. Init a WeaviateDocumentStore in Haystack

Limitations:
The current implementation is not supporting the storage of labels, so you cannot run any evaluation workflows.

<a id="weaviate.WeaviateDocumentStore.get_document_by_id"></a>

#### get\_document\_by\_id

```python
def get_document_by_id(id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]
```

Fetch a document by specifying its uuid string

<a id="weaviate.WeaviateDocumentStore.get_documents_by_id"></a>

#### get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Fetch documents by specifying a list of uuid strings.

<a id="weaviate.WeaviateDocumentStore.write_documents"></a>

#### write\_documents

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

#### update\_document\_meta

```python
def update_document_meta(id: str, meta: Dict[str, str], index: str = None)
```

Update the metadata dictionary of a document by specifying its string id.

<a id="weaviate.WeaviateDocumentStore.get_embedding_count"></a>

#### get\_embedding\_count

```python
def get_embedding_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int
```

Return the number of embeddings in the document store, which is the same as the number of documents since every document has a default embedding

<a id="weaviate.WeaviateDocumentStore.get_document_count"></a>

#### get\_document\_count

```python
def get_document_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False, headers: Optional[Dict[str, str]] = None) -> int
```

Return the number of documents in the document store.

<a id="weaviate.WeaviateDocumentStore.get_all_documents"></a>

#### get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a id="weaviate.WeaviateDocumentStore.get_all_documents_generator"></a>

#### get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
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

<a id="weaviate.WeaviateDocumentStore.query"></a>

#### query

```python
def query(query: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query as defined by Weaviate semantic search.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `custom_query`: Custom query that will executed using query.raw method, for more details refer
https://www.semi.technology/developers/weaviate/current/graphql-references/filters.html
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a id="weaviate.WeaviateDocumentStore.query_by_embedding"></a>

#### query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[dict] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: index name for storing the docs and metadata
- `return_embedding`: To return document embedding

<a id="weaviate.WeaviateDocumentStore.update_embeddings"></a>

#### update\_embeddings

```python
def update_embeddings(retriever, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.

This can be useful if want to change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to update the embeddings.
- `index`: Index name to update
- `update_existing_embeddings`: Weaviate mandates an embedding while creating the document itself.
This option must be always true for weaviate and it will update the embeddings for all the documents.
- `filters`: Optional filters to narrow down the documents for which embeddings are to be updated.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

**Returns**:

None

<a id="weaviate.WeaviateDocumentStore.delete_all_documents"></a>

#### delete\_all\_documents

```python
def delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a id="weaviate.WeaviateDocumentStore.delete_documents"></a>

#### delete\_documents

```python
def delete_documents(index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None, headers: Optional[Dict[str, str]] = None)
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

<a id="graphdb"></a>

# Module graphdb

<a id="graphdb.GraphDBKnowledgeGraph"></a>

## GraphDBKnowledgeGraph

```python
class GraphDBKnowledgeGraph(BaseKnowledgeGraph)
```

Knowledge graph store that runs on a GraphDB instance.

<a id="graphdb.GraphDBKnowledgeGraph.create_index"></a>

#### create\_index

```python
def create_index(config_path: Path, headers: Optional[Dict[str, str]] = None)
```

Create a new index (also called repository) stored in the GraphDB instance

**Arguments**:

- `config_path`: path to a .ttl file with configuration settings, details:
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
https://graphdb.ontotext.com/documentation/free/configuring-a-repository.html#configure-a-repository-programmatically

<a id="graphdb.GraphDBKnowledgeGraph.delete_index"></a>

#### delete\_index

```python
def delete_index(headers: Optional[Dict[str, str]] = None)
```

Delete the index that GraphDBKnowledgeGraph is connected to. This method deletes all data stored in the index.

**Arguments**:

- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

<a id="graphdb.GraphDBKnowledgeGraph.import_from_ttl_file"></a>

#### import\_from\_ttl\_file

```python
def import_from_ttl_file(index: str, path: Path, headers: Optional[Dict[str, str]] = None)
```

Load an existing knowledge graph represented in the form of triples of subject, predicate, and object from a .ttl file into an index of GraphDB

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance where the imported triples shall be stored
- `path`: path to a .ttl containing a knowledge graph
- `headers`: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})

<a id="graphdb.GraphDBKnowledgeGraph.get_all_triples"></a>

#### get\_all\_triples

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

#### get\_all\_subjects

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

#### get\_all\_predicates

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

#### get\_all\_objects

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

#### query

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

<a id="deepsetcloud.DeepsetCloudDocumentStore"></a>

## DeepsetCloudDocumentStore

```python
class DeepsetCloudDocumentStore(KeywordDocumentStore)
```

<a id="deepsetcloud.DeepsetCloudDocumentStore.get_all_documents"></a>

#### get\_all\_documents

```python
def get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: Number of documents that are passed to bulk function at a time.
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="deepsetcloud.DeepsetCloudDocumentStore.get_all_documents_generator"></a>

#### get\_all\_documents\_generator

```python
def get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> Generator[Document, None, None]
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
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="deepsetcloud.DeepsetCloudDocumentStore.query_by_embedding"></a>

#### query\_by\_embedding

```python
def query_by_embedding(query_emb: np.ndarray, filters: Optional[Optional[Dict[str, List[str]]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding
- `headers`: Custom HTTP headers to pass to requests

<a id="deepsetcloud.DeepsetCloudDocumentStore.query"></a>

#### query

```python
def query(query: Optional[str], filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query as defined by the BM25 algorithm.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `custom_query`: Custom query to be executed.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to requests

<a id="deepsetcloud.DeepsetCloudDocumentStore.write_documents"></a>

#### write\_documents

```python
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

