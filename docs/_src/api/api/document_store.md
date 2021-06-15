<a name="base"></a>
# Module base

<a name="base.BaseDocumentStore"></a>
## BaseDocumentStore Objects

```python
class BaseDocumentStore(BaseComponent)
```

Base class for implementing Document Stores.

<a name="base.BaseDocumentStore.write_documents"></a>
#### write\_documents

```python
 | @abstractmethod
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None)
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

**Returns**:

None

<a name="base.BaseDocumentStore.get_all_documents"></a>
#### get\_all\_documents

```python
 | @abstractmethod
 | get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.

<a name="base.BaseDocumentStore.get_all_labels_aggregated"></a>
#### get\_all\_labels\_aggregated

```python
 | get_all_labels_aggregated(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, open_domain: bool = True, aggregate_by_meta: Optional[Union[str, list]] = None) -> List[MultiLabel]
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
- `aggregate_by_meta`: The names of the Label meta fields by which to aggregate. For example: ["product_id"]

<a name="base.BaseDocumentStore.add_eval_data"></a>
#### add\_eval\_data

```python
 | add_eval_data(filename: str, doc_index: str = "eval_document", label_index: str = "label", batch_size: Optional[int] = None, preprocessor: Optional[PreProcessor] = None, max_docs: Union[int, bool] = None, open_domain: bool = False)
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

<a name="elasticsearch"></a>
# Module elasticsearch

<a name="elasticsearch.ElasticsearchDocumentStore"></a>
## ElasticsearchDocumentStore Objects

```python
class ElasticsearchDocumentStore(BaseDocumentStore)
```

<a name="elasticsearch.ElasticsearchDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(host: Union[str, List[str]] = "localhost", port: Union[int, List[int]] = 9200, username: str = "", password: str = "", api_key_id: Optional[str] = None, api_key: Optional[str] = None, aws4auth=None, index: str = "document", label_index: str = "label", search_fields: Union[str, list] = "text", text_field: str = "text", name_field: str = "name", embedding_field: str = "embedding", embedding_dim: int = 768, custom_mapping: Optional[dict] = None, excluded_meta_data: Optional[list] = None, faq_question_field: Optional[str] = None, analyzer: str = "standard", scheme: str = "http", ca_certs: Optional[str] = None, verify_certs: bool = True, create_index: bool = True, refresh_type: str = "wait_for", similarity="dot_product", timeout=30, return_embedding: bool = False, duplicate_documents: str = 'overwrite')
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
- `search_fields`: Name of fields used by ElasticsearchRetriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
- `text_field`: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
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
- `create_index`: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case
- `refresh_type`: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
                     If set to 'wait_for', continue only after changes are visible (slow, but safe).
                     If set to 'false', continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion).
                     More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default sine it is
                   more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
- `timeout`: Number of seconds after which an ElasticSearch request times out.
- `return_embedding`: To return document embedding
- `duplicate_documents`: Handle duplicates document based on parameter options.
                            Parameter options : ( 'skip','overwrite','fail')
                            skip: Ignore the duplicates documents
                            overwrite: Update any existing documents with the same ID when adding documents.
                            fail: an error is raised if the document ID of the document being added already
                            exists.

<a name="elasticsearch.ElasticsearchDocumentStore.get_document_by_id"></a>
#### get\_document\_by\_id

```python
 | get_document_by_id(id: str, index: Optional[str] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a name="elasticsearch.ElasticsearchDocumentStore.get_documents_by_id"></a>
#### get\_documents\_by\_id

```python
 | get_documents_by_id(ids: List[str], index: Optional[str] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings

<a name="elasticsearch.ElasticsearchDocumentStore.get_metadata_values_by_key"></a>
#### get\_metadata\_values\_by\_key

```python
 | get_metadata_values_by_key(key: str, query: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> List[dict]
```

Get values associated with a metadata key. The output is in the format:
    [{"value": "my-value-1", "count": 23}, {"value": "my-value-2", "count": 12}, ... ]

**Arguments**:

- `key`: the meta key name to get the values for.
- `query`: narrow down the scope to documents matching the query string.
- `filters`: narrow down the scope to documents that match the given filters.
- `index`: Elasticsearch index where the meta values should be searched. If not supplied,
              self.index will be used.

<a name="elasticsearch.ElasticsearchDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None)
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
                  For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                  Optionally: Include meta data via {"text": "<the-actual-text>",
                  "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                  It can be used for filtering and is accessible in the responses of the Finder.
                  Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
                  should be changed to what you have set for self.text_field and self.name_field.
- `index`: Elasticsearch index where the documents should be indexed. If not supplied, self.index will be used.
- `batch_size`: Number of documents that are passed to Elasticsearch's bulk function at a time.
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

<a name="elasticsearch.ElasticsearchDocumentStore.write_labels"></a>
#### write\_labels

```python
 | write_labels(labels: Union[List[Label], List[dict]], index: Optional[str] = None, batch_size: int = 10_000)
```

Write annotation labels into document store.

**Arguments**:

- `labels`: A list of Python dictionaries or a list of Haystack Label objects.
- `batch_size`: Number of labels that are passed to Elasticsearch's bulk function at a time.

<a name="elasticsearch.ElasticsearchDocumentStore.update_document_meta"></a>
#### update\_document\_meta

```python
 | update_document_meta(id: str, meta: Dict[str, str])
```

Update the metadata dictionary of a document by specifying its string id

<a name="elasticsearch.ElasticsearchDocumentStore.get_document_count"></a>
#### get\_document\_count

```python
 | get_document_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None, only_documents_without_embedding: bool = False) -> int
```

Return the number of documents in the document store.

<a name="elasticsearch.ElasticsearchDocumentStore.get_label_count"></a>
#### get\_label\_count

```python
 | get_label_count(index: Optional[str] = None) -> int
```

Return the number of labels in the document store

<a name="elasticsearch.ElasticsearchDocumentStore.get_embedding_count"></a>
#### get\_embedding\_count

```python
 | get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int
```

Return the count of embeddings in the document store.

<a name="elasticsearch.ElasticsearchDocumentStore.get_all_documents"></a>
#### get\_all\_documents

```python
 | get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a name="elasticsearch.ElasticsearchDocumentStore.get_all_documents_generator"></a>
#### get\_all\_documents\_generator

```python
 | get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> Generator[Document, None, None]
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

<a name="elasticsearch.ElasticsearchDocumentStore.get_all_labels"></a>
#### get\_all\_labels

```python
 | get_all_labels(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, batch_size: int = 10_000) -> List[Label]
```

Return all labels in the document store

<a name="elasticsearch.ElasticsearchDocumentStore.query"></a>
#### query

```python
 | query(query: Optional[str], filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents
that are most relevant to the query as defined by the BM25 algorithm.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a name="elasticsearch.ElasticsearchDocumentStore.query_by_embedding"></a>
#### query\_by\_embedding

```python
 | query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding

**Returns**:



<a name="elasticsearch.ElasticsearchDocumentStore.describe_documents"></a>
#### describe\_documents

```python
 | describe_documents(index=None)
```

Return a summary of the documents in the document store

<a name="elasticsearch.ElasticsearchDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000)
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
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

**Returns**:

None

<a name="elasticsearch.ElasticsearchDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a name="elasticsearch.ElasticsearchDocumentStore.delete_documents"></a>
#### delete\_documents

```python
 | delete_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a name="elasticsearch.OpenDistroElasticsearchDocumentStore"></a>
## OpenDistroElasticsearchDocumentStore Objects

```python
class OpenDistroElasticsearchDocumentStore(ElasticsearchDocumentStore)
```

Document Store using the Open Distro for Elasticsearch. It is compatible with the AWS Elasticsearch Service.

In addition to native Elasticsearch query & filtering, it provides efficient vector similarity search using
the KNN plugin that can scale to a large number of documents.

<a name="memory"></a>
# Module memory

<a name="memory.InMemoryDocumentStore"></a>
## InMemoryDocumentStore Objects

```python
class InMemoryDocumentStore(BaseDocumentStore)
```

In-memory document store

<a name="memory.InMemoryDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(index: str = "document", label_index: str = "label", embedding_field: Optional[str] = "embedding", embedding_dim: int = 768, return_embedding: bool = False, similarity: str = "dot_product", progress_bar: bool = True, duplicate_documents: str = 'overwrite')
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

<a name="memory.InMemoryDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, duplicate_documents: Optional[str] = None)
```

Indexes documents for later queries.


**Arguments**:

- `documents`: a list of Python dictionaries or a list of Haystack Document objects.
                   For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                   Optionally: Include meta data via {"text": "<the-actual-text>",
                   "meta": {"name": "<some-document-name>, "author": "somebody", ...}}
                   It can be used for filtering and is accessible in the responses of the Finder.
- `index`: write documents to a custom namespace. For instance, documents for evaluation can be indexed in a
               separate index than the documents for search.
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

<a name="memory.InMemoryDocumentStore.write_labels"></a>
#### write\_labels

```python
 | write_labels(labels: Union[List[dict], List[Label]], index: Optional[str] = None)
```

Write annotation labels into document store.

<a name="memory.InMemoryDocumentStore.get_document_by_id"></a>
#### get\_document\_by\_id

```python
 | get_document_by_id(id: str, index: Optional[str] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a name="memory.InMemoryDocumentStore.get_documents_by_id"></a>
#### get\_documents\_by\_id

```python
 | get_documents_by_id(ids: List[str], index: Optional[str] = None) -> List[Document]
```

Fetch documents by specifying a list of text id strings

<a name="memory.InMemoryDocumentStore.query_by_embedding"></a>
#### query\_by\_embedding

```python
 | query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: Index name for storing the docs and metadata
- `return_embedding`: To return document embedding

**Returns**:



<a name="memory.InMemoryDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever: BaseRetriever, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000)
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

<a name="memory.InMemoryDocumentStore.get_document_count"></a>
#### get\_document\_count

```python
 | get_document_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int
```

Return the number of documents in the document store.

<a name="memory.InMemoryDocumentStore.get_embedding_count"></a>
#### get\_embedding\_count

```python
 | get_embedding_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int
```

Return the count of embeddings in the document store.

<a name="memory.InMemoryDocumentStore.get_label_count"></a>
#### get\_label\_count

```python
 | get_label_count(index: Optional[str] = None) -> int
```

Return the number of labels in the document store

<a name="memory.InMemoryDocumentStore.get_all_documents_generator"></a>
#### get\_all\_documents\_generator

```python
 | get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> Generator[Document, None, None]
```

Get all documents from the document store. The methods returns a Python Generator that yields individual
documents.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.

<a name="memory.InMemoryDocumentStore.get_all_labels"></a>
#### get\_all\_labels

```python
 | get_all_labels(index: str = None, filters: Optional[Dict[str, List[str]]] = None) -> List[Label]
```

Return all labels in the document store

<a name="memory.InMemoryDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a name="memory.InMemoryDocumentStore.delete_documents"></a>
#### delete\_documents

```python
 | delete_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a name="sql"></a>
# Module sql

<a name="sql.SQLDocumentStore"></a>
## SQLDocumentStore Objects

```python
class SQLDocumentStore(BaseDocumentStore)
```

<a name="sql.SQLDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(url: str = "sqlite://", index: str = "document", label_index: str = "label", duplicate_documents: str = "overwrite")
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

<a name="sql.SQLDocumentStore.get_document_by_id"></a>
#### get\_document\_by\_id

```python
 | get_document_by_id(id: str, index: Optional[str] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a name="sql.SQLDocumentStore.get_documents_by_id"></a>
#### get\_documents\_by\_id

```python
 | get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000) -> List[Document]
```

Fetch documents by specifying a list of text id strings

<a name="sql.SQLDocumentStore.get_documents_by_vector_ids"></a>
#### get\_documents\_by\_vector\_ids

```python
 | get_documents_by_vector_ids(vector_ids: List[str], index: Optional[str] = None, batch_size: int = 10_000)
```

Fetch documents by specifying a list of text vector id strings

<a name="sql.SQLDocumentStore.get_all_documents_generator"></a>
#### get\_all\_documents\_generator

```python
 | get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> Generator[Document, None, None]
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

<a name="sql.SQLDocumentStore.get_all_labels"></a>
#### get\_all\_labels

```python
 | get_all_labels(index=None, filters: Optional[dict] = None)
```

Return all labels in the document store

<a name="sql.SQLDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None)
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

<a name="sql.SQLDocumentStore.write_labels"></a>
#### write\_labels

```python
 | write_labels(labels, index=None)
```

Write annotation labels into document store.

<a name="sql.SQLDocumentStore.update_vector_ids"></a>
#### update\_vector\_ids

```python
 | update_vector_ids(vector_id_map: Dict[str, str], index: Optional[str] = None, batch_size: int = 10_000)
```

Update vector_ids for given document_ids.

**Arguments**:

- `vector_id_map`: dict containing mapping of document_id -> vector_id.
- `index`: filter documents by the optional index attribute for documents in database.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a name="sql.SQLDocumentStore.reset_vector_ids"></a>
#### reset\_vector\_ids

```python
 | reset_vector_ids(index: Optional[str] = None)
```

Set vector IDs for all documents as None

<a name="sql.SQLDocumentStore.update_document_meta"></a>
#### update\_document\_meta

```python
 | update_document_meta(id: str, meta: Dict[str, str])
```

Update the metadata dictionary of a document by specifying its string id

<a name="sql.SQLDocumentStore.get_document_count"></a>
#### get\_document\_count

```python
 | get_document_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int
```

Return the number of documents in the document store.

<a name="sql.SQLDocumentStore.get_label_count"></a>
#### get\_label\_count

```python
 | get_label_count(index: Optional[str] = None) -> int
```

Return the number of labels in the document store

<a name="sql.SQLDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a name="sql.SQLDocumentStore.delete_documents"></a>
#### delete\_documents

```python
 | delete_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

<a name="faiss"></a>
# Module faiss

<a name="faiss.FAISSDocumentStore"></a>
## FAISSDocumentStore Objects

```python
class FAISSDocumentStore(SQLDocumentStore)
```

Document store for very large scale embedding based dense retrievers like the DPR.

It implements the FAISS library(https://github.com/facebookresearch/faiss)
to perform similarity search on vectors.

The document text and meta-data (for filtering) are stored using the SQLDocumentStore, while
the vector embeddings are indexed in a FAISS Index.

<a name="faiss.FAISSDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(sql_url: str = "sqlite:///", vector_dim: int = 768, faiss_index_factory_str: str = "Flat", faiss_index: Optional["faiss.swigfaiss.Index"] = None, return_embedding: bool = False, index: str = "document", similarity: str = "dot_product", embedding_field: str = "embedding", progress_bar: bool = True, duplicate_documents: str = 'overwrite', **kwargs, ,)
```

**Arguments**:

- `sql_url`: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                deployment, Postgres is recommended.
- `vector_dim`: the embedding vector size.
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
- `return_embedding`: To return document embedding
- `index`: Name of index in document store to use.
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default sine it is
           more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
- `embedding_field`: Name of field containing an embedding vector.
- `progress_bar`: Whether to show a tqdm progress bar or not.
                     Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicates document based on parameter options.
                            Parameter options : ( 'skip','overwrite','fail')
                            skip: Ignore the duplicates documents
                            overwrite: Update any existing documents with the same ID when adding documents.
                            fail: an error is raised if the document ID of the document being added already
                            exists.

<a name="faiss.FAISSDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None)
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



<a name="faiss.FAISSDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever: BaseRetriever, index: Optional[str] = None, update_existing_embeddings: bool = True, filters: Optional[Dict[str, List[str]]] = None, batch_size: int = 10_000)
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

<a name="faiss.FAISSDocumentStore.get_all_documents_generator"></a>
#### get\_all\_documents\_generator

```python
 | get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> Generator[Document, None, None]
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

<a name="faiss.FAISSDocumentStore.get_embedding_count"></a>
#### get\_embedding\_count

```python
 | get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int
```

Return the count of embeddings in the document store.

<a name="faiss.FAISSDocumentStore.train_index"></a>
#### train\_index

```python
 | train_index(documents: Optional[Union[List[dict], List[Document]]], embeddings: Optional[np.ndarray] = None, index: Optional[str] = None)
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

<a name="faiss.FAISSDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete all documents from the document store.

<a name="faiss.FAISSDocumentStore.delete_documents"></a>
#### delete\_documents

```python
 | delete_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete all documents from the document store.

<a name="faiss.FAISSDocumentStore.query_by_embedding"></a>
#### query\_by\_embedding

```python
 | query_by_embedding(query_emb: np.ndarray, filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: Index name to query the document from.
- `return_embedding`: To return document embedding

**Returns**:



<a name="faiss.FAISSDocumentStore.save"></a>
#### save

```python
 | save(file_path: Union[str, Path])
```

Save FAISS Index to the specified file.

**Arguments**:

- `file_path`: Path to save to.

**Returns**:

None

<a name="faiss.FAISSDocumentStore.load"></a>
#### load

```python
 | @classmethod
 | load(cls, faiss_file_path: Union[str, Path], sql_url: str, index: str)
```

Load a saved FAISS index from a file and connect to the SQL database.
Note: In order to have a correct mapping from FAISS to SQL,
      make sure to use the same SQL DB that you used when calling `save()`.

**Arguments**:

- `faiss_file_path`: Stored FAISS index file. Can be created via calling `save()`
- `sql_url`: Connection string to the SQL database that contains your docs and metadata.
- `index`: Index name to load the FAISS index as. It must match the index name used for
              when creating the FAISS index.

**Returns**:



<a name="milvus"></a>
# Module milvus

<a name="milvus.MilvusDocumentStore"></a>
## MilvusDocumentStore Objects

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

<a name="milvus.MilvusDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(sql_url: str = "sqlite:///", milvus_url: str = "tcp://localhost:19530", connection_pool: str = "SingletonThread", index: str = "document", vector_dim: int = 768, index_file_size: int = 1024, similarity: str = "dot_product", index_type: IndexType = IndexType.FLAT, index_param: Optional[Dict[str, Any]] = None, search_param: Optional[Dict[str, Any]] = None, return_embedding: bool = False, embedding_field: str = "embedding", progress_bar: bool = True, duplicate_documents: str = 'overwrite', **kwargs, ,)
```

**Arguments**:

- `sql_url`: SQL connection URL for storing document texts and metadata. It defaults to a local, file based SQLite DB. For large scale
                deployment, Postgres is recommended. If using MySQL then same server can also be used for
                Milvus metadata. For more details see https://milvus.io/docs/v1.0.0/data_manage.md.
- `milvus_url`: Milvus server connection URL for storing and processing vectors.
                   Protocol, host and port will automatically be inferred from the URL.
                   See https://milvus.io/docs/v1.0.0/install_milvus.md for instructions to start a Milvus instance.
- `connection_pool`: Connection pool type to connect with Milvus server. Default: "SingletonThread".
- `index`: Index name for text, embedding and metadata (in Milvus terms, this is the "collection name").
- `vector_dim`: The embedding vector size. Default: 768.
- `index_file_size`: Specifies the size of each segment file that is stored by Milvus and its default value is 1024 MB.
When the size of newly inserted vectors reaches the specified volume, Milvus packs these vectors into a new segment.
Milvus creates one index file for each segment. When conducting a vector search, Milvus searches all index files one by one.
As a rule of thumb, we would see a 30% ~ 50% increase in the search performance after changing the value of index_file_size from 1024 to 2048.
Note that an overly large index_file_size value may cause failure to load a segment into the memory or graphics memory.
(From https://milvus.io/docs/v1.0.0/performance_faq.md#How-can-I-get-the-best-performance-from-Milvus-through-setting-index_file_size)
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default and recommended for DPR embeddings.
                   'cosine' is recommended for Sentence Transformers, but is not directly supported by Milvus.
                   However, you can normalize your embeddings and use `dot_product` to get the same results.
                   See https://milvus.io/docs/v1.0.0/metric.md?Inner-product-(IP)`floating`.
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

<a name="milvus.MilvusDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None, index_param: Optional[Dict[str, Any]] = None)
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



<a name="milvus.MilvusDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever: BaseRetriever, index: Optional[str] = None, batch_size: int = 10_000, update_existing_embeddings: bool = True, filters: Optional[Dict[str, List[str]]] = None)
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

<a name="milvus.MilvusDocumentStore.query_by_embedding"></a>
#### query\_by\_embedding

```python
 | query_by_embedding(query_emb: np.ndarray, filters: Optional[dict] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None) -> List[Document]
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



<a name="milvus.MilvusDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete all documents (from SQL AND Milvus).

**Arguments**:

- `index`: (SQL) index name for storing the docs and metadata
- `filters`: Optional filters to narrow down the search space.
                Example: {"name": ["some", "more"], "category": ["only_one"]}

**Returns**:

None

<a name="milvus.MilvusDocumentStore.delete_documents"></a>
#### delete\_documents

```python
 | delete_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete all documents (from SQL AND Milvus).

**Arguments**:

- `index`: (SQL) index name for storing the docs and metadata
- `filters`: Optional filters to narrow down the search space.
                Example: {"name": ["some", "more"], "category": ["only_one"]}

**Returns**:

None

<a name="milvus.MilvusDocumentStore.get_all_documents_generator"></a>
#### get\_all\_documents\_generator

```python
 | get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> Generator[Document, None, None]
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

<a name="milvus.MilvusDocumentStore.get_all_documents"></a>
#### get\_all\_documents

```python
 | get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> List[Document]
```

Get documents from the document store (optionally using filter criteria).

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a name="milvus.MilvusDocumentStore.get_document_by_id"></a>
#### get\_document\_by\_id

```python
 | get_document_by_id(id: str, index: Optional[str] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

**Arguments**:

- `id`: ID of the document
- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.

<a name="milvus.MilvusDocumentStore.get_documents_by_id"></a>
#### get\_documents\_by\_id

```python
 | get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000) -> List[Document]
```

Fetch multiple documents by specifying their IDs (strings)

**Arguments**:

- `ids`: List of IDs of the documents
- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a name="milvus.MilvusDocumentStore.get_all_vectors"></a>
#### get\_all\_vectors

```python
 | get_all_vectors(index: Optional[str] = None) -> List[np.ndarray]
```

Helper function to dump all vectors stored in Milvus server.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.

**Returns**:

List[np.array]: List of vectors.

<a name="milvus.MilvusDocumentStore.get_embedding_count"></a>
#### get\_embedding\_count

```python
 | get_embedding_count(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int
```

Return the count of embeddings in the document store.

<a name="weaviate"></a>
# Module weaviate

<a name="weaviate.WeaviateDocumentStore"></a>
## WeaviateDocumentStore Objects

```python
class WeaviateDocumentStore(BaseDocumentStore)
```

Weaviate is a cloud-native, modular, real-time vector search engine built to scale your machine learning models.
(See https://www.semi.technology/developers/weaviate/current/index.html#what-is-weaviate)

Some of the key differences in contrast to FAISS & Milvus:
1. Stores everything in one place: documents, meta data and vectors - so less network overhead when scaling this up
2. Allows combination of vector search and scalar filtering, i.e. you can filter for a certain tag and do dense retrieval on that subset 
3. Has less variety of ANN algorithms, as of now only HNSW.  

Weaviate python client is used to connect to the server, more details are here
https://weaviate-python-client.readthedocs.io/en/docs/weaviate.html

Usage:
1. Start a Weaviate server (see https://www.semi.technology/developers/weaviate/current/getting-started/installation.html)
2. Init a WeaviateDocumentStore in Haystack

<a name="weaviate.WeaviateDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(host: Union[str, List[str]] = "http://localhost", port: Union[int, List[int]] = 8080, timeout_config: tuple = (5, 15), username: str = None, password: str = None, index: str = "Document", embedding_dim: int = 768, text_field: str = "text", name_field: str = "name", faq_question_field="question", similarity: str = "dot_product", index_type: str = "hnsw", custom_schema: Optional[dict] = None, return_embedding: bool = False, embedding_field: str = "embedding", progress_bar: bool = True, duplicate_documents: str = 'overwrite', **kwargs, ,)
```

**Arguments**:

- `host`: Weaviate server connection URL for storing and processing documents and vectors.
                     For more details, refer "https://www.semi.technology/developers/weaviate/current/getting-started/installation.html"
- `port`: port of Weaviate instance
- `timeout_config`: Weaviate Timeout config as a tuple of (retries, time out seconds).
- `username`: username (standard authentication via http_auth)
- `password`: password (standard authentication via http_auth)
- `index`: Index name for document text, embedding and metadata (in Weaviate terminology, this is a "Class" in Weaviate schema).
- `embedding_dim`: The embedding vector size. Default: 768.
- `text_field`: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
                   If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
- `name_field`: Name of field that contains the title of the the doc
- `faq_question_field`: Name of field containing the question in case of FAQ-Style QA
- `similarity`: The similarity function used to compare document vectors. 'dot_product' is the default.
- `index_type`: Index type of any vector object defined in weaviate schema. The vector index type is pluggable.
                   Currently, HSNW is only supported.
                   See: https://www.semi.technology/developers/weaviate/current/more-resources/performance.html
- `custom_schema`: Allows to create custom schema in Weaviate, for more details
                   See https://www.semi.technology/developers/weaviate/current/data-schema/schema-configuration.html
- `module_name`: Vectorization module to convert data into vectors. Default is "text2vec-trasnformers"
                    For more details, See https://www.semi.technology/developers/weaviate/current/modules/
- `return_embedding`: To return document embedding.
- `embedding_field`: Name of field containing an embedding vector.
- `progress_bar`: Whether to show a tqdm progress bar or not.
                     Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_documents`: Handle duplicates document based on parameter options.
                            Parameter options : ( 'skip','overwrite','fail')
                            skip: Ignore the duplicates documents
                            overwrite: Update any existing documents with the same ID when adding documents.
                            fail: an error is raised if the document ID of the document being added already exists.

<a name="weaviate.WeaviateDocumentStore.get_document_by_id"></a>
#### get\_document\_by\_id

```python
 | get_document_by_id(id: str, index: Optional[str] = None) -> Optional[Document]
```

Fetch a document by specifying its text id string

<a name="weaviate.WeaviateDocumentStore.get_documents_by_id"></a>
#### get\_documents\_by\_id

```python
 | get_documents_by_id(ids: List[str], index: Optional[str] = None, batch_size: int = 10_000) -> List[Document]
```

Fetch documents by specifying a list of text id strings

<a name="weaviate.WeaviateDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000, duplicate_documents: Optional[str] = None)
```

Add new documents to the DocumentStore.

**Arguments**:

- `documents`: List of `Dicts` or List of `Documents`. Passing an Embedding/Vector is mandatory in case weaviate is not
                configured with a module. If a module is configured, the embedding is automatically generated by Weaviate.
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

<a name="weaviate.WeaviateDocumentStore.update_document_meta"></a>
#### update\_document\_meta

```python
 | update_document_meta(id: str, meta: Dict[str, str])
```

Update the metadata dictionary of a document by specifying its string id

<a name="weaviate.WeaviateDocumentStore.get_document_count"></a>
#### get\_document\_count

```python
 | get_document_count(filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int
```

Return the number of documents in the document store.

<a name="weaviate.WeaviateDocumentStore.get_all_documents"></a>
#### get\_all\_documents

```python
 | get_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> List[Document]
```

Get documents from the document store.

**Arguments**:

- `index`: Name of the index to get the documents from. If None, the
              DocumentStore's default index (self.index) will be used.
- `filters`: Optional filters to narrow down the documents to return.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `return_embedding`: Whether to return the document embeddings.
- `batch_size`: When working with large number of documents, batching can help reduce memory footprint.

<a name="weaviate.WeaviateDocumentStore.get_all_documents_generator"></a>
#### get\_all\_documents\_generator

```python
 | get_all_documents_generator(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, return_embedding: Optional[bool] = None, batch_size: int = 10_000) -> Generator[Document, None, None]
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

<a name="weaviate.WeaviateDocumentStore.query"></a>
#### query

```python
 | query(query: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, top_k: int = 10, custom_query: Optional[str] = None, index: Optional[str] = None) -> List[Document]
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

<a name="weaviate.WeaviateDocumentStore.query_by_embedding"></a>
#### query\_by\_embedding

```python
 | query_by_embedding(query_emb: np.ndarray, filters: Optional[dict] = None, top_k: int = 10, index: Optional[str] = None, return_embedding: Optional[bool] = None) -> List[Document]
```

Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

**Arguments**:

- `query_emb`: Embedding of the query (e.g. gathered from DPR)
- `filters`: Optional filters to narrow down the search space.
                Example: {"name": ["some", "more"], "category": ["only_one"]}
- `top_k`: How many documents to return
- `index`: index name for storing the docs and metadata
- `return_embedding`: To return document embedding

**Returns**:



<a name="weaviate.WeaviateDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, update_existing_embeddings: bool = True, batch_size: int = 10_000)
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

<a name="weaviate.WeaviateDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None)
```

Delete documents in an index. All documents are deleted if no filters are passed.

**Arguments**:

- `index`: Index name to delete the document from.
- `filters`: Optional filters to narrow down the documents to be deleted.

**Returns**:

None

