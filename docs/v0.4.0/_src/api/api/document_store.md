<a name="elasticsearch"></a>
# elasticsearch

<a name="elasticsearch.ElasticsearchDocumentStore"></a>
## ElasticsearchDocumentStore

```python
class ElasticsearchDocumentStore(BaseDocumentStore)
```

<a name="elasticsearch.ElasticsearchDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(host: str = "localhost", port: int = 9200, username: str = "", password: str = "", index: str = "document", label_index: str = "label", search_fields: Union[str, list] = "text", text_field: str = "text", name_field: str = "name", embedding_field: str = "embedding", embedding_dim: int = 768, custom_mapping: Optional[dict] = None, excluded_meta_data: Optional[list] = None, faq_question_field: Optional[str] = None, scheme: str = "http", ca_certs: bool = False, verify_certs: bool = True, create_index: bool = True, update_existing_documents: bool = False, refresh_type: str = "wait_for")
```

A DocumentStore using Elasticsearch to store and query the documents for our search.

* Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or boosts to your queries, and storing embeddings
* You can either use an existing Elasticsearch index or create a new one via haystack
* Retrievers operate on top of this DocumentStore to find the relevant documents for a query

**Arguments**:

- `host`: url of elasticsearch
- `port`: port of elasticsearch
- `username`: username
- `password`: password
- `index`: Name of index in elasticsearch to use. If not existing yet, we will create one.
- `search_fields`: Name of fields used by ElasticsearchRetriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
- `text_field`: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
- `name_field`: Name of field that contains the title of the the doc
- `embedding_field`: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
- `embedding_dim`: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
- `custom_mapping`: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
- `excluded_meta_data`: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
- `scheme`: 'https' or 'http', protocol used to connect to your elasticsearch instance
- `ca_certs`: Root certificates for SSL
- `verify_certs`: Whether to be strict about ca certificates
- `create_index`: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case)
- `update_existing_documents`: Whether to update any existing documents with the same ID when adding
documents. When set as True, any document with an existing ID gets updated.
If set to False, an error is raised if the document ID of the document being
added already exists.
- `refresh_type`: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
Values:
- 'wait_for' => continue only after changes are visible (slow, but safe)
- 'false' => continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion)
More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html

<a name="elasticsearch.ElasticsearchDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None)
```

Indexes documents for later queries in Elasticsearch.

When using explicit document IDs, any existing document with the same ID gets updated.

**Arguments**:

- `documents`: a list of Python dictionaries or a list of Haystack Document objects.
For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
Optionally: Include meta data via {"text": "<the-actual-text>",
"meta":{"name": "<some-document-name>, "author": "somebody", ...}}
It can be used for filtering and is accessible in the responses of the Finder.
Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
should be changed to what you have set for self.text_field and self.name_field.
- `index`: Elasticsearch index where the documents should be indexed. If not supplied, self.index will be used.

**Returns**:

None

<a name="elasticsearch.ElasticsearchDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever: BaseRetriever, index: Optional[str] = None)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.
This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever
- `index`: Index name to update

**Returns**:

None

<a name="elasticsearch.ElasticsearchDocumentStore.add_eval_data"></a>
#### add\_eval\_data

```python
 | add_eval_data(filename: str, doc_index: str = "eval_document", label_index: str = "label")
```

Adds a SQuAD-formatted file to the DocumentStore in order to be able to perform evaluation on it.

**Arguments**:

- `filename`: Name of the file containing evaluation data
:type filename: str
- `doc_index`: Elasticsearch index where evaluation documents should be stored
:type doc_index: str
- `label_index`: Elasticsearch index where labeled questions should be stored
:type label_index: str

<a name="elasticsearch.ElasticsearchDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: str)
```

Delete all documents in an index.

**Arguments**:

- `index`: index name

**Returns**:

None

<a name="memory"></a>
# memory

<a name="memory.InMemoryDocumentStore"></a>
## InMemoryDocumentStore

```python
class InMemoryDocumentStore(BaseDocumentStore)
```

In-memory document store

<a name="memory.InMemoryDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None)
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

**Returns**:

None

<a name="memory.InMemoryDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever: BaseRetriever, index: Optional[str] = None)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.
This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever
- `index`: Index name to update

**Returns**:

None

<a name="memory.InMemoryDocumentStore.add_eval_data"></a>
#### add\_eval\_data

```python
 | add_eval_data(filename: str, doc_index: Optional[str] = None, label_index: Optional[str] = None)
```

Adds a SQuAD-formatted file to the DocumentStore in order to be able to perform evaluation on it.

**Arguments**:

- `filename`: Name of the file containing evaluation data
:type filename: str
- `doc_index`: Elasticsearch index where evaluation documents should be stored
:type doc_index: str
- `label_index`: Elasticsearch index where labeled questions should be stored
:type label_index: str

<a name="memory.InMemoryDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None)
```

Delete all documents in a index.

**Arguments**:

- `index`: index name

**Returns**:

None

<a name="sql"></a>
# sql

<a name="sql.SQLDocumentStore"></a>
## SQLDocumentStore

```python
class SQLDocumentStore(BaseDocumentStore)
```

<a name="sql.SQLDocumentStore.write_documents"></a>
#### write\_documents

```python
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None)
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

**Returns**:

None

<a name="sql.SQLDocumentStore.update_vector_ids"></a>
#### update\_vector\_ids

```python
 | update_vector_ids(vector_id_map: Dict[str, str], index: Optional[str] = None)
```

Update vector_ids for given document_ids.

**Arguments**:

- `vector_id_map`: dict containing mapping of document_id -> vector_id.
- `index`: filter documents by the optional index attribute for documents in database.

<a name="sql.SQLDocumentStore.add_eval_data"></a>
#### add\_eval\_data

```python
 | add_eval_data(filename: str, doc_index: str = "eval_document", label_index: str = "label")
```

Adds a SQuAD-formatted file to the DocumentStore in order to be able to perform evaluation on it.

**Arguments**:

- `filename`: Name of the file containing evaluation data
:type filename: str
- `doc_index`: Elasticsearch index where evaluation documents should be stored
:type doc_index: str
- `label_index`: Elasticsearch index where labeled questions should be stored
:type label_index: str

<a name="sql.SQLDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index=None)
```

Delete all documents in a index.

**Arguments**:

- `index`: index name

**Returns**:

None

<a name="base"></a>
# base

<a name="base.BaseDocumentStore"></a>
## BaseDocumentStore

```python
class BaseDocumentStore(ABC)
```

Base class for implementing Document Stores.

<a name="base.BaseDocumentStore.write_documents"></a>
#### write\_documents

```python
 | @abstractmethod
 | write_documents(documents: Union[List[dict], List[Document]], index: Optional[str] = None)
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

**Returns**:

None

<a name="faiss"></a>
# faiss

<a name="faiss.FAISSDocumentStore"></a>
## FAISSDocumentStore

```python
class FAISSDocumentStore(SQLDocumentStore)
```

Document store for very large scale embedding based dense retrievers like the DPR.

It implements the FAISS library(https://github.com/facebookresearch/faiss)
to perform similarity search on vectors.

The document text and meta-data(for filtering) is stored using the SQLDocumentStore, while
the vector embeddings are indexed in a FAISS Index.

<a name="faiss.FAISSDocumentStore.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(sql_url: str = "sqlite:///", index_buffer_size: int = 10_000, vector_size: int = 768, faiss_index: Optional[IndexHNSWFlat] = None)
```

**Arguments**:

- `sql_url`: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
deployment, Postgres is recommended.
- `index_buffer_size`: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered in
smaller chunks to reduce memory footprint.
- `vector_size`: the embedding vector size.
- `faiss_index`: load an existing FAISS Index.

<a name="faiss.FAISSDocumentStore.update_embeddings"></a>
#### update\_embeddings

```python
 | update_embeddings(retriever: BaseRetriever, index: Optional[str] = None)
```

Updates the embeddings in the the document store using the encoding model specified in the retriever.
This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

**Arguments**:

- `retriever`: Retriever to use to get embeddings for text
- `index`: Index name to update

**Returns**:

None

<a name="faiss.FAISSDocumentStore.save"></a>
#### save

```python
 | save(file_path: Union[str, Path])
```

Save FAISS Index to the specified file.

<a name="faiss.FAISSDocumentStore.load"></a>
#### load

```python
 | @classmethod
 | load(cls, faiss_file_path: Union[str, Path], sql_url: str, index_buffer_size: int = 10_000, vector_size: int = 768)
```

Load a saved FAISS index from a file and connect to the SQL database.

