<!---
title: "Database"
metaTitle: "Database"
metaDescription: ""
slug: "/docs/apidatabase"
date: "2020-09-03"
id: "apidatabasemd"
--->

Get Started
===========
Installation
--------------
<div class="filter">
<a href="#floating">Floating point embeddings</a> <a href="#binary">Binary embeddings</a>
</div>
<div class="filter-floating table-wrapper" markdown="block">
    <div class="filter1">
    <a href="#first">Floating point embeddings</a> <a href="#second">Binary embeddings</a>
    </div>
    <div class="filter1-first table-wrapper" markdown="block">
    zzz
    </div>
    <div class="filter1-second table-wrapper" markdown="block">
    lll
    </div>
</div>
<div class="filter-binary table-wrapper" markdown="block">
yyyyy
</div>

<a name="database.elasticsearch"></a>
# database.elasticsearch

<a name="database.elasticsearch.ElasticsearchDocumentStore"></a>
## ElasticsearchDocumentStore Objects

```python
class ElasticsearchDocumentStore(BaseDocumentStore)
```

<a name="database.elasticsearch.ElasticsearchDocumentStore.__init__"></a>
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
- 'false' => continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after indexing)
More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html

<a name="database.elasticsearch.ElasticsearchDocumentStore.write_documents"></a>
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

<a name="database.elasticsearch.ElasticsearchDocumentStore.update_embeddings"></a>
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

<a name="database.elasticsearch.ElasticsearchDocumentStore.add_eval_data"></a>
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

<a name="database.elasticsearch.ElasticsearchDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: str)
```

Delete all documents in an index.

**Arguments**:

- `index`: index name

**Returns**:

None

<a name="database.memory"></a>
# database.memory

<a name="database.memory.InMemoryDocumentStore"></a>
## InMemoryDocumentStore Objects

```python
class InMemoryDocumentStore(BaseDocumentStore)
```

In-memory document store

<a name="database.memory.InMemoryDocumentStore.write_documents"></a>
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

<a name="database.memory.InMemoryDocumentStore.update_embeddings"></a>
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

<a name="database.memory.InMemoryDocumentStore.add_eval_data"></a>
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

<a name="database.memory.InMemoryDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index: Optional[str] = None)
```

Delete all documents in a index.

**Arguments**:

- `index`: index name

**Returns**:

None

<a name="database.sql"></a>
# database.sql

<a name="database.sql.SQLDocumentStore"></a>
## SQLDocumentStore Objects

```python
class SQLDocumentStore(BaseDocumentStore)
```

<a name="database.sql.SQLDocumentStore.write_documents"></a>
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

<a name="database.sql.SQLDocumentStore.add_eval_data"></a>
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

<a name="database.sql.SQLDocumentStore.delete_all_documents"></a>
#### delete\_all\_documents

```python
 | delete_all_documents(index=None)
```

Delete all documents in a index.

**Arguments**:

- `index`: index name

**Returns**:

None

<a name="database.base"></a>
# database.base

<a name="database.base.Document"></a>
## Document Objects

```python
class Document()
```

<a name="database.base.Document.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(text: str, id: str = None, query_score: Optional[float] = None, question: Optional[str] = None, meta: Optional[Dict[str, Any]] = None, embedding: Optional[np.array] = None)
```

Object used to represent documents / passages in a standardized way within Haystack.
For example, this is what the retriever will return from the DocumentStore,
regardless if it's ElasticsearchDocumentStore or InMemoryDocumentStore.

Note that there can be multiple Documents originating from one file (e.g. PDF),
if you split the text into smaller passages. We'll have one Document per passage in this case.

**Arguments**:

- `id`: ID used within the DocumentStore
- `text`: Text of the document
- `query_score`: Retriever's query score for a retrieved document
- `question`: Question text for FAQs.
- `meta`: Meta fields for a document like name, url, or author.
- `embedding`: Vector encoding of the text

<a name="database.base.Label"></a>
## Label Objects

```python
class Label()
```

<a name="database.base.Label.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(question: str, answer: str, is_correct_answer: bool, is_correct_document: bool, origin: str, document_id: Optional[str] = None, offset_start_in_doc: Optional[int] = None, no_answer: Optional[bool] = None, model_id: Optional[int] = None)
```

Object used to represent label/feedback in a standardized way within Haystack.
This includes labels from dataset like SQuAD, annotations from labeling tools,
or, user-feedback from the Haystack REST API.

**Arguments**:

- `question`: the question(or query) for finding answers.
- `answer`: the answer string.
- `is_correct_answer`: whether the sample is positive or negative.
- `is_correct_document`: in case of negative sample(is_correct_answer is False), there could be two cases;
incorrect answer but correct document & incorrect document. This flag denotes if
the returned document was correct.
- `origin`: the source for the labels. It can be used to later for filtering.
- `document_id`: the document_store's ID for the returned answer document.
- `offset_start_in_doc`: the answer start offset in the document.
- `no_answer`: whether the question in unanswerable.
- `model_id`: model_id used for prediction (in-case of user feedback).

<a name="database.base.MultiLabel"></a>
## MultiLabel Objects

```python
class MultiLabel()
```

<a name="database.base.MultiLabel.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(question: str, multiple_answers: List[str], is_correct_answer: bool, is_correct_document: bool, origin: str, multiple_document_ids: List[Any], multiple_offset_start_in_docs: List[Any], no_answer: Optional[bool] = None, model_id: Optional[int] = None)
```

Object used to aggregate multiple possible answers for the same question

**Arguments**:

- `question`: the question(or query) for finding answers.
- `multiple_answers`: list of possible answer strings
- `is_correct_answer`: whether the sample is positive or negative.
- `is_correct_document`: in case of negative sample(is_correct_answer is False), there could be two cases;
incorrect answer but correct document & incorrect document. This flag denotes if
the returned document was correct.
- `origin`: the source for the labels. It can be used to later for filtering.
- `multiple_document_ids`: the document_store's IDs for the returned answer documents.
- `multiple_offset_start_in_docs`: the answer start offsets in the document.
- `no_answer`: whether the question in unanswerable.
- `model_id`: model_id used for prediction (in-case of user feedback).

<a name="database.base.BaseDocumentStore"></a>
## BaseDocumentStore Objects

```python
class BaseDocumentStore(ABC)
```

Base class for implementing Document Stores.

<a name="database.base.BaseDocumentStore.write_documents"></a>
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

