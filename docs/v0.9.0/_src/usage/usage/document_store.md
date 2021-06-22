<!---
title: "Document Store"
metaTitle: "Document Store"
metaDescription: ""
slug: "/docs/documentstore"
date: "2020-09-03"
id: "documentstoremd"
--->


# DocumentStores

You can think of the DocumentStore as a "database" that:
- stores your texts and meta data
- provides them to the retriever at query time

There are different DocumentStores in Haystack to fit different use cases and tech stacks.

## Initialisation

Initialising a new DocumentStore within Haystack is straight forward.

<div class="tabs tabsdsinstall">

<div class="tab">
<input type="radio" id="tab-1-1" name="tab-group-1" checked>
<label class="labelouter" for="tab-1-1">Elasticsearch</label>
<div class="tabcontent">

[Install](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)
Elasticsearch and then [start](https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html)
an instance. 

If you have Docker set up, we recommend pulling the Docker image and running it.
```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.9.2
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
```

Note that we also have a utility function `haystack.utils.launch_es` that can start up an Elasticsearch instance.

Next you can initialize the Haystack object that will connect to this instance.

```python
from haystack.document_store import ElasticsearchDocumentStore

document_store = ElasticsearchDocumentStore()
```

Note that we also support [Open Distro for Elasticsearch](https://opendistro.github.io/for-elasticsearch-docs/).
Follow [their documentation](https://opendistro.github.io/for-elasticsearch-docs/docs/install/)
to run it and connect to it using Haystack's `OpenDistroElasticsearchDocumentStore` class.

We further support [AWS Elastic Search Service](https://aws.amazon.com/elasticsearch-service/) with [signed Requests](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html):
Use e.g. [aws-requests-auth](https://github.com/davidmuller/aws-requests-auth) to create an auth object and pass it as `aws4auth` to the `ElasticsearchDocumentStore` constructor.

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-2" name="tab-group-1">
<label class="labelouter" for="tab-1-2">Milvus</label>
<div class="tabcontent">

Follow the [official documentation](https://www.milvus.io/docs/v1.0.0/milvus_docker-cpu.md) to start a Milvus instance via Docker. 
Note that we also have a utility function `haystack.utils.launch_milvus` that can start up a Milvus instance.

You can initialize the Haystack object that will connect to this instance as follows:
```python
from haystack.document_store import MilvusDocumentStore

document_store = MilvusDocumentStore()
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-3" name="tab-group-1">
<label class="labelouter" for="tab-1-3">FAISS</label>
<div class="tabcontent">

The `FAISSDocumentStore` requires no external setup. Start it by simply using this line.
```python
from haystack.document_store import FAISSDocumentStore

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-4" name="tab-group-1">
<label class="labelouter" for="tab-1-4">In Memory</label>
<div class="tabcontent">

The `InMemoryDocumentStore()` requires no external setup. Start it by simply using this line.
```python
from haystack.document_store import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-5" name="tab-group-1">
<label class="labelouter" for="tab-1-5">SQL</label>
<div class="tabcontent">

The `SQLDocumentStore` requires SQLite, PostgresQL or MySQL to be installed and started.
Note that SQLite already comes packaged with most operating systems.

```python
from haystack.document_store import SQLDocumentStore

document_store = SQLDocumentStore()
```

</div>
</div>
    
<div class="tab">
<input type="radio" id="tab-1-6" name="tab-group-1">
<label class="labelouter" for="tab-1-6">Weaviate</label>
<div class="tabcontent">

The `WeaviateDocumentStore` requires a running Weaviate Server. 
You can start a basic instance like this (see the [Weaviate docs](https://www.semi.technology/developers/weaviate/current/) for details):
```
    docker run -d -p 8080:8080 --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED='true' --env PERSISTENCE_DATA_PATH='/var/lib/weaviate' semitechnologies/weaviate:1.4.0
```
  
Afterwards, you can use it in Haystack:
```python
from haystack.document_store import WeaviateDocumentStore

document_store = WeaviateDocumentStore()
```
    
</div>
</div>

</div>

Each DocumentStore constructor allows for arguments specifying how to connect to existing databases and the names of indexes.
See API documentation for more info.

## Input Format

DocumentStores expect Documents in dictionary form, like that below.
They are loaded using the `DocumentStore.write_documents()` method.
See [Preprocessing](/docs/v0.9.0/preprocessingmd) for more information on the cleaning and splitting steps that will help you maximize Haystack's performance.

[//]: # (Add link to preprocessing section)

```python
from haystack.document_store import ElasticsearchDocumentStore

document_store = ElasticsearchDocumentStore()
dicts = [
    {
        'text': DOCUMENT_TEXT_HERE,
        'meta': {'name': DOCUMENT_NAME, ...}
    }, ...
]
document_store.write_documents(dicts)
```

## Writing Documents (Sparse Retrievers)

Haystack allows for you to write store documents in an optimised fashion so that query times can be kept low.
For **sparse**, keyword based retrievers such as BM25 and TF-IDF,
you simply have to call `DocumentStore.write_documents()`.
The creation of the inverted index which optimises querying speed is handled automatically.

```python
document_store.write_documents(dicts)
```

## Writing Documents (Dense Retrievers)

For **dense** neural network based retrievers like Dense Passage Retrieval, or Embedding Retrieval,
indexing involves computing the Document embeddings which will be compared against the Query embedding.

The storing of the text is handled by `DocumentStore.write_documents()` and the computation of the
embeddings is started by `DocumentStore.update_embeddings()`.

```python
document_store.write_documents(dicts)
document_store.update_embeddings(retriever)
```

This step is computationally intensive since it will engage the transformer based encoders.
Having GPU acceleration will significantly speed this up.

<!-- _comment: !! Diagrams of inverted index / document embeds !! -->
<!-- _comment: !! Make this a tab element to show how different datastores are initialized !! -->
## Choosing the Right Document Store

The Document Stores have different characteristics. You should choose one depending on the maturity of your project, the use case and technical environment:

<div class="tabs tabsdschoose">

<div class="tab">
<input type="radio" id="tab-2-1" name="tab-group-2" checked>
<label class="labelouter" for="tab-2-1">Elasticsearch</label>
<div class="tabcontent">

**Pros:**
- Fast & accurate sparse retrieval with many tuning options
- Basic support for dense retrieval
- Production-ready
- Support also for Open Distro

**Cons:**
- Slow for dense retrieval with more than ~ 1 Mio documents

</div>
</div>

<div class="tab">
<input type="radio" id="tab-2-2" name="tab-group-2">
<label class="labelouter" for="tab-2-2">Milvus</label>
<div class="tabcontent">

**Pros:**
- Scalable DocumentStore that excels at handling vectors (hence suited to dense retrieval methods like DPR)
- Encapsulates multiple ANN libraries (e.g. FAISS and ANNOY) and provides added reliability
- Runs as a separate service (e.g. a Docker container)
- Allows dynamic data management

**Cons:**
- No efficient sparse retrieval

</div>
</div>

<div class="tab">
<input type="radio" id="tab-2-3" name="tab-group-2">
<label class="labelouter" for="tab-2-3">FAISS</label>
<div class="tabcontent">

**Pros:**
- Fast & accurate dense retrieval
- Highly scalable due to approximate nearest neighbour algorithms (ANN)
- Many options to tune dense retrieval via different index types (more info [here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index))

**Cons:**
- No efficient sparse retrieval

</div>
</div>

<div class="tab">
<input type="radio" id="tab-2-4" name="tab-group-2">
<label class="labelouter" for="tab-2-4">In Memory</label>
<div class="tabcontent">

**Pros:**
- Simple
- Exists already in many environments

**Cons:**
- Only compatible with minimal TF-IDF Retriever
- Bad retrieval performance
- Not recommended for production

</div>
</div>

<div class="tab">
<input type="radio" id="tab-2-5" name="tab-group-2">
<label class="labelouter" for="tab-2-5">SQL</label>
<div class="tabcontent">

**Pros:**
- Simple & fast to test
- No database requirements
- Supports MySQL, PostgreSQL and SQLite

**Cons:**
- Not scalable
- Not persisting your data on disk

</div>
</div>

    
<div class="tab">
<input type="radio" id="tab-2-6" name="tab-group-2">
<label class="labelouter" for="tab-2-6">Weaviate</label>
<div class="tabcontent">

**Pros:**
- Simple vector search
- Stores everything in one place: documents, meta data and vectors - so less network overhead when scaling this up
- Allows combination of vector search and scalar filtering, i.e. you can filter for a certain tag and do dense retrieval on that subset 

**Cons:**
- Less options for ANN algorithms than FAISS or Milvus
- No BM25 / Tf-idf retrieval
    
</div>
</div>
    
</div>

<div class="recommendation">

#### Our Recommendations

**Restricted environment:** Use the `InMemoryDocumentStore`, if you are just giving Haystack a quick try on a small sample and are working in a restricted environment that complicates running Elasticsearch or other databases

**Allrounder:** Use the `ElasticSearchDocumentStore`, if you want to evaluate the performance of different retrieval options (dense vs. sparse) and are aiming for a smooth transition from PoC to production

**Vector Specialist:** Use the `MilvusDocumentStore`, if you want to focus on dense retrieval and possibly deal with larger datasets

</div>
