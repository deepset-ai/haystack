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

Initialising a new DocumentStore is straight forward.

<div class="tabs tabsdsinstall">

<div class="tab">
<input type="radio" id="tab-1-1" name="tab-group-1" checked>
<label class="labelouter" for="tab-1-1">Elasticsearch</label>
<div class="tabcontent">

```python
document_store = ElasticsearchDocumentStore()
# or
document_store = OpenDistroElasticsearchDocumentStore()

```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-2" name="tab-group-1">
<label class="labelouter" for="tab-1-2">FAISS</label>
<div class="tabcontent">

```python
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-3" name="tab-group-1">
<label class="labelouter" for="tab-1-3">In Memory</label>
<div class="tabcontent">

```python
document_store = InMemoryDocumentStore()
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-4" name="tab-group-1">
<label class="labelouter" for="tab-1-4">SQL</label>
<div class="tabcontent">

```python
document_store = SQLDocumentStore()
```

</div>
</div>

</div>

Each DocumentStore constructor allows for arguments specifying how to connect to existing databases and the names of indexes.
See API documentation for more info.

## Input Format

DocumentStores expect Documents in dictionary form, like that below.
They are loaded using the `DocumentStore.write_documents()` method.
See [Preprocessing](/docs/latest/preprocessingmd) for more information on the cleaning and splitting steps that will help you maximize Haystack's performance.

[//]: # (Add link to preprocessing section)

```python
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
<label class="labelouter" for="tab-2-2">FAISS</label>
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
<input type="radio" id="tab-2-3" name="tab-group-2">
<label class="labelouter" for="tab-2-3">In Memory</label>
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
<input type="radio" id="tab-2-4" name="tab-group-2">
<label class="labelouter" for="tab-2-4">SQL</label>
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

</div>

<div class="recommendation">

#### Our Recommendations

**Restricted environment:** Use the `InMemoryDocumentStore`, if you are just giving Haystack a quick try on a small sample and are working in a restricted environment that complicates running Elasticsearch or other databases  

**Allrounder:** Use the `ElasticSearchDocumentStore`, if you want to evaluate the performance of different retrieval options (dense vs. sparse) and are aiming for a smooth transition from PoC to production

**Vector Specialist:** Use the `FAISSDocumentStore`, if you want to focus on dense retrieval and possibly deal with larger datasets

</div>
