<!---
title: "Database"
metaTitle: "Document Store"
metaDescription: ""
slug: "/docs/database"
date: "2020-09-03"
id: "databasemd"
--->


# Document Stores

You can think of the Document Store as a "database" that:
- stores your texts and meta data  
- provides them to the retriever at query time 

There are different DocumentStores in Haystack to fit different use cases and tech stacks. 

## Initialisation

Initialising a new Document Store is straight forward.

<div class="tabs tabsdsinstall">

<div class="tab">
<input type="radio" id="tab-1-1" name="tab-group-1" checked>
<label class="labelouter" for="tab-1-1">Elasticsearch</label>
<div class="tabcontent">

```python
document_store = ElasticsearchDocumentStore()
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-2" name="tab-group-1">
<label class="labelouter" for="tab-1-2">FAISS</label>
<div class="tabcontent">

```python
document_store = FAISSDocumentStore()
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

## Preparing Documents

DocumentStores expect Documents in dictionary form, like that below.
They are loaded using the `DocumentStore.write_documents()` method.

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

## File Conversion

There are a range of different file converters in Haystack that can help get your data into the right format.
Haystack features support for txt, pdf and docx formats and there is even a converted that leverages Apache Tika.
See the File Converters section in the API docs for more information.

<!-- _comment: !! Code snippets for each type !! -->
Haystack also has a `convert_files_to_dicts()` utility function that will convert
all txt or pdf files in a given folder into this dictionary format.

```python
document_store = ElasticsearchDocumentStore()
dicts = convert_files_to_dicts(dir_path=doc_dir)
document_store.write_documents(dicts)
```

## Writing Documents

Haystack allows for you to write store documents in an optimised fashion so that query times can be kept low.

### For Sparse Retrievers

For **sparse**, keyword based retrievers such as BM25 and TF-IDF,
you simply have to call `DocumentStore.write_documents()`.
The creation of the inverted index which optimises querying speed is handled automatically.

```python
document_store.write_documents(dicts)
```

### For Dense Retrievers

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
## Choosing the right document store

The Document stores have different characteristics. You should choose one depending on the maturity of your project, the use case and technical environment: 

<div class="tabs tabsdschoose">

<div class="tab">
<input type="radio" id="tab-2-1" name="tab-group-2" checked>
<label class="labelouter" for="tab-2-1">Elasticsearch</label>
<div class="tabcontent">

**Pros:** 
- Fast & accurate sparse retrieval
- Basic support for dense retrieval
- Production-ready 
- Many options to tune sparse retrieval

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
- Many options to tune dense retrieval via different index types 

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

**Cons:** 
- Not scalable
- Not persisting your data on disk

</div>
</div>

</div>

#### Our recommendations

**Restricted environment:** Use the `InMemoryDocumentStore`, if you are just giving Haystack a quick try on a small sample and are working in a restricted environment that complicates running Elasticsearch or other databases  

**Allrounder:** Use the `ElasticSearchDocumentStore`, if you want to evaluate the performance of different retrieval options (dense vs. sparse) and are aiming for a smooth transition from PoC to production

**Vector Specialist:** Use the `FAISSDocumentStore`, if you want to focus on dense retrieval and possibly deal with larger datasets
