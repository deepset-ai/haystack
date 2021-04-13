<!---
title: "Get Started"
metaTitle: "Get Started"
metaDescription: ""
slug: "/docs/get_started"
date: "2020-09-03"
id: "get_startedmd"
--->

# Get Started

## Installation

<div class="tabs tabsgetstarted">

<div class="tab">
<input type="radio" id="tab-1" name="tab-group-1" checked>
<label class="labelouter" for="tab-1">Basic</label>
<div class="tabcontent">
The most straightforward way to install Haystack is through pip.<br/><br/>

```python
$ pip install farm-haystack
```

</div> 
</div>

<div class="tab">
<input type="radio" id="tab-2" name="tab-group-1">
<label class="labelouter" for="tab-2">Editable</label>
<div class="tabcontent">
If you’d like to run a specific, unreleased version of Haystack, or make edits to the way Haystack runs,
you’ll want to install it using `git` and `pip --editable`.
This clones a copy of the repo to a local directory and runs Haystack from there. <br/><br/>

```python
$ git clone https://github.com/deepset-ai/haystack.git
$ cd haystack
$ pip install --editable .
```

By default, this will give you the latest version of the master branch. Use regular git commands to switch between different branches and commits.
</div> 
</div>

</div>

Note: On Windows add the arg `-f https://download.pytorch.org/whl/torch_stable.html` to install PyTorch correctly

<!-- _comment: !! Have a tab for docker!! -->
<!-- -comment: !! Have a hello world example!! -->
## The Building Blocks of Haystack

Here’s a sample of some Haystack code showing the most important components.
For a working code example, check out our starter tutorial.

<!-- _comment: !!link!! -->
```python
# DocumentStore: holds all your data
document_store = ElasticsearchDocumentStore()

# Clean & load your documents into the DocumentStore
dicts = convert_files_to_dicts(doc_dir, clean_func=clean_wiki_text)
document_store.write_documents(dicts)

# Retriever: A Fast and simple algo to indentify the most promising candidate documents
retriever = ElasticsearchRetriever(document_store)

# Reader: Powerful but slower neural network trained for QA
model_name = "deepset/roberta-base-squad2"
reader = FARMReader(model_name)

# Finder: Combines Reader and Retriever
finder = Finder(reader, retriever)

# Voilà! Ask a question!
question = "Who is the father of Sansa Stark?"
prediction = finder.get_answers(question)
print_answers(prediction)
```

## Loading Documents into the DocumentStore

In Haystack, DocumentStores expect Documents in a dictionary format. They are loaded as follows:

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

When we talk about Documents in Haystack, we are referring specifically to the individual blocks of text that are being held in the DocumentStore.
You might want to use all the text in one file as a Document, or split it into multiple Documents.
This splitting can have a big impact on speed and performance.

<div class="recommendation">

**Tip:** If Haystack is running very slowly, you might want to try splitting your text into smaller Documents.
If you want an improvement to performance, you might want to try concatenating text to make larger Documents.

</div>

## Running Queries

**Querying** involves searching for an answer to a given question within the full document store.
This process will:


* make the Retriever filter for a small set of relevant candidate documents


* get the Reader to process this set of candidate documents


* return potential answers to the given question

Usually, there are tight time constraints on querying and so it needs to be a lightweight operation.
When documents are loaded, Haystack will precompute any of the results that might be useful at query time.

In Haystack, querying is performed with a `Finder` object which connects the reader to the retriever.

```python
# The Finder sticks together reader and retriever in a pipeline to answer our questions.
finder = Finder(reader, retriever)

# Voilà! Ask a question!
question = "Who is the father of Sansa Stark?"
prediction = finder.get_answers(question)
```

When the query is complete, you can expect to see results that look something like this:

```python
[
    {   'answer': 'Eddard',
        'context': 's Nymeria after a legendary warrior queen. She travels '
                   "with her father, Eddard, to King's Landing when he is made "
                   'Hand of the King. Before she leaves,'
    }, ...
]
```

<div class="recommendation">

**Tip:** The Finder class is being deprecated and has been replaced by a more powerful [Pipelines class](/docs/latest/pipelinesmd).

</div>
