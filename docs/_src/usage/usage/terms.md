<!---
title: "Glossary"
metaTitle: "Glossary"
metaDescription: ""
slug: "/docs/terms"
date: "2020-09-03"
id: "termsmd"
--->

# Glossary

**BERT** - A popular, transformer based language model which has been improved upon but is still considered a common benchmark.

**Dense** - Vectors that contain many non-zero values are considered dense.
Retrieval methods can also be called dense if they create dense vector representations of documents.

**Document** - A Document in Haystack refers to the individual pieces of text that are stored in the DocumentStore.
Multiple Documents might originally come from the one file.
It is ultimately up to you how to divide up your corpus into Documents.

**Document Store** - The component in Haystack that stores the text documents and their metadata.
Can have a variety of backends such as Elasticsearch, SQL or FAISS.

**FARM** - An open-source transfer learning [framework](https://github.com/deepset-ai/FARM) by deepset.
FARM’s question answering models are used in Haystack’s Readers.

**Indexing** - To store data in a database in a way that optimises retrieval time.
The exact steps involved in indexing depend on what kind of retrieval method is chosen.

**Language Model** - The component in an NLP model that stores general language understanding, but no task specific knowledge.

**Model Hub** - The [repository](https://huggingface.co/models) set up by HuggingFace where trained models can be saved to and loaded from.
With Haystack, you can directly load and use any question answering model found on the model hub.

**Neural Network** - A machine learning architecture composed of artificial neurons that learn a task when exposed to labelled training data.

**Prediction Head** - The modelling component that adapts the general knowledge of the language model for a specific task.
In question answering models (and hence in Haystack Readers), this is usually a single layer neural network.

**Querying** - The task of returning relevant documents from a database.

**Question Answering (QA)** - A popular task in the world of NLP where systems have to find answers to questions.
The term is generally used to refer to extractive question answering,
where a system has to find the minimal text span in a given document that contains the answer to the question.
Note however, that it may also refer to abstractive question answering or FAQ matching.

**Reader** - The component in Haystack that does the closest reading of a document to extract
the exact text which answers a question.
It is, at its core, a trained Question Answering model.

**Retriever** - A lightweight filter that selects only the most relevant documents for the Reader to further process.

**Semantic Search** - A style of search that relies not on the matching of exact string forms
but on the similarity of meaning between a query and a piece of text.

**Sparse** - Vectors that are composed primarily of zeros are called sparse.
Retrieval methods are also considered sparse if they build sparse vector representations of documents.

**SQuAD** - The [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) is the defacto standard QA dataset.
The documents are paragraphs from Wikipedia and the question / answer pairs are created by human annotators.

**Transformers** - Originally refers to the deep learning architecture that is composed of stacked self-attention layers
(first conceptualised [here](https://arxiv.org/pdf/1706.03762.pdf)).
Can also refer to HuggingFace’s [repository](https://github.com/huggingface/transformers)
which contains implementations of popular model architectures.
