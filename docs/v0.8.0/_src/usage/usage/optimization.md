<!---
title: "Optimization"
metaTitle: "Optimization"
metaDescription: ""
slug: "/docs/optimization"
date: "2020-11-05"
id: "optimizationmd"
--->

# Optimization

## Document Length

Document length has a very direct impact on the speed of the Reader 
which is why we recommend using the `PreProcessor` class to clean and split your documents.
**If you halve the length of your documents, you will halve the workload placed onto your Reader.**

For **sparse retrievers**, very long documents pose a challenge since the signal of the relevant section of text
can get washed out by the rest of the document.
We would recommend making sure that **documents are no longer than 10,000 words**.

**Dense retrievers** are limited in the length of text that they can read in one pass.
As such, it is important that documents are not longer than the dense retriever's maximum input length.
By default, Haystack's DensePassageRetriever model has a maximum length of 256 tokens.
As such, we recommend that documents contain significantly less words.
We have found decent performance with **documents around 100 words long**.

## Respecting Sentence Boundaries

When splitting documents, it is generally not a good idea to let document boundaries fall in the middle of sentences. 
Doing so means that each document will contain incomplete sentence fragments 
which maybe be hard for both retriever and reader to interpret.
It is therefore recommended to set `split_respect_sentence_boundary=True` when initializing your `PreProcessor`.

## Choosing the Right top-k Values

The `top-k` parameter in both the `Retriever` and `Reader` determine how many results they return.
More specifically, `Retriever` `top-k` dictates how many retrieved documents are passed on to the next stage,
while `Reader` `top-k` determines how many answer candidates to show.

In our experiments, we have found that **`Retriever` `top_k=10`
gives decent overall performance** and so we have set this as the default in Haystack.

The choice of `Retriever` `top-k` is a trade-off between speed and accuracy, 
especially when there is a `Reader` in the pipeline.
Setting it higher means passing more documents to the `Reader`, 
thus reducing the chance that the answer-containing passage is missed.
However, passing more documents to the `Reader` will create a larger workload for the component.

These parameters can easily be tweaked as follows if using a `Finder`:
``` python
answers = finder.get_answers(retriever_top_k=10,
                             reader_top_k=5)
```
or like this if directly calling the `Retriever`:
``` python
retrieved_docs = retriever.retrieve(top_k=10)
```

<div class="recommendation">

**Tip:** The Finder class is being deprecated and has been replaced by a more powerful [Pipelines class](/docs/latest/pipelinesmd).

</div>
