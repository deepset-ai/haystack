<!---
title: "Summarizer"
metaTitle: "Summarizer"
metaDescription: "Using the Summarizer Node"
slug: "/docs/summarizer"
date: "2021-04-16"
id: "summarizermd"
--->

# Summarizer

Retrievers are excellent at returning a set of candidate documents,
but you might not have the time to read through them all.
Haystack's Summmarizer is here to help you make sense of the documents at a glance.

There is a full integration with Huggingface Transformers and using any of their summarization
models is as simple as providing the model name.
See the up-to-date list of available models [here](https://huggingface.co/models?filter=summarization).
By default, the Google [Pegasus](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html) model is loaded.

```python
from haystack.summarizer import TransformersSummarizer
from haystack.schema import Document

docs = [Document("PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions.\
                 The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by\
                 the shutoffs which were expected to last through at least midday tomorrow.")]

summarizer = TransformersSummarizer(model_name_or_path="google/pegasus-xsum")
summary = summarizer.predict(documents=docs, generate_single_summary=True)
```

The contents of summary should contain both the summarization and also the original document text.

```python
[
    {
        "text": "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
        "meta": {
            "context": "PGE stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions."
        },
        ...
    }
]
```

The summarizer can also functions as a node in a pipeline.

```python
from haystack.pipeline import Pipeline

p = Pipeline()
p.add_node(component=retriever, name="ESRetriever1", inputs=["Query"])
p.add_node(component=summarizer, name="Summarizer", inputs=["ESRetriever1"])
res = p.run(query="What did Einstein work on?", top_k_retriever=10)
``` 
