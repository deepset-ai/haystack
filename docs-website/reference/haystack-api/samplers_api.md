---
title: "Samplers"
id: samplers-api
description: "Filters documents based on their similarity scores using top-p sampling."
slug: "/samplers-api"
---

<a id="top_p"></a>

# Module top\_p

<a id="top_p.TopPSampler"></a>

## TopPSampler

Implements top-p (nucleus) sampling for document filtering based on cumulative probability scores.

This component provides functionality to filter a list of documents by selecting those whose scores fall
within the top 'p' percent of the cumulative distribution. It is useful for focusing on high-probability
documents while filtering out less relevant ones based on their assigned scores.

Usage example:

```python
from haystack import Document
from haystack.components.samplers import TopPSampler

sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
docs = [
    Document(content="Berlin", meta={"similarity_score": -10.6}),
    Document(content="Belgrade", meta={"similarity_score": -8.9}),
    Document(content="Sarajevo", meta={"similarity_score": -4.6}),
]
output = sampler.run(documents=docs)
docs = output["documents"]
assert len(docs) == 1
assert docs[0].content == "Sarajevo"
```

<a id="top_p.TopPSampler.__init__"></a>

#### TopPSampler.\_\_init\_\_

```python
def __init__(top_p: float = 1.0,
             score_field: Optional[str] = None,
             min_top_k: Optional[int] = None)
```

Creates an instance of TopPSampler.

**Arguments**:

- `top_p`: Float between 0 and 1 representing the cumulative probability threshold for document selection.
A value of 1.0 indicates no filtering (all documents are retained).
- `score_field`: Name of the field in each document's metadata that contains the score. If None, the default
document score field is used.
- `min_top_k`: If specified, the minimum number of documents to return. If the top_p selects
fewer documents, additional ones with the next highest scores are added to the selection.

<a id="top_p.TopPSampler.run"></a>

#### TopPSampler.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document], top_p: Optional[float] = None)
```

Filters documents using top-p sampling based on their scores.

If the specified top_p results in no documents being selected (especially in cases of a low top_p value), the
method returns the document with the highest score.

**Arguments**:

- `documents`: List of Document objects to be filtered.
- `top_p`: If specified, a float to override the cumulative probability threshold set during initialization.

**Raises**:

- `ValueError`: If the top_p value is not within the range [0, 1].

**Returns**:

A dictionary with the following key:
- `documents`: List of Document objects that have been selected based on the top-p sampling.

