---
title: "pyversity"
id: integrations-pyversity
description: "pyversity integration for Haystack"
slug: "/integrations-pyversity"
---


## haystack_integrations.components.rankers.pyversity.ranker

Haystack integration for `pyversity <https://github.com/Pringled/pyversity>`\_.

Wraps pyversity's diversification algorithms as a Haystack `@component`,
making it easy to drop result diversification into any Haystack pipeline.

### PyversityRanker

Reranks documents using [pyversity](https://github.com/Pringled/pyversity)'s diversification algorithms.

Balances relevance and diversity in a ranked list of documents. Documents
must have both `score` and `embedding` populated (e.g. as returned by
a dense retriever with `return_embedding=True`).

Usage example:

```python
from haystack import Document
from haystack_integrations.components.rankers.pyversity import PyversityRanker
from pyversity import Strategy

ranker = PyversityRanker(top_k=5, strategy=Strategy.MMR, diversity=0.5)

docs = [
    Document(content="Paris", score=0.9, embedding=[0.1, 0.2]),
    Document(content="Berlin", score=0.8, embedding=[0.3, 0.4]),
]
output = ranker.run(documents=docs)
docs = output["documents"]
```

#### __init__

```python
__init__(
    top_k: int | None = None,
    *,
    strategy: Strategy = Strategy.DPP,
    diversity: float = 0.5
) -> None
```

Creates an instance of PyversityRanker.

**Parameters:**

- **top_k** (<code>int | None</code>) – Number of documents to return after diversification.
  If `None`, all documents are returned in diversified order.
- **strategy** (<code>Strategy</code>) – Pyversity diversification strategy (e.g. `Strategy.MMR`). Defaults to `Strategy.DPP`.
- **diversity** (<code>float</code>) – Trade-off between relevance and diversity in [0, 1].
  `0.0` keeps only the most relevant documents; `1.0` maximises
  diversity regardless of relevance. Defaults to `0.5`.

**Raises:**

- <code>ValueError</code> – If `top_k` is not a positive integer or `diversity` is not in [0, 1].

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> PyversityRanker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>PyversityRanker</code> – The deserialized component instance.

#### run

```python
run(
    documents: list[Document],
    top_k: int | None = None,
    strategy: Strategy | None = None,
    diversity: float | None = None,
) -> dict[str, list[Document]]
```

Rerank the list of documents using pyversity's diversification algorithm.

Documents missing `score` or `embedding` are skipped with a warning.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to rerank. Each document must have `score` and `embedding` set.
- **top_k** (<code>int | None</code>) – Overrides the initialized `top_k` for this call. `None` falls back to the initialized value.
- **strategy** (<code>Strategy | None</code>) – Overrides the initialized `strategy` for this call. `None` falls back to the initialized value.
- **diversity** (<code>float | None</code>) – Overrides the initialized `diversity` for this call.
  `None` falls back to the initialized value.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of up to `top_k` reranked Documents, ordered by the diversification algorithm.

**Raises:**

- <code>ValueError</code> – If `top_k` is not a positive integer or `diversity` is not in [0, 1].
