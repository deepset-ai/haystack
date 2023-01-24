<a id="docs2answers"></a>

# Module docs2answers

<a id="docs2answers.Docs2Answers"></a>

## Docs2Answers

```python
class Docs2Answers(BaseComponent)
```

This Node is used to convert retrieved documents into predicted answers format.
It is useful for situations where you are calling a Retriever only pipeline via REST API.
This ensures that your output is in a compatible format.

<a id="join_docs"></a>

# Module join\_docs

<a id="join_docs.JoinDocuments"></a>

## JoinDocuments

```python
class JoinDocuments(BaseComponent)
```

A node to join documents outputted by multiple retriever nodes.

The node allows multiple join modes:
* concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
* merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
         `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
* reciprocal_rank_fusion: combines the documents based on their rank in multiple nodes.

<a id="join_docs.JoinDocuments.__init__"></a>

#### \_\_init\_\_

```python
def __init__(join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None)
```

**Arguments**:

- `join_mode`: `concatenate` to combine documents from multiple retrievers `merge` to aggregate scores of
individual documents, `reciprocal_rank_fusion` to apply rank based scoring.
- `weights`: A node-wise list(length of list must be equal to the number of input nodes) of weights for
adjusting document scores when using the `merge` join_mode. By default, equal weight is given
to each retriever score. This param is not compatible with the `concatenate` join_mode.
- `top_k_join`: Limit documents to top_k based on the resulting scores of the join.
