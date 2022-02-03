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

