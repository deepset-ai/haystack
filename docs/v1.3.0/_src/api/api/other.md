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

<a id="join_answers"></a>

# Module join\_answers

<a id="join_answers.JoinAnswers"></a>

## JoinAnswers

```python
class JoinAnswers(BaseComponent)
```

A node to join `Answer`s produced by multiple `Reader` nodes.

<a id="join_answers.JoinAnswers.__init__"></a>

#### \_\_init\_\_

```python
def __init__(join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None)
```

**Arguments**:

- `join_mode`: `"concatenate"` to combine documents from multiple `Reader`s. `"merge"` to aggregate scores
of individual `Answer`s.
- `weights`: A node-wise list (length of list must be equal to the number of input nodes) of weights for
adjusting `Answer` scores when using the `"merge"` join_mode. By default, equal weight is assigned to each
`Reader` score. This parameter is not compatible with the `"concatenate"` join_mode.
- `top_k_join`: Limit `Answer`s to top_k based on the resulting scored of the join.

<a id="route_documents"></a>

# Module route\_documents

<a id="route_documents.RouteDocuments"></a>

## RouteDocuments

```python
class RouteDocuments(BaseComponent)
```

A node to split a list of `Document`s by `content_type` or by the values of a metadata field and route them to
different nodes.

<a id="route_documents.RouteDocuments.__init__"></a>

#### \_\_init\_\_

```python
def __init__(split_by: str = "content_type", metadata_values: Optional[List[str]] = None)
```

**Arguments**:

- `split_by`: Field to split the documents by, either `"content_type"` or a metadata field name.
If this parameter is set to `"content_type"`, the list of `Document`s will be split into a list containing
only `Document`s of type `"text"` (will be routed to `"output_1"`) and a list containing only `Document`s of
type `"text"` (will be routed to `"output_2"`).
If this parameter is set to a metadata field name, you need to specify the parameter `metadata_values` as
well.
- `metadata_values`: If the parameter `split_by` is set to a metadata field name, you need to provide a list
of values to group the `Document`s to. `Document`s whose metadata field is equal to the first value of the
provided list will be routed to `"output_1"`, `Document`s whose metadata field is equal to the second
value of the provided list will be routed to `"output_2"`, etc.
