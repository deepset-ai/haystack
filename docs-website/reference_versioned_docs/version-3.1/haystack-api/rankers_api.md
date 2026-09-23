---
title: "Rankers"
id: rankers-api
description: "Reorders a set of Documents based on their relevance to the query."
slug: "/rankers-api"
---


## llm_ranker

### LLMRanker

Ranks documents for a query using a Large Language Model.

The LLM is expected to return a JSON object containing ranked document indices.

Usage example:

```python
from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.rankers import LLMRanker

chat_generator = OpenAIChatGenerator(
    model="gpt-4.1-mini",
    generation_kwargs={
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "document_ranking",
                "schema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"index": {"type": "integer"}},
                                "required": ["index"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["documents"],
                    "additionalProperties": False,
                },
            },
        },
    },
)

ranker = LLMRanker(chat_generator=chat_generator)

documents = [
    Document(id="paris", content="Paris is the capital of France."),
    Document(id="berlin", content="Berlin is the capital of Germany."),
]

result = ranker.run(query="capital of Germany", documents=documents)
print(result["documents"][0].id)
```

#### __init__

```python
__init__(
    *,
    chat_generator: ChatGenerator | None = None,
    prompt: str = DEFAULT_PROMPT_TEMPLATE,
    top_k: int = 10,
    raise_on_failure: bool = False
) -> None
```

Initialize the LLMRanker component.

**Parameters:**

- **chat_generator** (<code>ChatGenerator | None</code>) – The chat generator to use for reranking. If `None`, a default `OpenAIChatGenerator` configured for JSON
  output is used.
- **prompt** (<code>str</code>) – Custom prompt template for reranking. The prompt must include exactly the variables `query` and
  `documents` and instruct the LLM to return ranked 1-based document indices as JSON.
- **top_k** (<code>int</code>) – The maximum number of documents to return.
- **raise_on_failure** (<code>bool</code>) – If `True`, raise when generation or response parsing fails. If `False`, log the failure and return the
  input documents in fallback order.

#### warm_up

```python
warm_up() -> None
```

Warm up the underlying chat generator.

#### warm_up_async

```python
warm_up_async() -> None
```

Warm up the underlying chat generator on the serving event loop.

#### close

```python
close() -> None
```

Release the underlying chat generator's resources.

#### close_async

```python
close_async() -> None
```

Release the underlying chat generator's async resources.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> LLMRanker
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of the component.

**Returns:**

- <code>LLMRanker</code> – The deserialized component instance.

#### run

```python
run(
    query: str, documents: list[Document], top_k: int | None = None
) -> dict[str, list[Document]]
```

Rank documents for a query using an LLM.

Before ranking, duplicate documents are removed.

**Parameters:**

- **query** (<code>str</code>) – The query used for reranking.
- **documents** (<code>list\[Document\]</code>) – Candidate documents to rerank.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. Overrides the instance's `top_k` if provided.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the ranked documents under the `documents` key.

#### run_async

```python
run_async(
    query: str, documents: list[Document], top_k: int | None = None
) -> dict[str, list[Document]]
```

Asynchronously rank documents for a query using an LLM.

Before ranking, duplicate documents are removed.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in an async code. If the chat generator only implements a synchronous
`run` method, it is executed in a thread to avoid blocking the event loop.

**Parameters:**

- **query** (<code>str</code>) – The query used for reranking.
- **documents** (<code>list\[Document\]</code>) – Candidate documents to rerank.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. Overrides the instance's `top_k` if provided.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the ranked documents under the `documents` key.

## lost_in_the_middle

### LostInTheMiddleRanker

A LostInTheMiddle Ranker.

Ranks documents based on the 'lost in the middle' order so that the most relevant documents are either at the
beginning or end, while the least relevant are in the middle.

LostInTheMiddleRanker assumes that some prior component in the pipeline has already ranked documents by relevance
and requires no query as input but only documents. It is typically used as the last component before building a
prompt for an LLM to prepare the input context for the LLM.

Lost in the Middle ranking lays out document contents into LLM context so that the most relevant contents are at
the beginning or end of the input context, while the least relevant is in the middle of the context. See the
paper ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) for more
details.

Usage example:

```python
from haystack.components.rankers import LostInTheMiddleRanker
from haystack import Document

ranker = LostInTheMiddleRanker()
docs = [Document(content="Paris"), Document(content="Berlin"), Document(content="Madrid")]
result = ranker.run(documents=docs)
for doc in result["documents"]:
    print(doc.content)
```

#### __init__

```python
__init__(
    word_count_threshold: int | None = None, top_k: int | None = None
) -> None
```

Initialize the LostInTheMiddleRanker.

If 'word_count_threshold' is specified, this ranker includes all documents up until the point where adding
another document would exceed the 'word_count_threshold'. The last document that causes the threshold to
be breached will be included in the resulting list of documents, but all subsequent documents will be
discarded.

**Parameters:**

- **word_count_threshold** (<code>int | None</code>) – The maximum total number of words across all documents selected by the ranker.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.

#### run

```python
run(
    documents: list[Document],
    top_k: int | None = None,
    word_count_threshold: int | None = None,
) -> dict[str, list[Document]]
```

Reranks documents based on the "lost in the middle" order.

Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
if a score is present.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to reorder.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **word_count_threshold** (<code>int | None</code>) – The maximum total number of words across all documents selected by the ranker.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Reranked list of Documents

**Raises:**

- <code>ValueError</code> – If any of the documents is not textual.

## meta_field

### MetaFieldRanker

Ranks Documents based on the value of their specific meta field.

The ranking can be performed in descending order or ascending order.

Usage example:

```python
from haystack import Document
from haystack.components.rankers import MetaFieldRanker

ranker = MetaFieldRanker(meta_field="rating")
docs = [
    Document(content="Paris", meta={"rating": 1.3}),
    Document(content="Berlin", meta={"rating": 0.7}),
    Document(content="Barcelona", meta={"rating": 2.1}),
]

output = ranker.run(documents=docs)
docs = output["documents"]
assert docs[0].content == "Barcelona"
```

#### __init__

```python
__init__(
    meta_field: str,
    weight: float = 1.0,
    top_k: int | None = None,
    ranking_mode: Literal[
        "reciprocal_rank_fusion", "linear_score"
    ] = "reciprocal_rank_fusion",
    sort_order: Literal["ascending", "descending"] = "descending",
    missing_meta: Literal["drop", "top", "bottom"] = "bottom",
    meta_value_type: Literal["float", "int", "date"] | None = None,
) -> None
```

Creates an instance of MetaFieldRanker.

**Parameters:**

- **meta_field** (<code>str</code>) – The name of the meta field to rank by.
- **weight** (<code>float</code>) – In range [0,1].
  0 disables ranking by a meta field.
  0.5 ranking from previous component and based on meta field have the same weight.
  1 ranking by a meta field only.
- **top_k** (<code>int | None</code>) – The maximum number of Documents to return per query.
  If not provided, the Ranker returns all documents it receives in the new ranking order.
- **ranking_mode** (<code>Literal['reciprocal_rank_fusion', 'linear_score']</code>) – The mode used to combine the Retriever's and Ranker's scores.
  Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
  Use the 'linear_score' mode only with Retrievers or Rankers that return a score in range [0,1].
- **sort_order** (<code>Literal['ascending', 'descending']</code>) – Whether to sort the meta field by ascending or descending order.
  Possible values are `descending` (default) and `ascending`.
- **missing_meta** (<code>Literal['drop', 'top', 'bottom']</code>) – What to do with documents that are missing the sorting metadata field.
  Possible values are:
  - 'drop' will drop the documents entirely.
  - 'top' will place the documents at the top of the metadata-sorted list
    (regardless of 'ascending' or 'descending').
  - 'bottom' will place the documents at the bottom of metadata-sorted list
    (regardless of 'ascending' or 'descending').
- **meta_value_type** (<code>Literal['float', 'int', 'date'] | None</code>) – Parse the meta value into the data type specified before sorting.
  This will only work if all meta values stored under `meta_field` in the provided documents are strings.
  For example, if we specified `meta_value_type="date"` then for the meta value `"date": "2015-02-01"`
  we would parse the string into a datetime object and then sort the documents by date.
  The available options are:
- 'float' will parse the meta values into floats.
- 'int' will parse the meta values into integers.
- 'date' will parse the meta values into datetime objects.
- 'None' (default) will do no parsing.

#### run

```python
run(
    documents: list[Document],
    top_k: int | None = None,
    weight: float | None = None,
    ranking_mode: (
        Literal["reciprocal_rank_fusion", "linear_score"] | None
    ) = None,
    sort_order: Literal["ascending", "descending"] | None = None,
    missing_meta: Literal["drop", "top", "bottom"] | None = None,
    meta_value_type: Literal["float", "int", "date"] | None = None,
) -> dict[str, Any]
```

Ranks a list of Documents based on the selected meta field by:

1. Sorting the Documents by the meta field in descending or ascending order.
1. Merging the rankings from the previous component and based on the meta field according to ranking mode and
   weight.
1. Returning the top-k documents.

Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
if a score is present.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to be ranked.
- **top_k** (<code>int | None</code>) – The maximum number of Documents to return per query.
  If not provided, the top_k provided at initialization time is used.
- **weight** (<code>float | None</code>) – In range [0,1].
  0 disables ranking by a meta field.
  0.5 ranking from previous component and based on meta field have the same weight.
  1 ranking by a meta field only.
  If not provided, the weight provided at initialization time is used.
- **ranking_mode** (<code>Literal['reciprocal_rank_fusion', 'linear_score'] | None</code>) – (optional) The mode used to combine the Retriever's and Ranker's scores.
  Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
  Use the 'score' mode only with Retrievers or Rankers that return a score in range [0,1].
  If not provided, the ranking_mode provided at initialization time is used.
- **sort_order** (<code>Literal['ascending', 'descending'] | None</code>) – Whether to sort the meta field by ascending or descending order.
  Possible values are `descending` (default) and `ascending`.
  If not provided, the sort_order provided at initialization time is used.
- **missing_meta** (<code>Literal['drop', 'top', 'bottom'] | None</code>) – What to do with documents that are missing the sorting metadata field.
  Possible values are:
- 'drop' will drop the documents entirely.
- 'top' will place the documents at the top of the metadata-sorted list
  (regardless of 'ascending' or 'descending').
- 'bottom' will place the documents at the bottom of metadata-sorted list
  (regardless of 'ascending' or 'descending').
  If not provided, the missing_meta provided at initialization time is used.
- **meta_value_type** (<code>Literal['float', 'int', 'date'] | None</code>) – Parse the meta value into the data type specified before sorting.
  This will only work if all meta values stored under `meta_field` in the provided documents are strings.
  For example, if we specified `meta_value_type="date"` then for the meta value `"date": "2015-02-01"`
  we would parse the string into a datetime object and then sort the documents by date.
  The available options are:
  -'float' will parse the meta values into floats.
  -'int' will parse the meta values into integers.
  -'date' will parse the meta values into datetime objects.
  -'None' (default) will do no parsing.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: List of Documents sorted by the specified meta field.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.
  If `weight` is not in range [0,1].
  If `ranking_mode` is not 'reciprocal_rank_fusion' or 'linear_score'.
  If `sort_order` is not 'ascending' or 'descending'.
  If `meta_value_type` is not 'float', 'int', 'date' or `None`.

## meta_field_grouping_ranker

### MetaFieldGroupingRanker

Reorders the documents by grouping them based on metadata keys.

The MetaFieldGroupingRanker can group documents by a primary metadata key `group_by`, and subgroup them with an optional
secondary key, `subgroup_by`.
Within each group or subgroup, it can also sort documents by a metadata key `sort_docs_by`.

The output is a flat list of documents ordered by `group_by` and `subgroup_by` values.
Any documents without a group are placed at the end of the list.

The proper organization of documents helps improve the efficiency and performance of subsequent processing by an LLM.

### Usage example

```python
from haystack.components.rankers import MetaFieldGroupingRanker
from haystack.dataclasses import Document


docs = [
    Document(content="Javascript is a popular programming language", meta={"group": "42", "split_id": 7, "subgroup": "subB"}),
    Document(content="Python is a popular programming language",meta={"group": "42", "split_id": 4, "subgroup": "subB"}),
    Document(content="A chromosome is a package of DNA", meta={"group": "314", "split_id": 2, "subgroup": "subC"}),
    Document(content="An octopus has three hearts", meta={"group": "11", "split_id": 2, "subgroup": "subD"}),
    Document(content="Java is a popular programming language", meta={"group": "42", "split_id": 3, "subgroup": "subB"})
]

ranker = MetaFieldGroupingRanker(group_by="group",subgroup_by="subgroup", sort_docs_by="split_id")
result = ranker.run(documents=docs)
print(result["documents"])

# >> 
# >>  Document(id=d665bbc83e52c08c3d8275bccf4f22bf2bfee21c6e77d78794627637355b8ebc,
# >>          content: 'Java is a popular programming language', meta: {'group': '42', 'split_id': 3, 'subgroup': 'subB'}),
# >>  Document(id=a20b326f07382b3cbf2ce156092f7c93e8788df5d48f2986957dce2adb5fe3c2,
# >>          content: 'Python is a popular programming language', meta: {'group': '42', 'split_id': 4, 'subgroup': 'subB'}),
# >>  Document(id=ce12919795d22f6ca214d0f161cf870993889dcb146f3bb1b3e1ffdc95be960f,
# >>          content: 'Javascript is a popular programming language', meta: {'group': '42', 'split_id': 7, 'subgroup': 'subB'}),
# >>  Document(id=d9fc857046c904e5cf790b3969b971b1bbdb1b3037d50a20728fdbf82991aa94,
# >>          content: 'A chromosome is a package of DNA', meta: {'group': '314', 'split_id': 2, 'subgroup': 'subC'}),
# >>  Document(id=6d3b7bdc13d09aa01216471eb5fb0bfdc53c5f2f3e98ad125ff6b85d3106c9a3,
# >>          content: 'An octopus has three hearts', meta: {'group': '11', 'split_id': 2, 'subgroup': 'subD'})
```

#### __init__

```python
__init__(
    group_by: str,
    subgroup_by: str | None = None,
    sort_docs_by: str | None = None,
) -> None
```

Creates an instance of MetaFieldGroupingRanker.

**Parameters:**

- **group_by** (<code>[str</code>) – The metadata key to aggregate the documents by.
- **subgroup_by** (<code>str | None</code>) – The metadata key to aggregate the documents within a group that was created by the
  `group_by` key.
- **sort_docs_by** (<code>str | None</code>) – Determines which metadata key is used to sort the documents. If not provided, the
  documents within the groups or subgroups are not sorted and are kept in the same order as
  they were inserted in the subgroups.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Groups the provided list of documents based on the `group_by` parameter and optionally the `subgroup_by`.

Before grouping, documents are deduplicated by their id, retaining only the document with the highest score
if a score is present.

The output is a list of documents reordered based on how they were grouped.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to group.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- documents: The list of documents ordered by the `group_by` and `subgroup_by` metadata values.
