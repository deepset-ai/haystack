---
title: "Query"
id: query-api
description: "Components for query processing and expansion."
slug: "/query-api"
---

<a id="query_expander"></a>

## Module query\_expander

<a id="query_expander.QueryExpander"></a>

### QueryExpander

A component that returns a list of semantically similar queries to improve retrieval recall in RAG systems.

The component uses a chat generator to expand queries. The chat generator is expected to return a JSON response
with the following structure:

### Usage example

```json
{"queries": ["expanded query 1", "expanded query 2", "expanded query 3"]}
```
```python
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.query import QueryExpander

expander = QueryExpander(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
    n_expansions=3
)

result = expander.run(query="green energy sources")
print(result["queries"])
# Output: ['alternative query 1', 'alternative query 2', 'alternative query 3', 'green energy sources']
# Note: Up to 3 additional queries + 1 original query (if include_original_query=True)

# To control total number of queries:
expander = QueryExpander(n_expansions=2, include_original_query=True)  # Up to 3 total
# or
expander = QueryExpander(n_expansions=3, include_original_query=False)  # Exactly 3 total
```

<a id="query_expander.QueryExpander.__init__"></a>

#### QueryExpander.\_\_init\_\_

```python
def __init__(*,
             chat_generator: Optional[ChatGenerator] = None,
             prompt_template: Optional[str] = None,
             n_expansions: int = 4,
             include_original_query: bool = True) -> None
```

Initialize the QueryExpander component.

**Arguments**:

- `chat_generator`: The chat generator component to use for query expansion.
If None, a default OpenAIChatGenerator with gpt-4.1-mini model is used.
- `prompt_template`: Custom [PromptBuilder](https://docs.haystack.deepset.ai/docs/promptbuilder)
template for query expansion. The template should instruct the LLM to return a JSON response with the
structure: `{"queries": ["query1", "query2", "query3"]}`. The template should include 'query' and
'n_expansions' variables.
- `n_expansions`: Number of alternative queries to generate (default: 4).
- `include_original_query`: Whether to include the original query in the output.

<a id="query_expander.QueryExpander.to_dict"></a>

#### QueryExpander.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="query_expander.QueryExpander.from_dict"></a>

#### QueryExpander.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "QueryExpander"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary with serialized data.

**Returns**:

Deserialized component.

<a id="query_expander.QueryExpander.run"></a>

#### QueryExpander.run

```python
@component.output_types(queries=list[str])
def run(query: str,
        n_expansions: Optional[int] = None) -> dict[str, list[str]]
```

Expand the input query into multiple semantically similar queries.

The language of the original query is preserved in the expanded queries.

**Arguments**:

- `query`: The original query to expand.
- `n_expansions`: Number of additional queries to generate (not including the original).
If None, uses the value from initialization. Can be 0 to generate no additional queries.

**Raises**:

- `ValueError`: If n_expansions is not positive (less than or equal to 0).

**Returns**:

Dictionary with "queries" key containing the list of expanded queries.
If include_original_query=True, the original query will be included in addition
to the n_expansions alternative queries.

<a id="query_expander.QueryExpander.warm_up"></a>

#### QueryExpander.warm\_up

```python
def warm_up()
```

Warm up the LLM provider component.

