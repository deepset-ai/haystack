---
title: "Query"
id: query-api
description: "Components for query processing and expansion."
slug: "/query-api"
---


## `haystack.components.query.query_expander`

### `QueryExpander`

A component that returns a list of semantically similar queries to improve retrieval recall in RAG systems.

The component uses a chat generator to expand queries. The chat generator is expected to return a JSON response
with the following structure:

```json
{"queries": ["expanded query 1", "expanded query 2", "expanded query 3"]}
```

### Usage example

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

#### `__init__`

```python
__init__(
    *,
    chat_generator: ChatGenerator | None = None,
    prompt_template: str | None = None,
    n_expansions: int = 4,
    include_original_query: bool = True
) -> None
```

Initialize the QueryExpander component.

**Parameters:**

- **chat_generator** (<code>ChatGenerator | None</code>) – The chat generator component to use for query expansion.
  If None, a default OpenAIChatGenerator with gpt-4.1-mini model is used.
- **prompt_template** (<code>str | None</code>) – Custom [PromptBuilder](https://docs.haystack.deepset.ai/docs/promptbuilder)
  template for query expansion. The template should instruct the LLM to return a JSON response with the
  structure: `{"queries": ["query1", "query2", "query3"]}`. The template should include 'query' and
  'n_expansions' variables.
- **n_expansions** (<code>int</code>) – Number of alternative queries to generate (default: 4).
- **include_original_query** (<code>bool</code>) – Whether to include the original query in the output.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> QueryExpander
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary with serialized data.

**Returns:**

- <code>QueryExpander</code> – Deserialized component.

#### `run`

```python
run(query: str, n_expansions: int | None = None) -> dict[str, list[str]]
```

Expand the input query into multiple semantically similar queries.

The language of the original query is preserved in the expanded queries.

**Parameters:**

- **query** (<code>str</code>) – The original query to expand.
- **n_expansions** (<code>int | None</code>) – Number of additional queries to generate (not including the original).
  If None, uses the value from initialization. Can be 0 to generate no additional queries.

**Returns:**

- <code>dict\[str, list\[str\]\]</code> – Dictionary with "queries" key containing the list of expanded queries.
  If include_original_query=True, the original query will be included in addition
  to the n_expansions alternative queries.

**Raises:**

- <code>ValueError</code> – If n_expansions is not positive (less than or equal to 0).

#### `warm_up`

```python
warm_up()
```

Warm up the LLM provider component.
