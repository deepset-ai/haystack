---
title: "Ragas"
id: integrations-ragas
description: "Ragas integration for Haystack"
slug: "/integrations-ragas"
---


## haystack_integrations.components.evaluators.ragas.evaluator

### RagasEvaluator

A component that uses the Ragas framework to evaluate inputs against specified Ragas metrics.

See the [Ragas framework](https://docs.ragas.io/) for more details.

This component supports the modern Ragas metrics API (`ragas.metrics.collections`).
Each metric must be a `SimpleBaseMetric` instance with its LLM configured at construction time.

Usage example:

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from haystack_integrations.components.evaluators.ragas import RagasEvaluator

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

evaluator = RagasEvaluator(
    ragas_metrics=[Faithfulness(llm=llm)],
)
output = evaluator.run(
    query="Which is the most popular global sport?",
    documents=[
        "Football is undoubtedly the world's most popular sport with"
        " major events like the FIFA World Cup and sports personalities"
        " like Ronaldo and Messi, drawing a followership of more than 4"
        " billion people."
    ],
    reference="Football is the most popular sport with around 4 billion"
              " followers worldwide",
)

output['result']
```

#### __init__

```python
__init__(
    ragas_metrics: list[SimpleBaseMetric], concurrency_limit: int = 4
) -> None
```

Constructs a new Ragas evaluator.

**Parameters:**

- **ragas_metrics** (<code>list\[SimpleBaseMetric\]</code>) – A list of modern Ragas metrics from `ragas.metrics.collections`.
  Each metric must be fully configured (including its LLM) at construction time.
  Available metrics can be found in the
  [Ragas documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/).
- **concurrency_limit** (<code>int</code>) – The maximum number of metric evaluations that should be allowed to run concurrently.
  This parameter is only used in the `run_async` method.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> RagasEvaluator
```

Deserialize this component from a dictionary.

Metrics are reconstructed from their stored class path and LLM/embedding
configuration. Only the `openai` provider is supported for automatic
deserialization; the API key is read from the `OPENAI_API_KEY` environment
variable at load time.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>RagasEvaluator</code> – Deserialized component.

#### run

```python
run(
    query: str | None = None,
    response: list[ChatMessage] | str | None = None,
    documents: list[Document | str] | None = None,
    reference_contexts: list[str] | None = None,
    multi_responses: list[str] | None = None,
    reference: str | None = None,
    rubrics: dict[str, str] | None = None,
) -> dict[str, dict[str, MetricResult]]
```

Evaluates the provided inputs against each metric and returns the results.

**Parameters:**

- **query** (<code>str | None</code>) – The input query from the user.
- **response** (<code>list\[ChatMessage\] | str | None</code>) – A list of ChatMessage responses (typically from a language model or agent).
- **documents** (<code>list\[Document | str\] | None</code>) – A list of Haystack Document or strings that were retrieved for the query.
- **reference_contexts** (<code>list\[str\] | None</code>) – A list of reference contexts that should have been retrieved for the query.
- **multi_responses** (<code>list\[str\] | None</code>) – List of multiple responses generated for the query.
- **reference** (<code>str | None</code>) – A string reference answer for the query.
- **rubrics** (<code>dict\[str, str\] | None</code>) – A dictionary of evaluation rubric, where keys represent the score
  and the values represent the corresponding evaluation criteria.

**Returns:**

- <code>dict\[str, dict\[str, MetricResult\]\]</code> – A dictionary with key `result` mapping metric names to their `MetricResult`.

#### run_async

```python
run_async(
    query: str | None = None,
    response: list[ChatMessage] | str | None = None,
    documents: list[Document | str] | None = None,
    reference_contexts: list[str] | None = None,
    multi_responses: list[str] | None = None,
    reference: str | None = None,
    rubrics: dict[str, str] | None = None,
) -> dict[str, dict[str, MetricResult]]
```

Asynchronously evaluates the provided inputs against each metric and returns the results.

**Parameters:**

- **query** (<code>str | None</code>) – The input query from the user.
- **response** (<code>list\[ChatMessage\] | str | None</code>) – A list of ChatMessage responses (typically from a language model or agent).
- **documents** (<code>list\[Document | str\] | None</code>) – A list of Haystack Document or strings that were retrieved for the query.
- **reference_contexts** (<code>list\[str\] | None</code>) – A list of reference contexts that should have been retrieved for the query.
- **multi_responses** (<code>list\[str\] | None</code>) – List of multiple responses generated for the query.
- **reference** (<code>str | None</code>) – A string reference answer for the query.
- **rubrics** (<code>dict\[str, str\] | None</code>) – A dictionary of evaluation rubric, where keys represent the score
  and the values represent the corresponding evaluation criteria.

**Returns:**

- <code>dict\[str, dict\[str, MetricResult\]\]</code> – A dictionary with key `result` mapping metric names to their `MetricResult`.
