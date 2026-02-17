---
title: "Ragas"
id: integrations-ragas
description: "Ragas integration for Haystack"
slug: "/integrations-ragas"
---


## `haystack_integrations.components.evaluators.ragas.evaluator`

### `RagasEvaluator`

A component that uses the [Ragas framework](https://docs.ragas.io/) to evaluate
inputs against specified Ragas metrics.

Usage example:

```python
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.evaluators.ragas import RagasEvaluator
from ragas.metrics import ContextPrecision
from ragas.llms import HaystackLLMWrapper

llm = OpenAIGenerator(model="gpt-4o-mini")
evaluator_llm = HaystackLLMWrapper(llm)

evaluator = RagasEvaluator(
    ragas_metrics=[ContextPrecision()],
    evaluator_llm=evaluator_llm
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

#### `__init__`

```python
__init__(
    ragas_metrics: list[Metric],
    evaluator_llm: BaseRagasLLM | None = None,
    evaluator_embedding: BaseRagasEmbeddings | None = None,
)
```

Constructs a new Ragas evaluator.

**Parameters:**

- **ragas_metrics** (<code>list\[Metric\]</code>) – A list of evaluation metrics from the [Ragas](https://docs.ragas.io/) library.
- **evaluator_llm** (<code>BaseRagasLLM | None</code>) – A language model used by metrics that require LLMs for evaluation.
- **evaluator_embedding** (<code>BaseRagasEmbeddings | None</code>) – An embedding model used by metrics that require embeddings for evaluation.

#### `run`

```python
run(
    query: str | None = None,
    response: list[ChatMessage] | str | None = None,
    documents: list[Document | str] | None = None,
    reference_contexts: list[str] | None = None,
    multi_responses: list[str] | None = None,
    reference: str | None = None,
    rubrics: dict[str, str] | None = None,
) -> dict[str, Any]
```

Evaluates the provided query against the documents and returns the evaluation result.

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

- <code>dict\[str, Any\]</code> – A dictionary containing the evaluation result.
