---
title: "Ragas"
id: integrations-ragas
description: "Ragas integration for Haystack"
slug: "/integrations-ragas"
---

<a id="haystack_integrations.components.evaluators.ragas.evaluator"></a>

## Module haystack\_integrations.components.evaluators.ragas.evaluator

<a id="haystack_integrations.components.evaluators.ragas.evaluator.RagasEvaluator"></a>

### RagasEvaluator

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

<a id="haystack_integrations.components.evaluators.ragas.evaluator.RagasEvaluator.__init__"></a>

#### RagasEvaluator.\_\_init\_\_

```python
def __init__(ragas_metrics: list[Metric],
             evaluator_llm: BaseRagasLLM | None = None,
             evaluator_embedding: BaseRagasEmbeddings | None = None)
```

Constructs a new Ragas evaluator.

**Arguments**:

- `ragas_metrics`: A list of evaluation metrics from the [Ragas](https://docs.ragas.io/) library.
- `evaluator_llm`: A language model used by metrics that require LLMs for evaluation.
- `evaluator_embedding`: An embedding model used by metrics that require embeddings for evaluation.

<a id="haystack_integrations.components.evaluators.ragas.evaluator.RagasEvaluator.run"></a>

#### RagasEvaluator.run

```python
@component.output_types(result=EvaluationResult)
def run(query: str | None = None,
        response: list[ChatMessage] | str | None = None,
        documents: list[Document | str] | None = None,
        reference_contexts: list[str] | None = None,
        multi_responses: list[str] | None = None,
        reference: str | None = None,
        rubrics: dict[str, str] | None = None) -> dict[str, Any]
```

Evaluates the provided query against the documents and returns the evaluation result.

**Arguments**:

- `query`: The input query from the user.
- `response`: A list of ChatMessage responses (typically from a language model or agent).
- `documents`: A list of Haystack Document or strings that were retrieved for the query.
- `reference_contexts`: A list of reference contexts that should have been retrieved for the query.
- `multi_responses`: List of multiple responses generated for the query.
- `reference`: A string reference answer for the query.
- `rubrics`: A dictionary of evaluation rubric, where keys represent the score
and the values represent the corresponding evaluation criteria.

**Returns**:

A dictionary containing the evaluation result.

