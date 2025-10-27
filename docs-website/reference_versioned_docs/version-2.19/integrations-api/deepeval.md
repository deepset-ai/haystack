---
title: "DeepEval"
id: integrations-deepeval
description: "DeepEval integration for Haystack"
slug: "/integrations-deepeval"
---

<a id="haystack_integrations.components.evaluators.deepeval.evaluator"></a>

## Module haystack\_integrations.components.evaluators.deepeval.evaluator

<a id="haystack_integrations.components.evaluators.deepeval.evaluator.DeepEvalEvaluator"></a>

### DeepEvalEvaluator

A component that uses the [DeepEval framework](https://docs.confident-ai.com/docs/evaluation-introduction)
to evaluate inputs against a specific metric. Supported metrics are defined by `DeepEvalMetric`.

Usage example:
```python
from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric

evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.FAITHFULNESS,
    metric_params={"model": "gpt-4"},
)
output = evaluator.run(
    questions=["Which is the most popular global sport?"],
    contexts=[
        [
            "Football is undoubtedly the world's most popular sport with"
            "major events like the FIFA World Cup and sports personalities"
            "like Ronaldo and Messi, drawing a followership of more than 4"
            "billion people."
        ]
    ],
    responses=["Football is the most popular sport with around 4 billion" "followers worldwide"],
)
print(output["results"])
```

<a id="haystack_integrations.components.evaluators.deepeval.evaluator.DeepEvalEvaluator.__init__"></a>

#### DeepEvalEvaluator.\_\_init\_\_

```python
def __init__(metric: Union[str, DeepEvalMetric],
             metric_params: Optional[Dict[str, Any]] = None)
```

Construct a new DeepEval evaluator.

**Arguments**:

- `metric`: The metric to use for evaluation.
- `metric_params`: Parameters to pass to the metric's constructor.
Refer to the `RagasMetric` class for more details
on required parameters.

<a id="haystack_integrations.components.evaluators.deepeval.evaluator.DeepEvalEvaluator.run"></a>

#### DeepEvalEvaluator.run

```python
@component.output_types(results=List[List[Dict[str, Any]]])
def run(**inputs: Any) -> Dict[str, Any]
```

Run the DeepEval evaluator on the provided inputs.

**Arguments**:

- `inputs`: The inputs to evaluate. These are determined by the
metric being calculated. See `DeepEvalMetric` for more
information.

**Returns**:

A dictionary with a single `results` entry that contains
a nested list of metric results. Each input can have one or more
results, depending on the metric. Each result is a dictionary
containing the following keys and values:
- `name` - The name of the metric.
- `score` - The score of the metric.
- `explanation` - An optional explanation of the score.

<a id="haystack_integrations.components.evaluators.deepeval.evaluator.DeepEvalEvaluator.to_dict"></a>

#### DeepEvalEvaluator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Raises**:

- `DeserializationError`: If the component cannot be serialized.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.evaluators.deepeval.evaluator.DeepEvalEvaluator.from_dict"></a>

#### DeepEvalEvaluator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "DeepEvalEvaluator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.evaluators.deepeval.metrics"></a>

## Module haystack\_integrations.components.evaluators.deepeval.metrics

<a id="haystack_integrations.components.evaluators.deepeval.metrics.DeepEvalMetric"></a>

### DeepEvalMetric

Metrics supported by DeepEval.

All metrics require a `model` parameter, which specifies
the model to use for evaluation. Refer to the DeepEval
documentation for information on the supported models.

<a id="haystack_integrations.components.evaluators.deepeval.metrics.DeepEvalMetric.ANSWER_RELEVANCY"></a>

#### ANSWER\_RELEVANCY

Answer relevancy.\
Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`

<a id="haystack_integrations.components.evaluators.deepeval.metrics.DeepEvalMetric.FAITHFULNESS"></a>

#### FAITHFULNESS

Faithfulness.\
Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`

<a id="haystack_integrations.components.evaluators.deepeval.metrics.DeepEvalMetric.CONTEXTUAL_PRECISION"></a>

#### CONTEXTUAL\_PRECISION

Contextual precision.\
Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str], ground_truths: List[str]`\
The ground truth is the expected response.

<a id="haystack_integrations.components.evaluators.deepeval.metrics.DeepEvalMetric.CONTEXTUAL_RECALL"></a>

#### CONTEXTUAL\_RECALL

Contextual recall.\
Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str], ground_truths: List[str]`\
The ground truth is the expected response.\

<a id="haystack_integrations.components.evaluators.deepeval.metrics.DeepEvalMetric.CONTEXTUAL_RELEVANCE"></a>

#### CONTEXTUAL\_RELEVANCE

Contextual relevance.\
Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`

<a id="haystack_integrations.components.evaluators.deepeval.metrics.DeepEvalMetric.from_str"></a>

#### DeepEvalMetric.from\_str

```python
@classmethod
def from_str(cls, string: str) -> "DeepEvalMetric"
```

Create a metric type from a string.

**Arguments**:

- `string`: The string to convert.

**Returns**:

The metric.
