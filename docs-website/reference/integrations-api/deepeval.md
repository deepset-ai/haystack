---
title: "DeepEval"
id: integrations-deepeval
description: "DeepEval integration for Haystack"
slug: "/integrations-deepeval"
---


## `haystack_integrations.components.evaluators.deepeval.evaluator`

### `DeepEvalEvaluator`

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

#### `__init__`

```python
__init__(
    metric: str | DeepEvalMetric, metric_params: dict[str, Any] | None = None
)
```

Construct a new DeepEval evaluator.

**Parameters:**

- **metric** (<code>str | DeepEvalMetric</code>) – The metric to use for evaluation.
- **metric_params** (<code>dict\[str, Any\] | None</code>) – Parameters to pass to the metric's constructor.
  Refer to the `RagasMetric` class for more details
  on required parameters.

#### `run`

```python
run(**inputs: Any) -> dict[str, Any]
```

Run the DeepEval evaluator on the provided inputs.

**Parameters:**

- **inputs** (<code>Any</code>) – The inputs to evaluate. These are determined by the
  metric being calculated. See `DeepEvalMetric` for more
  information.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with a single `results` entry that contains
  a nested list of metric results. Each input can have one or more
  results, depending on the metric. Each result is a dictionary
  containing the following keys and values:
- `name` - The name of the metric.
- `score` - The score of the metric.
- `explanation` - An optional explanation of the score.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

**Raises:**

- <code>DeserializationError</code> – If the component cannot be serialized.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> DeepEvalEvaluator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>DeepEvalEvaluator</code> – Deserialized component.

## `haystack_integrations.components.evaluators.deepeval.metrics`

### `DeepEvalMetric`

Bases: <code>Enum</code>

Metrics supported by DeepEval.

All metrics require a `model` parameter, which specifies
the model to use for evaluation. Refer to the DeepEval
documentation for information on the supported models.

#### `from_str`

```python
from_str(string: str) -> DeepEvalMetric
```

Create a metric type from a string.

**Parameters:**

- **string** (<code>str</code>) – The string to convert.

**Returns:**

- <code>DeepEvalMetric</code> – The metric.

### `MetricResult`

Result of a metric evaluation.

**Parameters:**

- **name** (<code>str</code>) – The name of the metric.
- **score** (<code>float | None</code>) – The score of the metric.
- **explanation** (<code>str | None</code>) – An optional explanation of the metric.

### `MetricDescriptor`

Descriptor for a metric.

**Parameters:**

- **metric** (<code>DeepEvalMetric</code>) – The metric.
- **backend** (<code>type\[BaseMetric\]</code>) – The associated DeepEval metric class.
- **input_parameters** (<code>dict\[str, type\]</code>) – Parameters accepted by the metric. This is used
  to set the input types of the evaluator component.
- **input_converter** (<code>Callable\\[[Any\], Iterable\[LLMTestCase\]\]</code>) – Callable that converts input parameters to the DeepEval input format.
- **output_converter** (<code>Callable\\[[TestResult\], list\[MetricResult\]\]</code>) – Callable that converts the DeepEval output format to our output format.
  Accepts a single output parameter and returns a list of results derived from it.
- **init_parameters** (<code>Mapping\[str, type\] | None</code>) – Additional parameters that need to be passed to the metric class during initialization.

### `InputConverters`

Converters for input parameters.

The signature of the converter functions serves as the ground-truth of the
expected input parameters of a given metric. They are also responsible for validating
the input parameters and converting them to the format expected by DeepEval.

### `OutputConverters`

Converters for results returned by DeepEval.

They are responsible for converting the results to our output format.
