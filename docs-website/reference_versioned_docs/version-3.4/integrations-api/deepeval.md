---
title: "DeepEval"
id: integrations-deepeval
description: "DeepEval integration for Haystack"
slug: "/integrations-deepeval"
---


## haystack_integrations.components.evaluators.deepeval.evaluator

### DeepEvalEvaluator

A component that uses DeepEval to evaluate inputs against a specific metric.

Uses the [DeepEval framework](https://docs.confident-ai.com/docs/evaluation-introduction).
Supported metrics are defined by `DeepEvalMetric`.

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

#### __init__

```python
__init__(
    metric: str | DeepEvalMetric, metric_params: dict[str, Any] | None = None
) -> None
```

Construct a new DeepEval evaluator.

**Parameters:**

- **metric** (<code>str | DeepEvalMetric</code>) – The metric to use for evaluation.
- **metric_params** (<code>dict\[str, Any\] | None</code>) – Parameters to pass to the metric's constructor.
  Refer to the `RagasMetric` class for more details
  on required parameters.

#### run

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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

**Raises:**

- <code>DeserializationError</code> – If the component cannot be serialized.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> DeepEvalEvaluator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>DeepEvalEvaluator</code> – Deserialized component.

## haystack_integrations.components.evaluators.deepeval.metrics

### DeepEvalMetric

Bases: <code>Enum</code>

Metrics supported by DeepEval.

All metrics require a `model` parameter, which specifies
the model to use for evaluation. Refer to the DeepEval
documentation for information on the supported models.

#### from_str

```python
from_str(string: str) -> DeepEvalMetric
```

Create a metric type from a string.

**Parameters:**

- **string** (<code>str</code>) – The string to convert.

**Returns:**

- <code>DeepEvalMetric</code> – The metric.

### MetricResult

Result of a metric evaluation.

**Parameters:**

- **name** (<code>str</code>) – The name of the metric.
- **score** (<code>float | None</code>) – The score of the metric.
- **explanation** (<code>str | None</code>) – An optional explanation of the metric.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Convert the metric result to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the metric result fields.

### MetricDescriptor

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

#### new

```python
new(
    metric: DeepEvalMetric,
    backend: type[BaseMetric],
    input_converter: Callable[[Any], Iterable[LLMTestCase]],
    output_converter: Callable[[TestResult], list[MetricResult]] | None = None,
    *,
    init_parameters: Mapping[str, type] | None = None
) -> MetricDescriptor
```

Create a new metric descriptor, inferring input parameters from the converter signature.

**Parameters:**

- **metric** (<code>DeepEvalMetric</code>) – The metric enum value.
- **backend** (<code>type\[BaseMetric\]</code>) – The DeepEval metric class to instantiate.
- **input_converter** (<code>Callable\\[[Any\], Iterable\[LLMTestCase\]\]</code>) – Callable that converts Haystack inputs to DeepEval test cases.
- **output_converter** (<code>Callable\\[[TestResult\], list\[MetricResult\]\] | None</code>) – Callable that converts DeepEval results to `MetricResult` objects.
  If `None`, the default output converter is used.
- **init_parameters** (<code>Mapping\[str, type\] | None</code>) – Optional mapping of parameter names to types accepted by the backend metric's constructor.

**Returns:**

- <code>MetricDescriptor</code> – A new `MetricDescriptor` instance.

### InputConverters

Converters for input parameters.

The signature of the converter functions serves as the ground-truth of the
expected input parameters of a given metric. They are also responsible for validating
the input parameters and converting them to the format expected by DeepEval.

#### validate_input_parameters

```python
validate_input_parameters(
    metric: DeepEvalMetric, expected: dict[str, Any], received: dict[str, Any]
) -> None
```

Validate that all expected input parameters are present in the received inputs.

**Parameters:**

- **metric** (<code>DeepEvalMetric</code>) – The metric being evaluated, used for error messages.
- **expected** (<code>dict\[str, Any\]</code>) – Dictionary of expected parameter names to their types.
- **received** (<code>dict\[str, Any\]</code>) – Dictionary of received parameter names to their values.

**Raises:**

- <code>ValueError</code> – If a required parameter is missing from `received`.

#### question_context_response

```python
question_context_response(
    questions: list[str], contexts: list[list[str]], responses: list[str]
) -> Iterable[LLMTestCase]
```

Convert question, context, and response inputs to DeepEval test cases.

**Parameters:**

- **questions** (<code>list\[str\]</code>) – List of input questions.
- **contexts** (<code>list\[list\[str\]\]</code>) – List of retrieval context lists, one per question.
- **responses** (<code>list\[str\]</code>) – List of model responses, one per question.

**Returns:**

- <code>Iterable\[LLMTestCase\]</code> – An iterable of `LLMTestCase` objects.

#### question_context_response_ground_truth

```python
question_context_response_ground_truth(
    questions: list[str],
    contexts: list[list[str]],
    responses: list[str],
    ground_truths: list[str],
) -> Iterable[LLMTestCase]
```

Convert question, context, response, and ground truth inputs to DeepEval test cases.

**Parameters:**

- **questions** (<code>list\[str\]</code>) – List of input questions.
- **contexts** (<code>list\[list\[str\]\]</code>) – List of retrieval context lists, one per question.
- **responses** (<code>list\[str\]</code>) – List of model responses, one per question.
- **ground_truths** (<code>list\[str\]</code>) – List of expected (ground truth) responses, one per question.

**Returns:**

- <code>Iterable\[LLMTestCase\]</code> – An iterable of `LLMTestCase` objects.

### OutputConverters

Converters for results returned by DeepEval.

They are responsible for converting the results to our output format.

#### default

```python
default(metric: DeepEvalMetric) -> Callable[[TestResult], list[MetricResult]]
```

Return the default output converter for a given metric.

**Parameters:**

- **metric** (<code>DeepEvalMetric</code>) – The metric for which to create the converter.

**Returns:**

- <code>Callable\\[[TestResult\], list\[MetricResult\]\]</code> – A callable that converts a `TestResult` to a list of `MetricResult` objects.
