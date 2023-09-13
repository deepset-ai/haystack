- Title: Evaluation in Haystack 2.0
- Decision driver: (Silvano Cerza, Julian Risch)
- Start Date: 2023-08-23
- Proposal PR: #5794
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/5628

# Summary

Given the below requirements we redefine how evaluation and metrics are handled in Haystack 2.x.

Our goal is to lower the barrier of entry for new comers but also making it more flexible and extensible for more advanced and expert users.
All this while making it more modular and easier to test and maintain.

The requirements are:

- compare the performance of different pipelines on level of pipeline outputs (user perspective, integrated eval)
  - while running the full pipeline we can store intermediate results and calculate metrics for each component that returns answers or documents
- find out which component is the performance bottleneck in one pipeline by evaluating subpipelines (isolated evaluation)
- as above, get evaluation metrics for every component in a pipeline that returns answer or documents (ranker, retriever, reader, PromptNode)
- compare the performance of two components, for example two Readers, without the need to create a full retriever-reader pipeline
- export evaluation results to a file (similar to Haystack 1.x but faster) and evaluation report
- choose evaluation metrics from a list of metrics (e.g. F1, BLEU, ROUGE, Semantic Answer Similarity) based on the output type of a component
- evaluate pipelines that return ExtractiveAnswers
- evaluate pipelines that return GenerativeAnswers
- evaluate hallucinations (check generated answers are backed up by retrieved documents)
- evaluate pipelines with PromptNodes and arbitrary PromptTemplates (for example with Semantic Answer Similarity or BLEU, ROUGE (metrics from machine translation and summarization) if I provide labels)
- load evaluation data for example from BEIR

# Basic example

```python
pipe = Pipeline()
...
inputs = [{"component_name": {"query": "some question"}, ...}, ...]
expected_output = [{"another_component_name": {"answer": "42"}, ...}, ...]
result = eval(pipe, inputs=inputs, expected_output=expected_output)
metrics = result.calculate_metrics(Metric.SAS)
metrics.save("path/to/file.csv")
```

A more thorough example of a real use case can be found in the `0000-evaluation-haystack-2.py` file. It creates a small RAG Pipeline and shows how one would run evaluation on it.

# Motivation

Since the new version 2.x of Haystack is going toward a completely different approach to run `Pipeline`s and `component`s we also need to rework how we evaluate them.

The current implementation in version 1.x is convoluted and mixes evaluation and metrics at different steps during the process. This makes it harder to reason about it and maintain. This is noticeable also by the fact that only a limited amount of nodes can be evaluated. Also it's currently not easy to calculate custom metrics.

The goal of this new approach is to make it easier for users to evaluate and calculate metrics for their `Pipeline`s and `component`s. Evaluation is not an easy task to reason about and understand. Making it simpler will also make it less intimidating to less experienced users, pushing more people to approach this important part of Haystack.

It must be flexible by making it possible to evaluate any type of `component`. This is also an hard requirement since we want to make it easier for users to define new `component`s. Limiting the types of `component`s that can be evaluated would also limit the usage of custom or novelty ones, thus limiting the user creativity.

Extensibility is part of the goal too. We want to make is possible for user to calculate some of most common metrics, but also use custom logic to calculate any other one. As of now it's not possible to calculate custom metrics without knowing the nitty gritty of the Haystack internals.

All these goals merge also into another goal, maintainability. By making evaluation and metrics calculation more modular and easy to use we're also going to make it more maintainable and testable.

# Detailed design

### The `eval` function

We'll implement an `eval` function that will be able to evaluate all `Pipeline`s and `Component`s.
A minimal implementation could look like this:

```python
def eval(runnable: Union[Pipeline, Component], inputs: List[Dict[str, Any]], expected_outputs: List[Dict[str, Any]]) -> EvaluationResult:
    outputs = []
    for input_ in inputs:
      output = runnable.run(input_)
      outputs.append(output)
    return EvaluationResult(runnable, inputs, outputs, expected_outputs)
```

This is obviously an overtly simplistic example but the core concept remains.
`inputs` must be a list of data that will be passed to either the `Pipeline` or the `Component`.
`expected_outputs` could be a list with the same length of `inputs` or an empty list for blind evaluation.

Blind in this context means running an evaluation without providing a list of expected output. This could be done for several reasons, like if we don't know what to expect as output, or to compare output of different components.

`EvaluationResult` could either be a `Dict` or its own class, this is open to discussion. Either way it must be easy to save to disk. When saving the results to disk we can also include the `Pipeline` or `Component` in a serialized form.

When evaluating a `Pipeline` we could also override its private `_run_component` function to evaluate every node it will run. This will 100% work for our implementation of `Pipeline`. If a user tries to evaluate a `Pipeline` that reimplements its own `run` method it might not be able to evaluate each `Component`. I believe this a worthy risky tradeoff.

Overriding `_run_component` would also give us the chance to simulate optimal component outputs. `eval` could also accept an optional `simulated_output` dictionary containing the outputs of one or more `Component` that are in the `Pipeline`. It would look similar to this:

```python
simulated_output = {
  "component_name": {"answer": "120"},
  "another_component_name": {"metadata": {"id": 1}}
}
```

Another alternative would be to use the `expected_output` also to simulate outputs of intermediate components.

#### Tracking progress

To track progress we can also go on another direction. We could return partials results while iterating our inputs:

```
def eval(runnable, inputs, expected_outputs):
    result = EvaluationResult(runnable, inputs, {}, expected_outputs)
    for input_ in inputs:
      output = runnable.run(input_)
      result.append_output(output)
      yield result
```

Or return some progress percentage and only in the end the final result.

```
def eval(runnable, inputs, expected_outputs):
    outputs = []
    total = len(inputs)
    for i, input_ in enumerate(inputs):
      output = runnable.run(input_)
      outputs.append(output)
      yield 100 * (i / total), None

    yield 100, EvaluationResult(runnable, inputs, outputs, expected_outputs)
```

### `EvaluationResult`

`EvaluationResult` won't contain partial metrics anymore as it does in v1.x but it will keep track of all the information used by `eval()`.
This way we'll be able to save all the necessary information to a single file. That information will probably be:

- Serialized Pipeline or Component
- Inputs
- Expected outputs
- Actual outputs

This data should be serializable to string so that it can be saved to file and loaded back whenever necessary.
We shouldn't expect all input and output data to implement serialization methods like `to_dict` and `from_dict` like `Pipeline` and `component`s do, so we probably should find an alternative to handle serialization of types that don't. An unsafe option would be the use of `pickle`, but that's dangerous as it could lead to malicious code being executed.

Given the above information we should be able to implement a single method to calculate predeterminated metrics or even custom ones.
Known metrics could be defined as an enum to ease discoverability and documentation.

```python
class Metric(Enum):
    RECALL = "Recall"
    MRR = "Mean Reciprocal Rank"
    MAP = "Mean Average Precision"
    EM = "Exact Match"
    F1 = "F1"
    SAS = "SemanticAnswerSimilarity"
```

The method to calculate metrics could look similar to this:

```python
MetricsResult = Dict[str, Dict[str, float]]
MetricCalculator = Callable[..., MetricResult]

def calculate_metrics(self: EvaluationResult, metric: Union[Metric, MetricCalculator], **kwargs) -> MetricsResult:
    # Verify if we're calculating a known metric
    if metric == Metric.RECALL:
      return self._calculate_recall(**kwargs)
    elif metric == Metric.MRR:
      return self._calculate_mrr(**kwargs)
    # Other metrics...

    # If it's not a known metric it must be a custom one
    return metric(self, **kwargs)
```

This gives the users the flexibility to easily calculate metrics that we support but also use custom logic to calculate any kind of metric given the available data.
Since users will need to save their calculated metrics to file we could create a simple `MetricResult` class that simply wraps the generated metrics dictionary, something similar:

```python
class MetricResult(dict):
    def save(self, file: Union[str, Path]):
        # Dump info to file here
```

### Ease input specification

When declaring the input of a `Pipeline` we must specify both a `component` name and its input names. This can be annoying so we could simplify this by making certain assumptions.

An approach to this problem could be letting the user skip the `component` name specification when declaring the `Pipeline` input. This could work up until the point the user changes their `Pipeline` to have multiple inputs with the same name.

So given a `Pipeline` that has single input `component` name `foo` that takes a `query` as its input we can let the user specify the `eval` input like so:

```python
eval(pipe, {"query": "This is the query"})
```

If the user adds a new `component` name `bar` that also takes a `query` as input we'll make evaluation fail preventively since we cannot be sure whether both `component`s must take `query` as input and force explicit specification:

```python
eval(pipe, {"foo": {"query": "This is the query"}, "bar": {"query": "This is the query"}})
```

# Drawbacks

The major drawback found from the feedback gathered is always the same, and it's also common to `Pipeline.run()`. It's frustrating and annoying having to update the evaluation input and/or output data if I make changes to my `Pipeline`.

Given that new `Pipeline` can have multiple inputs to different `component`s we must specify which component will take which input. As an example given a `Pipeline` with two input components called `foo` and `bar` that takes a `input_query` value we'll have to specify input as follow:

```python
input = {
  "foo": {"input_query": "This my input query"},
  "bar": {"input_query": "This my input query"}
}
```

I believe this a worthy tradeoff as it gives huge amount of customization that wasn't possible in Haystack 1.x.

We could ease inputs specification in some cases as specified in the above section but that would make evaluation prone to errors. I believe that to be a dangerous approach as it could mean making evaluation "easier to use" at the cost of making it also more error prone if not done properly and with the correct safe guards.

The same thing can be said for the `Pipeline` output. There will be multiple outputs most of the times since we're going to evaluate individual nodes and the whole `Pipeline` input with a single evaluation run. So the user must specify from which `component` the output should be expected from.

Expected output specification suffers from the same issue of inputs specification. So making assumption to try and match expected output with the actual `Pipeline` output is still error prone but probably can be handled for really simple `Pipeline`s with only a single output `component`.

Evaluation should be an advanced topic for users that know what they're actually doing. This might seem contradictory to what has been said above regarding approachability of the feature but I believe this to be an easy to use but and hard to master feature.

# Adoption strategy

This is obviously a breaking change as it's meant for Haystack 2.x.

# How we teach this

Much like the current situation we'll write tutorials, examples and documentation to go along with this new feature. We can also leverage future ready-made `Pipeline`s to show how to evaluate them.

We're also going to have new community spotlights in Discord to show how to approach evaluation in Haystack 2.x.

# Unresolved questions

Evaluation of pipelines containing Agents or other loops is out of scope for this proposal (except for integrated pipeline evaluation).
