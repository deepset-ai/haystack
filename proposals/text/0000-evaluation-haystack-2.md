- Title: Evaluation in Haystack 2.0
- Decision driver: (your name here)
- Start Date: 2023-08-23
- Proposal PR: (fill in after opening the PR)
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/5265

# Summary

As a requirement for evaluation in Haystack 2.0, as a user, I want to:

- compare the performance of different pipelines on level of pipeline outputs (user perspective, integrated eval)
  - while running the full pipeline we can store intermediate results and calculate metrics for each component that returns answers or documents
- find out which component is the performance bottleneck in one pipeline by evaluating subpipelines (isolated evaluation)
- as above, get evaluation metrics for every component in a pipeline that returns or documents (ranker, retriever, reader, PromptNode)
- compare the performance of two components, for example two Readers, without the need to create a full retriever-reader pipeline
- export evaluation results to a file (similar to Haystack 1.x but faster) and evaluation report
- choose evaluation metrics from a list of metrics (e.g. F1, BLEU, ROUGE, Semantic Answer Similarity) based on the output type of a component
- evaluate pipelines that return ExtractiveAnswers
- evaluate pipelines that return GenerativeAnswers
- evaluate hallucinations (check generated answers are backed up by retrieved documents)
- evaluate pipelines with PromptNodes and arbitrary PromptTemplates (for example with Semantic Answer Similarity or BLEU, ROUGE (metrics from machine translation and summarization) if I provide labels)
- load evaluation data for example from BEIR

# Basic example

```
pipe = Pipeline()
...
inputs = [{"component_name": {"query": "some question"}, ...}, ...]
expected_output = [{"another_component_name": {"answer": "42"}, ...}, ...]
result = eval(pipe, inputs=inputs, expected_output=expected_output)
metrics = result.calculate_metrics()
```

# Motivation

Give us more background and explanation: Why do we need this feature? What use cases does it support? What's the expected
outcome?

Focus on explaining the motivation for this feature. We'd like to understand it, so that even if we don't accept this
proposal, others can use the motivation to develop alternative solutions.

# Detailed design

### The `eval` function

With evaluation in Haystack 2.0 we want to give the users a flexible way to evaluate their Pipelines and Components.
We'll implement an `eval` function that will be able to evaluate all `Pipeline`s and `Component`s.
A minimal implementation could look like this:

```
def eval(runnable: Union[Pipeline, Component], inputs: List[Dict[str, Any]], expected_outputs: List[Dict[str, Any]]) -> EvaluationResult:
    output = []
    for input_ in inputs:
      output = runnable.run(input_)
      outputs.append(output)
    return EvaluationResult(runnable, outputs, expected_outputs)
```

This is obviously an overtly simplistic example but the core concept remains.
`inputs` must be a list of data that will be passed to either the `Pipeline` or the `Component`.
`expected_outputs` could be a list with the same length of `inputs` or an empty list for blind evaluation.

`EvaluationResult` could either be a `Dict` or its own class, this is open to discussion. Either way it must be easy to save to disk. When saving the results to disk we can also include the `Pipeline` or `Component` in a serialized form.

When evaluating a `Pipeline` we could also override its private `_run_component` function to evaluate every node it will run. This will 100% work for our implementation of `Pipeline`. If a user tries to evaluate a `Pipeline` that reimplements its own `run` method it might not be able to evaluate each `Component`. I believe this a worthy risky tradeoff.

Overriding `_run_component` would also give us the chance to simulate optimal component outputs. `eval` could also accept an optional `simulated_output` dictionary containing the outputs of one or more `Component` that are in the `Pipeline`. It would look similar to this:

```
simulated_output = {
  "component_name": {"answer": "120"},
  "another_component_name": {"metadata": {"id": 1}}
}
```

### `EvaluationResult`

TODO

### Export evaluation results to a file (similar to Haystack 1.x but faster)

Haystack 1.x iterates through the queries one by one when calculating metrics, which is slow.
Haystack 2.0 should increase the speed of calculating metrics by using a batched approach.

This is the bulk of the proposal. Explain the design in enough detail for somebody
familiar with Haystack to understand, and for somebody familiar with the
implementation to implement. Get into specifics and corner-cases,
and include examples of how the feature is used. Also, if there's any new terminology involved,
define it here.

# Drawbacks

Look at the feature from the other side: what are the reasons why we should _not_ work on it? Consider the following:

- What's the implementation cost, both in terms of code size and complexity?
- Can the solution you're proposing be implemented as a separate package, outside of Haystack?
- Does it teach people more about Haystack?
- How does this feature integrate with other existing and planned features?
- What's the cost of migrating existing Haystack pipelines (is it a breaking change?)?

There are tradeoffs to choosing any path. Attempt to identify them here.

# Alternatives

What other designs have you considered? What's the impact of not adding this feature?

# Adoption strategy

If we implement this proposal, how will the existing Haystack users adopt it? Is
this a breaking change? Can we write a migration script?

# How we teach this

Would implementing this feature mean the documentation must be re-organized
or updated? Does it change how Haystack is taught to new developers at any level?

How should this feature be taught to the existing Haystack users (for example with a page in the docs,
a tutorial, ...).

# Unresolved questions

Evaluation of pipelines containing Agents or other loops is out of scope for this proposal (except for integrated pipeline evaluation).
