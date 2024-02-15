- Title: Integration of LLM Evaluation Frameworks
- Decision driver: Madeesh Kannan, Julian Risch
- Start Date: 2024-01-19
- Proposal PR: https://github.com/deepset-ai/haystack/pull/6784
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/6672

# Summary

- Create integrations for three different LLM evaluation frameworks in https://github.com/deepset-ai/haystack-core-integrations
- The LLM evaluation frameworks in question are: [Uptrain](https://docs.uptrain.ai/getting-started/introduction), [RAGAS](https://docs.ragas.io/en/stable/index.html) and [DeepEval](https://docs.confident-ai.com/docs/getting-started).
- The integrations introduce the following components in this order: `UptrainEvaluator`, `RagasEvaluator`, `DeepEvalEvaluator`.
- Users can run a Haystack pipeline and evaluate the result with model-based metrics implemented by the evaluation frameworks.
- Calculation of metrics is done by the frameworks through running prompts with OpenAI (Uptrain) or using langchain to make the OpenAI call (RAGAS, DeepEval).

# Basic example

With the integration, users can use Haystack’s pipeline concept for the evaluation too. They need to provide the outputs of the RAG pipeline as inputs to the Evaluator component in an evaluation pipeline:

```python
p = Pipeline()
p.add_component(instance=DeepEvalEvaluator(metric="Hallucination", params={"threshold": 0.3)}, name="evaluator"))
# p.add_component(instance=RagasEvaluator()...

questions = [...]
contexts = [...]
answers = [...]

p.run({"evaluator": {"questions": questions, "context": contexts, "answer": answers})
# {"evaluator": DeepEvalResult(metric='hallucination', score=0.817)}
```

# Motivation

Users of Haystack that deploy RAG pipelines currently do not have an avenue of evaluating the outputs of the same. Traditional methods of evaluation that involves the creation of labelled datasets is often out of reach for open-source practitioners due resource- and time-constraints. Furthermore, this approach is not necessarily scalable and applicable to LLMs due to the generative property of RAG-based information retrieval.

This is given rise to model-based evaluation approaches, i.e., the method of training language models to act as classifiers and scorers. A very popular implementation of this approach revolves around designing prompt-based natural language metrics that are used in conjunction with an instruction-trained LLM. The LLM then acts as the judge, evaluating the outputs based on the criteria defined in the metrics. This approach is eminently more scalable and low-friction for the end-user.

Other LLM application frameworks such as LlamaIndex already provide support for this approach to evaluation, and it is in Haystack's interests to do the same. This will also help establish a baseline against which our future efforts in this area can be compared.

# Detailed design

As with evaluation in Haystack 1.x, we reaffirm the core idea of implementing different pipelines for different concerns. We consider evaluation a separate process and consequently separate the execution of RAG and the metric calculation into two different pipelines. This allows for greater flexibility - for instance, the evaluation pipeline could contain an additional component that routes the inputs to different evaluator components based on certain criteria, etc. Another example would be the ability to convert the inputs from/to different formats before passing them to the evaluator.

A further advantage of this approach is that any tool we develop in the future to facilitate introspection and observability of pipelines can transparently be appled to evaluation as well.

The implementation of the three evaluator components should follow the general guidelines for custom component development. There are two approaches we could take:

- **Metric wrapping**: Make each metric an individual component.

  Advantages:

  - Inputs and outputs are explicitly defined for each metric, which makes the API more explicit.
  - We can control which metrics to include.
  - We can use the execution graph for control flow.

  Disadvantages:

  - Duplication between frameworks.
  - Maintenance burden.
  - Less freedom for the user.

- **Framework wrapping**: Make each evaluation framework an individual component.

  Advantages:

  - Straightforward implementation.
  - Easier to reason about.
  - Low maintenance burden.
  - More freedom for the user.

  Disadvantages:

  - API has to accommodate all possible inputs for all supported metrics.
  - Less flexibility.

Given the above comparison, **we will be implementing the second approach, i.e., framework as the component**. The disadvantages mentioned above can be mitigated by leaning into Haystack 2.x's I/O system.

## API overview

- We implement three components: `DeepEvalEvaluator`, `RagasEvaluator`, `UpTrainEvaluator`.
- Their constructors take a two parameters: `Literal["MetricName"]` and `Optional[Dict[str, Any]]` for the metric name and optional parameters for the same.
  - We use JSON-serializable types here to ensure serde support.
- The component initializes the internal framework-specific metric representation.
- Depending on the metric, the component also calls `set_input_types` to register the appropriate input sockets for the given metric, e.g: `question`, `contexts`, `answer` for groundness, etc.
  - This approach lets the user modulate the inputs individually, which wouldn't be possible if we use a generic representation for all metrics/frameworks.
- Outputs can be implemented in one of two ways:

  - `set_output_types` is also called to enumerate the outputs of the metric
  - The output is a `dataclass` specific to the evaluation framework.

  As opposed to their inputs, the outputs of metrics are not likely to require sophisticated routing for their usage further downstream. So, each evaluator will implement a dataclass that encapsulates the results of the evaluation.

### Illustrative example

```python
from deepeval import BaseMetric, FaithfulnessMetric

@component
class DeepEvalEvaluator:
	self._metric: BaseMetric

	def __init__(self, metric: str, params: Optional[Dict[str, Any]]):
		params = {} if params is None
		if metric == "Faithfulness":
			self._metric = FaithfulnessMetric(**params)
			self.set_input_types(questions=List[str], answers=List[str], contexts=Listt[List[str]])
		elif metric == "ContextRecall":
			...

	def run(self, **kwargs):
		# Logic to unwrap the inputs based on the metric and
        # execute the backend code.
        ...

```

# Drawbacks

- **Lack of support for custom components**: The aforementioned API inherently restricts the user to the pre-defined metrics provided by the frameworks. On the other hand, we'd like to keep these integrations limited in their scope. Since we also plan on providing custom metric support in the future, this becomes a moot point.
- **No support for batching across metrics**: Batching here refers to calculating multiple metrics on the inputs with a single API call. With the exception of UpTrain, the frameworks in question do not support this either. UpTrain's implementation is not publicly available, so we cannot determine if this type of batching happens on their server.
- **Additional dependencies**: RAGAS and DeepEval depend on langchain, whereas UpTrain uses both its own client and OpenAI's.
  - https://github.com/confident-ai/deepeval/blob/main/pyproject.toml
  - https://github.com/explodinggradients/ragas/blob/main/pyproject.toml
  - https://github.com/uptrain-ai/uptrain/blob/main/pyproject.toml

# Alternatives

- Eschewing the pipeline paradigm and using a separate evaluation API.
  - Similar to how it was done in Haystack 1.x.
- Implement them as components but run them individually.
  - This would also be possible with the proposed API.

# Adoption strategy

This is a new feature with no breaking changes. Existing users can simply try out the evaluation with existing pipelines after installing additional dependencies and providing an OpenAI API key.

# How we teach this

We provide a new section in the documentation about evaluation. This proposal specifically deals with model-based evaluation, so it would be prudent to have a separate subsection for it.

We should impress upon them the idea that evaluation is "just" another pipeline with different steps. A tutorial would also be helpful to guide them through the same. Apart from that, we should include pointers to the API docs of the different evaluation frameworks, etc.

We clarify in the documentation how users can decide which Evaluation framework is best for them with a simple overview. They should not need to research the different frameworks before running their first evaluation.

# Unresolved questions & future work

- We see the integrated evaluation frameworks as a baseline. To what extent and when Haystack will have its own model-based metrics is out-of-scope for this proposal.
- We envision an `Evaluator` component in Haystack's core with customizable model and prompt.
- Terminology around existing `calculate_metrics`/`eval` functions and the `EvaluationResult` class need to be discussed (c.f https://github.com/deepset-ai/haystack/pull/6505).
- Alternative take on the evaluation API - https://github.com/deepset-ai/haystack/pull/5794/
  - The ideas proposed in the above should compatible with those of this proposal.
