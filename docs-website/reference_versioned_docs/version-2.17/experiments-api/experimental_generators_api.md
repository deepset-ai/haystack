---
title: "Generators"
id: experimental-generators-api
description: "Enables text generation using LLMs."
slug: "/experimental-generators-api"
---

<a id="haystack_experimental.components.generators.chat.openai"></a>

## Module haystack\_experimental.components.generators.chat.openai

<a id="haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator"></a>

### OpenAIChatGenerator

An OpenAI chat-based text generator component that supports hallucination risk scoring.

This is based on the paper
[LLMs are Bayesian, in Expectation, not in Realization](https://arxiv.org/abs/2507.11768).

## Usage Example:

    ```python
    from haystack.dataclasses import ChatMessage

    from haystack_experimental.utils.hallucination_risk_calculator.dataclasses import HallucinationScoreConfig
    from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator

    # Evidence-based Example
    llm = OpenAIChatGenerator(model="gpt-4o")
    rag_result = llm.run(
        messages=[
            ChatMessage.from_user(
                text="Task: Answer strictly based on the evidence provided below.
"
                "Question: Who won the Nobel Prize in Physics in 2019?
"
                "Evidence:
"
                "- Nobel Prize press release (2019): James Peebles (1/2); Michel Mayor & Didier Queloz (1/2).
"
                "Constraints: If evidence is insufficient or conflicting, refuse."
            )
        ],
        hallucination_score_config=HallucinationScoreConfig(skeleton_policy="evidence_erase"),
    )
    print(f"Decision: {rag_result['replies'][0].meta['hallucination_decision']}")
    print(f"Risk bound: {rag_result['replies'][0].meta['hallucination_risk']:.3f}")
    print(f"Rationale: {rag_result['replies'][0].meta['hallucination_rationale']}")
    print(f"Answer:
{rag_result['replies'][0].text}")
    print("---")
    ```

<a id="haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator.run"></a>

#### OpenAIChatGenerator.run

```python
@component.output_types(replies=list[ChatMessage])
def run(
    messages: list[ChatMessage],
    streaming_callback: Optional[StreamingCallbackT] = None,
    generation_kwargs: Optional[dict[str, Any]] = None,
    *,
    tools: Optional[ToolsType] = None,
    tools_strict: Optional[bool] = None,
    hallucination_score_config: Optional[HallucinationScoreConfig] = None
) -> dict[str, list[ChatMessage]]
```

Invokes chat completion based on the provided messages and generation parameters.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will
override the parameters passed during component initialization.
For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If set, it will override the `tools` parameter provided during initialization.
- `tools_strict`: Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
the schema provided in the `parameters` field of the tool definition, but this may increase latency.
If set, it will override the `tools_strict` parameter set during component initialization.
- `hallucination_score_config`: If provided, the generator will evaluate the hallucination risk of its responses using
the OpenAIPlanner and annotate each response with hallucination metrics.
This involves generating multiple samples and analyzing their consistency, which may increase
latency and cost. Use this option when you need to assess the reliability of the generated content
in scenarios where accuracy is critical.
For details, see the [research paper](https://arxiv.org/abs/2507.11768)

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances. If hallucination
scoring is enabled, each message will include additional metadata:
  - `hallucination_decision`: "ANSWER" if the model decided to answer, "REFUSE" if it abstained.
  - `hallucination_risk`: The EDFL hallucination risk bound.
  - `hallucination_rationale`: The rationale behind the hallucination decision.

<a id="haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator.run_async"></a>

#### OpenAIChatGenerator.run\_async

```python
@component.output_types(replies=list[ChatMessage])
async def run_async(
    messages: list[ChatMessage],
    streaming_callback: Optional[StreamingCallbackT] = None,
    generation_kwargs: Optional[dict[str, Any]] = None,
    *,
    tools: Optional[ToolsType] = None,
    tools_strict: Optional[bool] = None,
    hallucination_score_config: Optional[HallucinationScoreConfig] = None
) -> dict[str, list[ChatMessage]]
```

Asynchronously invokes chat completion based on the provided messages and generation parameters.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
Must be a coroutine.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will
override the parameters passed during component initialization.
For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If set, it will override the `tools` parameter provided during initialization.
- `tools_strict`: Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
the schema provided in the `parameters` field of the tool definition, but this may increase latency.
If set, it will override the `tools_strict` parameter set during component initialization.
- `hallucination_score_config`: If provided, the generator will evaluate the hallucination risk of its responses using
the OpenAIPlanner and annotate each response with hallucination metrics.
This involves generating multiple samples and analyzing their consistency, which may increase
latency and cost. Use this option when you need to assess the reliability of the generated content
in scenarios where accuracy is critical.
For details, see the [research paper](https://arxiv.org/abs/2507.11768)

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances. If hallucination
scoring is enabled, each message will include additional metadata:
  - `hallucination_decision`: "ANSWER" if the model decided to answer, "REFUSE" if it abstained.
  - `hallucination_risk`: The EDFL hallucination risk bound.
  - `hallucination_rationale`: The rationale behind the hallucination decision.

