---
title: "Respan"
id: integrations-respan
description: "Respan integration for tracing Haystack pipelines and routing LLM calls through the Respan gateway."
slug: "/integrations-respan"
---

# Respan

Respan provides observability and gateway routing for Haystack applications.
With `respan-instrumentation-haystack`, Haystack pipeline, component, and LLM
spans are captured through OpenInference and exported to the Respan tracing
pipeline.

Use this integration to:

- Trace full Haystack pipeline runs, including component inputs, outputs, and
  timings.
- Inspect LLM calls, token usage, and errors in the Respan traces view.
- Attach customer, thread, environment, and custom metadata to traces.
- Route OpenAI-compatible Haystack generator calls through the Respan gateway.
- Use Respan prompt management through Haystack's OpenAI chat generator request
  body.

## Installation

Install Haystack, the Respan SDK, and the Haystack instrumentation package:

```bash
pip install haystack-ai respan-ai respan-instrumentation-haystack
```

Set your API keys and enable Haystack content tracing:

```bash
export RESPAN_API_KEY="YOUR_RESPAN_API_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export HAYSTACK_CONTENT_TRACING_ENABLED="true"
```

`RESPAN_API_KEY` exports traces to Respan. `OPENAI_API_KEY` is used by
Haystack's OpenAI components when you call OpenAI directly.

## Trace a pipeline

Initialize Respan before running your Haystack pipeline. The instrumentor
automatically captures the pipeline run, component spans, and LLM request spans.

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor

respan = Respan(instrumentations=[HaystackInstrumentor()])

pipeline = Pipeline()
pipeline.add_component(
    "prompt_builder",
    PromptBuilder(template="Answer the following question: {{ question }}"),
)
pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))
pipeline.connect("prompt_builder", "generator")

result = pipeline.run(
    {"prompt_builder": {"question": "What is the capital of France?"}}
)
print(result["generator"]["replies"][0])

respan.flush()
```

Open [Respan traces](https://platform.respan.ai/platform/traces) to inspect the
pipeline run.

## Add trace metadata

Use `propagate_attributes()` when you want every span from a request to include
the same user, conversation, or application metadata.

```python
from respan import Respan, propagate_attributes
from respan_instrumentation_haystack import HaystackInstrumentor

respan = Respan(
    instrumentations=[HaystackInstrumentor()],
    environment="production",
    metadata={"service": "haystack-rag-api"},
)

with propagate_attributes(
    customer_identifier="user_123",
    thread_identifier="conversation_abc",
    metadata={"plan": "pro"},
):
    result = pipeline.run(
        {"prompt_builder": {"question": "What is retrieval-augmented generation?"}}
    )

respan.flush()
```

## Route LLM calls through the Respan gateway

Haystack's OpenAI-compatible generator can use the Respan gateway by pointing
the OpenAI client configuration at Respan.

```bash
export RESPAN_API_KEY="YOUR_RESPAN_API_KEY"
export OPENAI_API_KEY="$RESPAN_API_KEY"
export OPENAI_BASE_URL="https://api.respan.ai/api"
```

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

pipeline = Pipeline()
pipeline.add_component(
    "prompt_builder",
    PromptBuilder(template="Answer the following question: {{ question }}"),
)
pipeline.add_component("generator", OpenAIGenerator(model="gpt-5-mini"))
pipeline.connect("prompt_builder", "generator")

result = pipeline.run(
    {"prompt_builder": {"question": "What is the capital of France?"}}
)
print(result["generator"]["replies"][0])
```

Change the `model` value to switch gateway-routed models.

## Use Respan prompt management

For managed prompts, pass prompt configuration in the request body with
`generation_kwargs.extra_body.prompt`. Haystack still requires a non-empty
message, but the Respan gateway resolves and applies the managed prompt.

```python
import os

from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor

prompt_id = os.environ["RESPAN_PROMPT_ID"]
prompt_variables = {
    "question": "Who created Python?",
    "context": "Python was created by Guido van Rossum and first released in 1991.",
}

respan = Respan(instrumentations=[HaystackInstrumentor()])

pipeline = Pipeline()
pipeline.add_component(
    "managed_prompt_llm",
    OpenAIChatGenerator(model=os.getenv("RESPAN_MODEL", "gpt-4o-mini")),
)

with respan.propagate_attributes(
    prompt={"prompt_id": prompt_id, "variables": prompt_variables}
):
    result = pipeline.run(
        {
            "managed_prompt_llm": {
                "messages": [ChatMessage.from_user("Run the managed Respan prompt.")],
                "generation_kwargs": {
                    "temperature": 0.0,
                    "extra_body": {
                        "prompt": {
                            "prompt_id": prompt_id,
                            "variables": prompt_variables,
                            "override": True,
                        },
                    },
                },
            }
        }
    )

print(result["managed_prompt_llm"]["replies"][0].text)
respan.flush()
```

## Resources

- [Respan Haystack tracing docs](https://www.respan.ai/docs/integrations/haystack)
- [Respan Haystack gateway docs](https://www.respan.ai/docs/integrations/gateway/haystack)
- [Runnable Haystack examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/haystack)
- [Respan platform](https://platform.respan.ai)
