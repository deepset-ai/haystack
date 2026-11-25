---
title: "Classifiers"
id: classifiers-api
description: "Classifiers answer questions about documents and store the answers in their metadata."
slug: "/classifiers-api"
---


## openai_decisions_document_classifier

### OpenAIDecisionsDocumentClassifier

Answers typed questions about each document with an OpenAI Decisions model and stores the answers in its metadata.

Each question has one of three types:

- `choice`: picks one value from `choices`, a list of objects with a `value` and an optional `description`.
- `score`: rates the text on the ordered `levels`, a list of objects with a `label` and an optional `description`.
- `predicate`: returns the probability that the yes/no question in `instructions` is true.

Answers are stored under `meta[metadata_field]`, keyed by question name. Questions are dicts in the OpenAI format.
See the [OpenAI documentation](https://developers.openai.com/api/docs/guides/decisions) for the full question and
answer format. A refusal is stored with `type="refusal"`, preserving the answers to the other questions.

Documents without text to classify are returned in `failed_documents` with the reason in
`meta["classification_error"]`. So are documents whose request failed, unless `raise_on_failure` is `True`.

### Usage example

```python
from haystack import Document
from haystack.components.classifiers import OpenAIDecisionsDocumentClassifier

# Set OPENAI_API_KEY in the environment.
classifier = OpenAIDecisionsDocumentClassifier(
    questions=[
        {
            "name": "department",
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "choices": [{"value": "billing"}, {"value": "technical"}, {"value": "sales"}],
        },
        {
            "name": "urgency",
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "levels": [{"label": "can wait"}, {"label": "this week"}, {"label": "today"}],
        },
        {"name": "refund", "type": "predicate", "instructions": "Is the customer asking for a refund?"},
    ]
)

result = classifier.run(documents=[Document(content="I was charged twice, please refund me today.")])
print(result["documents"][0].meta["openai_decisions"])
# Example response (values may vary):
# >> {
#     "department": {"type": "choice", "choice": "billing", "confidence": 0.95,
#      "probabilities": [{"value": "billing", "probability": 0.97},
#                        {"value": "technical", "probability": 0.02},
#                        {"value": "sales", "probability": 0.01}]},
#     "urgency": {"type": "score", "score": 1.95, "confidence": 0.94,
#      "probabilities": [{"label": "can wait", "value": 0, "probability": 0.01},
#                        {"label": "this week", "value": 1, "probability": 0.03},
#                        {"label": "today", "value": 2, "probability": 0.96}]},
#     "refund": {"type": "predicate", "probability": 0.95},
# }
print(result["failed_documents"])
# >> []
```

#### __init__

```python
__init__(
    questions: list[Question],
    *,
    model: str = "gpt-6-luna",
    api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    api_base_url: str | None = None,
    classification_field: str | None = None,
    metadata_field: str = "openai_decisions",
    timeout: float | None = None,
    max_retries: int | None = None,
    max_workers: int = 3,
    raise_on_failure: bool = False
) -> None
```

Creates an OpenAIDecisionsDocumentClassifier.

**Parameters:**

- **questions** (<code>list\[Question\]</code>) – Questions to answer for every document. Each item is a dict in the OpenAI format with a `type` (`choice`,
  `score` or `predicate`), `instructions`, and for `choice` and `score` questions, `choices` or `levels`,
  respectively. Each question must have a non-empty, unique string `name`, used as its answer's metadata key.
- **model** (<code>str</code>) – The name of the Decisions model to use.
- **api_key** (<code>Secret</code>) – The OpenAI API key. You can set it with an environment variable `OPENAI_API_KEY`, or pass with this
  parameter during initialization.
- **api_base_url** (<code>str | None</code>) – Overrides the default base URL for all HTTP requests.
- **classification_field** (<code>str | None</code>) – Name of the document's metadata field to classify. If `None`, `Document.content` is classified.
- **metadata_field** (<code>str</code>) – Name of the metadata field the answers are written to.
- **timeout** (<code>float | None</code>) – Timeout for OpenAI client calls. If not set, it defaults to either the
  `OPENAI_TIMEOUT` environment variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact OpenAI after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **max_workers** (<code>int</code>) – Maximum number of worker threads in `run` and of concurrent requests in `run_async`. Must be at least 1.
- **raise_on_failure** (<code>bool</code>) – If `True`, the first failed request raises its error. If `False`, documents whose request failed are
  returned in `failed_documents`.

**Raises:**

- <code>ValueError</code> – If `questions` is empty, a question has a missing, empty, non-string or duplicate `name`, an unknown
  `type`, a `choice` or `score` question has no `choices` or `levels`, respectively, or `max_workers` is
  smaller than 1.

#### warm_up

```python
warm_up() -> None
```

Creates the synchronous OpenAI client.

#### warm_up_async

```python
warm_up_async() -> None
```

Creates the asynchronous OpenAI client.

#### close

```python
close() -> None
```

Releases the synchronous OpenAI client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous OpenAI client.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Answers the questions for each document and adds the answers to its metadata.

Each document is sent as its own request, with up to `max_workers` requests running at once.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to classify.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: The classified documents. Each has a `metadata_field` entry mapping each question name
  to its answer.
- `failed_documents`: The documents that have no text to classify or whose request failed, with the
  reason in `meta["classification_error"]`.

**Raises:**

- <code>APIError</code> – If a request fails and `raise_on_failure` is `True`.

#### run_async

```python
run_async(documents: list[Document]) -> dict[str, list[Document]]
```

Asynchronously answers the questions for each document and adds the answers to its metadata.

This is the asynchronous version of the `run` method with the same parameters and return values. Up to
`max_workers` requests run concurrently.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to classify.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: The classified documents. Each has a `metadata_field` entry mapping each question name
  to its answer.
- `failed_documents`: The documents that have no text to classify or whose request failed, with the
  reason in `meta["classification_error"]`.

**Raises:**

- <code>APIError</code> – If a request fails and `raise_on_failure` is `True`.
