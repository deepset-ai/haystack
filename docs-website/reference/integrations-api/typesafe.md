---
title: "TypeSafe"
id: integrations-typesafe
description: "TypeSafe integration for Haystack"
slug: "/integrations-typesafe"
---


## haystack_integrations.components.classifiers.typesafe.document_classifier

### TypeSafeDocumentClassifier

Answers typed questions about each document with a TypeSafe System One model and stores the answers in its metadata.

System One models such as Jev don't generate text: they answer every question in one request and return
calibrated probabilities. Each question has one of three types:

- `choice`: picks one label from `criteria`, a dict of label to description (or `None`).
- `score`: rates the text on the ordered levels in `criteria`, a list of level descriptions.
- `noul`: returns the probability that the yes/no question in `instructions` is true.

Answers are stored under `meta[metadata_field]`, keyed by question ID. Questions can be dicts or the SDK's
`Choice`, `Score` and `Noul` objects. See the [TypeSafe documentation](https://docs.typesafe.ai/primitives) for the
full question and answer format.

Documents without text to classify are returned in `failed_documents` with the reason in
`meta["classification_error"]`. So are documents whose request failed, unless `raise_on_failure` is `True`.

The component also works with servers that implement the TypeSafe API, such as [Ollaya](https://ollaya.dev),
which runs open decision models locally. Set `api_base_url` to the server and `model` to one of its models.

### Usage example

```python
from haystack import Document

from haystack_integrations.components.classifiers.typesafe import TypeSafeDocumentClassifier

classifier = TypeSafeDocumentClassifier(
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {"billing": None, "technical": None, "sales": None},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["can wait", "this week", "today"],
        },
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    }
)

result = classifier.run(documents=[Document(content="I was charged twice, please refund me today.")])
print(result["documents"][0].meta["typesafe"])
# {'department': {'type': 'choice', 'choice': 'billing', 'confidence': 0.9516,
#                 'probabilities': {'billing': 0.9677, 'technical': 0.0221, 'sales': 0.0102}},
#  'urgency': {'type': 'score', 'score': 1.948, 'confidence': 0.9389,
#              'legend': {'0': 'can wait', '1': 'this week', '2': 'today'},
#              'probabilities': {'0': 0.0113, '1': 0.0295, '2': 0.9593}},
#  'refund': {'type': 'noul', 'noul': 0.9498}}
print(result["failed_documents"])
# []
```

#### __init__

```python
__init__(
    questions: dict[str, Question],
    *,
    model: str = "jev-latest",
    api_key: Secret = Secret.from_env_var("TYPESAFE_API_KEY"),
    api_base_url: str | None = None,
    classification_field: str | None = None,
    metadata_field: str = "typesafe",
    timeout: float | None = None,
    max_retries: int | None = None,
    max_workers: int = 3,
    raise_on_failure: bool = False
) -> None
```

Creates a TypeSafeDocumentClassifier.

**Parameters:**

- **questions** (<code>dict\[str, Question\]</code>) – Questions to answer for every document, keyed by question ID. Each value is a dict or an SDK `Choice`,
  `Score` or `Noul` object with a `type` (`choice`, `score` or `noul`), `instructions`, and for `choice` and
  `score` questions, `criteria`.
- **model** (<code>str</code>) – Name of the System One model to use.
- **api_key** (<code>Secret</code>) – The TypeSafe API key. Servers such as Ollaya accept any non-empty value unless they are configured
  with a key.
- **api_base_url** (<code>str | None</code>) – Base URL of the API. If `None`, the `TYPESAFE_BASE_URL` environment variable is used, or
  `https://api.typesafe.ai` if it is unset.
- **classification_field** (<code>str | None</code>) – Name of the document's metadata field to classify. If `None`, `Document.content` is classified.
- **metadata_field** (<code>str</code>) – Name of the metadata field the answers are written to.
- **timeout** (<code>float | None</code>) – Timeout in seconds for each request attempt. If `None`, the SDK default of 10 seconds is used.
- **max_retries** (<code>int | None</code>) – Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
- **max_workers** (<code>int</code>) – Maximum number of worker threads in `run` and of concurrent requests in `run_async`. Must be at least 1.
- **raise_on_failure** (<code>bool</code>) – If `True`, the first failed request raises its error. If `False`, documents whose request failed are
  returned in `failed_documents`.

**Raises:**

- <code>ValueError</code> – If `questions` is empty, a question has an unknown `type` or is a `Box` question, a `choice` or `score`
  question has no `criteria`, or `max_workers` is smaller than 1.

#### warm_up

```python
warm_up() -> None
```

Creates the synchronous TypeSafe client.

#### warm_up_async

```python
warm_up_async() -> None
```

Creates the asynchronous TypeSafe client.

#### close

```python
close() -> None
```

Releases the synchronous TypeSafe client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous TypeSafe client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

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
- `documents`: The classified documents. Each has a `metadata_field` entry mapping each question ID to its
  answer.
- `failed_documents`: The documents that have no text to classify or whose request failed, with the
  reason in `meta["classification_error"]`.

**Raises:**

- <code>TypeSafeError</code> – If a request fails and `raise_on_failure` is `True`.

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
- `documents`: The classified documents. Each has a `metadata_field` entry mapping each question ID to its
  answer.
- `failed_documents`: The documents that have no text to classify or whose request failed, with the
  reason in `meta["classification_error"]`.

**Raises:**

- <code>TypeSafeError</code> – If a request fails and `raise_on_failure` is `True`.

## haystack_integrations.components.rankers.typesafe.ranker

### TypeSafeRanker

Ranks documents by relevance to a query with a TypeSafe System One model such as Jev.

Each document is judged on its own with a `score` question, in one request per document. The model sees the query
and the passage as a JSON object, `{"query": ..., "passage": ...}`, so `instructions` and `criteria` can refer to
both. The score is the model's expected level, divided by the highest level, so it is a relevance from 0 to 1.

Because each document is a separate request, latency and token usage grow with the number of documents. Place
the ranker after a retriever with a small `top_k`, and tune `max_workers` to the server's capacity.

The ranker also reports what the requests cost. Its `meta` output holds the model and the summed `usage`, in the
same shape generators use, so that tools reading usage from pipeline traces can pick it up.

Documents whose request failed are kept after every scored document, with the reason in `meta["ranking_error"]`,
unless `raise_on_failure` is `True`. They are only part of the output if fewer than `top_k` documents were scored.

The component also works with servers that implement the TypeSafe API, such as [Ollaya](https://ollaya.dev).
Set `api_base_url` to the server and `model` to one of its models.

### Usage example

```python
from haystack import Document

from haystack_integrations.components.rankers.typesafe import TypeSafeRanker

ranker = TypeSafeRanker(top_k=2)
result = ranker.run(
    query="How do I reset my password?",
    documents=[
        Document(content="Click 'Forgot password' on the login page to reset it."),
        Document(content="Our offices are closed on public holidays."),
    ],
)
print(result["documents"][0].content)
# Click 'Forgot password' on the login page to reset it.
print(result["meta"])
# {'model': 'jev-latest', 'usage': {'input_tokens': 96, 'output_tokens': 0}}
```

#### __init__

```python
__init__(
    *,
    model: str = "jev-latest",
    api_key: Secret = Secret.from_env_var("TYPESAFE_API_KEY"),
    api_base_url: str | None = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
    criteria: list[str] | None = None,
    meta_fields_to_embed: list[str] | None = None,
    max_chars: int | None = None,
    metadata_field: str = "typesafe",
    top_k: int = 10,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_workers: int = 3,
    raise_on_failure: bool = False
) -> None
```

Creates a TypeSafeRanker.

**Parameters:**

- **model** (<code>str</code>) – Name of the System One model to use.
- **api_key** (<code>Secret</code>) – The TypeSafe API key. Servers such as Ollaya accept any non-empty value unless they are configured
  with a key.
- **api_base_url** (<code>str | None</code>) – Base URL of the API. If `None`, the `TYPESAFE_BASE_URL` environment variable is used, or
  `https://api.typesafe.ai` if it is unset.
- **instructions** (<code>str</code>) – The question the model answers about each document. It can refer to the `query` and the `passage`.
- **criteria** (<code>list\[str\] | None</code>) – The ordered score levels, from the least to the most relevant, each described by a concrete situation. At
  least two. Levels are judged independently and never see their index. If `None`, five generic levels are
  used.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – Names of metadata fields to show the model next to the text, under their own names in the passage. Fields a
  document doesn't have are left out. Named like the other rankers' parameter, but nothing is embedded:
  the fields are sent as structured keys next to `text`, not concatenated into it.
- **max_chars** (<code>int | None</code>) – Truncate each document's text to this many characters. If `None`, the text is sent in full.
- **metadata_field** (<code>str</code>) – Name of the metadata field the raw answer is written to, under the `relevance` key. Answers already in
  this field, such as those of a `TypeSafeDocumentClassifier` using the same field, are kept.
- **top_k** (<code>int</code>) – Maximum number of documents to return. Can be overridden in `run`.
- **timeout** (<code>float | None</code>) – Timeout in seconds for each request attempt. If `None`, the SDK default of 10 seconds is used.
- **max_retries** (<code>int | None</code>) – Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
- **max_workers** (<code>int</code>) – Maximum number of worker threads in `run` and of concurrent requests in `run_async`. Must be at least 1.
- **raise_on_failure** (<code>bool</code>) – If `True`, the first failed request raises its error. If `False`, documents whose request failed are
  ranked last.

**Raises:**

- <code>ValueError</code> – If `criteria` has fewer than two levels, `top_k` is smaller than 1, `max_chars` is smaller than 1, or
  `max_workers` is smaller than 1.

#### warm_up

```python
warm_up() -> None
```

Creates the synchronous TypeSafe client.

#### warm_up_async

```python
warm_up_async() -> None
```

Creates the asynchronous TypeSafe client.

#### close

```python
close() -> None
```

Releases the synchronous TypeSafe client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous TypeSafe client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### run

```python
run(
    query: str, documents: list[Document], top_k: int | None = None
) -> dict[str, Any]
```

Ranks the documents by their relevance to the query.

Each document is sent as its own request, with up to `max_workers` requests running at once.

**Parameters:**

- **query** (<code>str</code>) – The query to rank the documents against.
- **documents** (<code>list\[Document\]</code>) – Documents to rank.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return. If `None`, the `top_k` set at initialization is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: Up to `top_k` documents, most relevant first. Each has its relevance from 0 to 1 as `score`
  and the raw answer under `meta[metadata_field]["relevance"]`.
- `meta`: The `model` and the `usage` (`input_tokens` and `output_tokens`) summed over all requests.

**Raises:**

- <code>ValueError</code> – If `top_k` is smaller than 1.
- <code>TypeSafeError</code> – If a request fails and `raise_on_failure` is `True`.
- <code>TypeError</code> – If the server answers the relevance question with something other than a `score` answer.

#### run_async

```python
run_async(
    query: str, documents: list[Document], top_k: int | None = None
) -> dict[str, Any]
```

Asynchronously ranks the documents by their relevance to the query.

This is the asynchronous version of the `run` method with the same parameters and return values. Up to
`max_workers` requests run concurrently.

**Parameters:**

- **query** (<code>str</code>) – The query to rank the documents against.
- **documents** (<code>list\[Document\]</code>) – Documents to rank.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return. If `None`, the `top_k` set at initialization is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the `documents` and `meta` keys, as in `run`.

**Raises:**

- <code>ValueError</code> – If `top_k` is smaller than 1.
- <code>TypeSafeError</code> – If a request fails and `raise_on_failure` is `True`.
- <code>TypeError</code> – If the server answers the relevance question with something other than a `score` answer.

## haystack_integrations.components.routers.typesafe.text_router

### TypeSafeTextRouter

Routes a text to the connection of the label a TypeSafe System One model picks for it.

The model answers a single `choice` question over `labels` and returns calibrated probabilities. The component
has one output per label. When `min_confidence` is set, it also has a `low_confidence` output that receives the
text whenever the answer's `confidence` is below the threshold, so uncertain texts can go to a fallback such as
an LLM or a human. See [confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing).

The component also works with servers that implement the TypeSafe API, such as [Ollaya](https://ollaya.dev),
which runs open decision models locally. Set `api_base_url` to the server and `model` to one of its models.

### Usage example

```python
from haystack_integrations.components.routers.typesafe import TypeSafeTextRouter

router = TypeSafeTextRouter(
    labels={
        "billing": "Payments, invoices, refunds and charges",
        "technical": "Bugs, crashes and errors in the product",
    },
    instructions="Which team should handle this support request?",
    min_confidence=0.6,
)

print(router.run(text="I was charged twice for my subscription."))
# {'billing': 'I was charged twice for my subscription.'}
```

#### __init__

```python
__init__(
    labels: list[str] | dict[str, str | None],
    *,
    instructions: str = "Which category does this text belong to?",
    model: str = "jev-latest",
    api_key: Secret = Secret.from_env_var("TYPESAFE_API_KEY"),
    api_base_url: str | None = None,
    min_confidence: float | None = None,
    timeout: float | None = None,
    max_retries: int | None = None
) -> None
```

Creates a TypeSafeTextRouter.

**Parameters:**

- **labels** (<code>list\[str\] | dict\[str, str | None\]</code>) – The labels to route between, each becoming an output connection. Pass a dict of label to description to
  tell the model what each label means.
- **instructions** (<code>str</code>) – The question the model answers by picking one of the labels.
- **model** (<code>str</code>) – Name of the System One model to use.
- **api_key** (<code>Secret</code>) – The TypeSafe API key. Servers such as Ollaya accept any non-empty value unless they are configured
  with a key.
- **api_base_url** (<code>str | None</code>) – Base URL of the API. If `None`, the `TYPESAFE_BASE_URL` environment variable is used, or
  `https://api.typesafe.ai` if it is unset.
- **min_confidence** (<code>float | None</code>) – Threshold between 0 and 1 for the answer's `confidence`. Texts below it are sent to the `low_confidence`
  output. If `None`, every text goes to a label output.
- **timeout** (<code>float | None</code>) – Timeout in seconds for each request attempt. If `None`, the SDK default of 10 seconds is used.
- **max_retries** (<code>int | None</code>) – Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.

**Raises:**

- <code>ValueError</code> – If `labels` is empty or contains `low_confidence`, or if `min_confidence` is outside 0 to 1.

#### warm_up

```python
warm_up() -> None
```

Creates the synchronous TypeSafe client.

#### warm_up_async

```python
warm_up_async() -> None
```

Creates the asynchronous TypeSafe client.

#### close

```python
close() -> None
```

Releases the synchronous TypeSafe client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous TypeSafe client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### run

```python
run(text: str) -> dict[str, str]
```

Routes the text to the output of the label the model picks.

**Parameters:**

- **text** (<code>str</code>) – The text to route.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with a single key, the picked label or `low_confidence`, mapped to the input text.

**Raises:**

- <code>TypeError</code> – If `text` is not a string.

#### run_async

```python
run_async(text: str) -> dict[str, str]
```

Asynchronously routes the text to the output of the label the model picks.

This is the asynchronous version of the `run` method with the same parameters and return values.

**Parameters:**

- **text** (<code>str</code>) – The text to route.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with a single key, the picked label or `low_confidence`, mapped to the input text.

**Raises:**

- <code>TypeError</code> – If `text` is not a string.
