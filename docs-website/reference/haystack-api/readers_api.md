---
title: "Readers"
id: readers-api
description: "Takes a query and a set of Documents as input and returns ExtractedAnswers by selecting a text span within the Documents."
slug: "/readers-api"
---


## `ExtractiveReader`

Locates and extracts answers to a given query from Documents.

The ExtractiveReader component performs extractive question answering.
It assigns a score to every possible answer span independently of other answer spans.
This fixes a common issue of other implementations which make comparisons across documents harder by normalizing
each document's answers independently.

Example usage:

```python
from haystack import Document
from haystack.components.readers import ExtractiveReader

docs = [
    Document(content="Python is a popular programming language"),
    Document(content="python ist eine beliebte Programmiersprache"),
]

reader = ExtractiveReader()
reader.warm_up()

question = "What is a popular programming language?"
result = reader.run(query=question, documents=docs)
assert "Python" in result["answers"][0].data
```

### `__init__`

```python
__init__(model: Path | str = 'deepset/roberta-base-squad2-distilled', device: ComponentDevice | None = None, token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), top_k: int = 20, score_threshold: float | None = None, max_seq_length: int = 384, stride: int = 128, max_batch_size: int | None = None, answers_per_seq: int | None = None, no_answer: bool = True, calibration_factor: float = 0.1, overlap_threshold: float | None = 0.01, model_kwargs: dict[str, Any] | None = None) -> None
```

Creates an instance of ExtractiveReader.

**Parameters:**

- **model** (<code>Path | str</code>) – A Hugging Face transformers question answering model.
  Can either be a path to a folder containing the model files or an identifier for the Hugging Face hub.
- **device** (<code>ComponentDevice | None</code>) – The device on which the model is loaded. If `None`, the default device is automatically selected.
- **token** (<code>Secret | None</code>) – The API token used to download private models from Hugging Face.
- **top_k** (<code>int</code>) – Number of answers to return per query. It is required even if score_threshold is set.
  An additional answer with no text is returned if no_answer is set to True (default).
- **score_threshold** (<code>float | None</code>) – Returns only answers with the probability score above this threshold.
- **max_seq_length** (<code>int</code>) – Maximum number of tokens. If a sequence exceeds it, the sequence is split.
- **stride** (<code>int</code>) – Number of tokens that overlap when sequence is split because it exceeds max_seq_length.
- **max_batch_size** (<code>int | None</code>) – Maximum number of samples that are fed through the model at the same time.
- **answers_per_seq** (<code>int | None</code>) – Number of answer candidates to consider per sequence.
  This is relevant when a Document was split into multiple sequences because of max_seq_length.
- **no_answer** (<code>bool</code>) – Whether to return an additional `no answer` with an empty text and a score representing the
  probability that the other top_k answers are incorrect.
- **calibration_factor** (<code>float</code>) – Factor used for calibrating probabilities.
- **overlap_threshold** (<code>float | None</code>) – If set this will remove duplicate answers if they have an overlap larger than the
  supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
  one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
  However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
  both of these answers could be kept if this variable is set to 0.24 or lower.
  If None is provided then all answers are kept.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments passed to `AutoModelForQuestionAnswering.from_pretrained`
  when loading the model specified in `model`. For details on what kwargs you can pass,
  see the model's documentation.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ExtractiveReader
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ExtractiveReader</code> – Deserialized component.

### `warm_up`

```python
warm_up()
```

Initializes the component.

### `deduplicate_by_overlap`

```python
deduplicate_by_overlap(answers: list[ExtractedAnswer], overlap_threshold: float | None) -> list[ExtractedAnswer]
```

De-duplicates overlapping Extractive Answers.

De-duplicates overlapping Extractive Answers from the same document based on how much the spans of the
answers overlap.

**Parameters:**

- **answers** (<code>list\[ExtractedAnswer\]</code>) – List of answers to be deduplicated.
- **overlap_threshold** (<code>float | None</code>) – If set this will remove duplicate answers if they have an overlap larger than the
  supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
  one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
  However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
  both of these answers could be kept if this variable is set to 0.24 or lower.
  If None is provided then all answers are kept.

**Returns:**

- <code>list\[ExtractedAnswer\]</code> – List of deduplicated answers.

### `run`

```python
run(query: str, documents: list[Document], top_k: int | None = None, score_threshold: float | None = None, max_seq_length: int | None = None, stride: int | None = None, max_batch_size: int | None = None, answers_per_seq: int | None = None, no_answer: bool | None = None, overlap_threshold: float | None = None)
```

Locates and extracts answers from the given Documents using the given query.

**Parameters:**

- **query** (<code>str</code>) – Query string.
- **documents** (<code>list\[Document\]</code>) – List of Documents in which you want to search for an answer to the query.
- **top_k** (<code>int | None</code>) – The maximum number of answers to return.
  An additional answer is returned if no_answer is set to True (default).
- **score_threshold** (<code>float | None</code>) – Returns only answers with the score above this threshold.
- **max_seq_length** (<code>int | None</code>) – Maximum number of tokens. If a sequence exceeds it, the sequence is split.
- **stride** (<code>int | None</code>) – Number of tokens that overlap when sequence is split because it exceeds max_seq_length.
- **max_batch_size** (<code>int | None</code>) – Maximum number of samples that are fed through the model at the same time.
- **answers_per_seq** (<code>int | None</code>) – Number of answer candidates to consider per sequence.
  This is relevant when a Document was split into multiple sequences because of max_seq_length.
- **no_answer** (<code>bool | None</code>) – Whether to return no answer scores.
- **overlap_threshold** (<code>float | None</code>) – If set this will remove duplicate answers if they have an overlap larger than the
  supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
  one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
  However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
  both of these answers could be kept if this variable is set to 0.24 or lower.
  If None is provided then all answers are kept.

**Returns:**

- – List of answers sorted by (desc.) answer score.
