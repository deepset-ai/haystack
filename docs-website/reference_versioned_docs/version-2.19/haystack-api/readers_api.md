---
title: Readers
id: readers-api
description: Takes a query and a set of Documents as input and returns ExtractedAnswers by selecting a text span within the Documents.
---

<a id="extractive"></a>

# Module extractive

<a id="extractive.ExtractiveReader"></a>

## ExtractiveReader

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

<a id="extractive.ExtractiveReader.__init__"></a>

#### ExtractiveReader.\_\_init\_\_

```python
def __init__(model: Union[Path, str] = "deepset/roberta-base-squad2-distilled",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             top_k: int = 20,
             score_threshold: Optional[float] = None,
             max_seq_length: int = 384,
             stride: int = 128,
             max_batch_size: Optional[int] = None,
             answers_per_seq: Optional[int] = None,
             no_answer: bool = True,
             calibration_factor: float = 0.1,
             overlap_threshold: Optional[float] = 0.01,
             model_kwargs: Optional[dict[str, Any]] = None) -> None
```

Creates an instance of ExtractiveReader.

**Arguments**:

- `model`: A Hugging Face transformers question answering model.
Can either be a path to a folder containing the model files or an identifier for the Hugging Face hub.
- `device`: The device on which the model is loaded. If `None`, the default device is automatically selected.
- `token`: The API token used to download private models from Hugging Face.
- `top_k`: Number of answers to return per query. It is required even if score_threshold is set.
An additional answer with no text is returned if no_answer is set to True (default).
- `score_threshold`: Returns only answers with the probability score above this threshold.
- `max_seq_length`: Maximum number of tokens. If a sequence exceeds it, the sequence is split.
- `stride`: Number of tokens that overlap when sequence is split because it exceeds max_seq_length.
- `max_batch_size`: Maximum number of samples that are fed through the model at the same time.
- `answers_per_seq`: Number of answer candidates to consider per sequence.
This is relevant when a Document was split into multiple sequences because of max_seq_length.
- `no_answer`: Whether to return an additional `no answer` with an empty text and a score representing the
probability that the other top_k answers are incorrect.
- `calibration_factor`: Factor used for calibrating probabilities.
- `overlap_threshold`: If set this will remove duplicate answers if they have an overlap larger than the
supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
both of these answers could be kept if this variable is set to 0.24 or lower.
If None is provided then all answers are kept.
- `model_kwargs`: Additional keyword arguments passed to `AutoModelForQuestionAnswering.from_pretrained`
when loading the model specified in `model`. For details on what kwargs you can pass,
see the model's documentation.

<a id="extractive.ExtractiveReader.to_dict"></a>

#### ExtractiveReader.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="extractive.ExtractiveReader.from_dict"></a>

#### ExtractiveReader.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ExtractiveReader"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="extractive.ExtractiveReader.warm_up"></a>

#### ExtractiveReader.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="extractive.ExtractiveReader.deduplicate_by_overlap"></a>

#### ExtractiveReader.deduplicate\_by\_overlap

```python
def deduplicate_by_overlap(
        answers: list[ExtractedAnswer],
        overlap_threshold: Optional[float]) -> list[ExtractedAnswer]
```

De-duplicates overlapping Extractive Answers.

De-duplicates overlapping Extractive Answers from the same document based on how much the spans of the
answers overlap.

**Arguments**:

- `answers`: List of answers to be deduplicated.
- `overlap_threshold`: If set this will remove duplicate answers if they have an overlap larger than the
supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
both of these answers could be kept if this variable is set to 0.24 or lower.
If None is provided then all answers are kept.

**Returns**:

List of deduplicated answers.

<a id="extractive.ExtractiveReader.run"></a>

#### ExtractiveReader.run

```python
@component.output_types(answers=list[ExtractedAnswer])
def run(query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        max_seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        answers_per_seq: Optional[int] = None,
        no_answer: Optional[bool] = None,
        overlap_threshold: Optional[float] = None)
```

Locates and extracts answers from the given Documents using the given query.

**Arguments**:

- `query`: Query string.
- `documents`: List of Documents in which you want to search for an answer to the query.
- `top_k`: The maximum number of answers to return.
An additional answer is returned if no_answer is set to True (default).
- `score_threshold`: Returns only answers with the score above this threshold.
- `max_seq_length`: Maximum number of tokens. If a sequence exceeds it, the sequence is split.
- `stride`: Number of tokens that overlap when sequence is split because it exceeds max_seq_length.
- `max_batch_size`: Maximum number of samples that are fed through the model at the same time.
- `answers_per_seq`: Number of answer candidates to consider per sequence.
This is relevant when a Document was split into multiple sequences because of max_seq_length.
- `no_answer`: Whether to return no answer scores.
- `overlap_threshold`: If set this will remove duplicate answers if they have an overlap larger than the
supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
both of these answers could be kept if this variable is set to 0.24 or lower.
If None is provided then all answers are kept.

**Raises**:

- `RuntimeError`: If the component was not warmed up by calling 'warm_up()' before.

**Returns**:

List of answers sorted by (desc.) answer score.
