---
title: Joiners
id: joiners-api
description: Components that join list of different objects
---

<a id="answer_joiner"></a>

# Module answer\_joiner

<a id="answer_joiner.JoinMode"></a>

## JoinMode

Enum for AnswerJoiner join modes.

<a id="answer_joiner.JoinMode.from_str"></a>

#### JoinMode.from\_str

```python
@staticmethod
def from_str(string: str) -> "JoinMode"
```

Convert a string to a JoinMode enum.

<a id="answer_joiner.AnswerJoiner"></a>

## AnswerJoiner

Merges multiple lists of `Answer` objects into a single list.

Use this component to combine answers from different Generators into a single list.
Currently, the component supports only one join mode: `CONCATENATE`.
This mode concatenates multiple lists of answers into a single list.

### Usage example

In this example, AnswerJoiner merges answers from two different Generators:

```python
from haystack.components.builders import AnswerBuilder
from haystack.components.joiners import AnswerJoiner

from haystack.core.pipeline import Pipeline

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage


query = "What's Natural Language Processing?"
messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant. Be super concise."),
            ChatMessage.from_user(query)]

pipe = Pipeline()
pipe.add_component("gpt-4o", OpenAIChatGenerator(model="gpt-4o"))
pipe.add_component("gpt-4o-mini", OpenAIChatGenerator(model="gpt-4o-mini"))
pipe.add_component("aba", AnswerBuilder())
pipe.add_component("abb", AnswerBuilder())
pipe.add_component("joiner", AnswerJoiner())

pipe.connect("gpt-4o.replies", "aba")
pipe.connect("gpt-4o-mini.replies", "abb")
pipe.connect("aba.answers", "joiner")
pipe.connect("abb.answers", "joiner")

results = pipe.run(data={"gpt-4o": {"messages": messages},
                            "gpt-4o-mini": {"messages": messages},
                            "aba": {"query": query},
                            "abb": {"query": query}})
```

<a id="answer_joiner.AnswerJoiner.__init__"></a>

#### AnswerJoiner.\_\_init\_\_

```python
def __init__(join_mode: Union[str, JoinMode] = JoinMode.CONCATENATE,
             top_k: Optional[int] = None,
             sort_by_score: bool = False)
```

Creates an AnswerJoiner component.

**Arguments**:

- `join_mode`: Specifies the join mode to use. Available modes:
- `concatenate`: Concatenates multiple lists of Answers into a single list.
- `top_k`: The maximum number of Answers to return.
- `sort_by_score`: If `True`, sorts the documents by score in descending order.
If a document has no score, it is handled as if its score is -infinity.

<a id="answer_joiner.AnswerJoiner.run"></a>

#### AnswerJoiner.run

```python
@component.output_types(answers=list[AnswerType])
def run(answers: Variadic[list[AnswerType]], top_k: Optional[int] = None)
```

Joins multiple lists of Answers into a single list depending on the `join_mode` parameter.

**Arguments**:

- `answers`: Nested list of Answers to be merged.
- `top_k`: The maximum number of Answers to return. Overrides the instance's `top_k` if provided.

**Returns**:

A dictionary with the following keys:
- `answers`: Merged list of Answers

<a id="answer_joiner.AnswerJoiner.to_dict"></a>

#### AnswerJoiner.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="answer_joiner.AnswerJoiner.from_dict"></a>

#### AnswerJoiner.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "AnswerJoiner"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="branch"></a>

# Module branch

<a id="branch.BranchJoiner"></a>

## BranchJoiner

A component that merges multiple input branches of a pipeline into a single output stream.

`BranchJoiner` receives multiple inputs of the same data type and forwards the first received value
to its output. This is useful for scenarios where multiple branches need to converge before proceeding.

### Common Use Cases:
- **Loop Handling:** `BranchJoiner` helps close loops in pipelines. For example, if a pipeline component validates
  or modifies incoming data and produces an error-handling branch, `BranchJoiner` can merge both branches and send
  (or resend in the case of a loop) the data to the component that evaluates errors. See "Usage example" below.

- **Decision-Based Merging:** `BranchJoiner` reconciles branches coming from Router components (such as
  `ConditionalRouter`, `TextLanguageRouter`). Suppose a `TextLanguageRouter` directs user queries to different
  Retrievers based on the detected language. Each Retriever processes its assigned query and passes the results
  to `BranchJoiner`, which consolidates them into a single output before passing them to the next component, such
  as a `PromptBuilder`.

### Example Usage:
```python
import json

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage

# Define a schema for validation
person_schema = {
    "type": "object",
    "properties": {
        "first_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
        "last_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
        "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "American"]},
    },
    "required": ["first_name", "last_name", "nationality"]
}

# Initialize a pipeline
pipe = Pipeline()

# Add components to the pipeline
pipe.add_component('joiner', BranchJoiner(list[ChatMessage]))
pipe.add_component('generator', OpenAIChatGenerator(model="gpt-4o-mini"))
pipe.add_component('validator', JsonSchemaValidator(json_schema=person_schema))
pipe.add_component('adapter', OutputAdapter("{{chat_message}}", list[ChatMessage], unsafe=True))

# And connect them
pipe.connect("adapter", "joiner")
pipe.connect("joiner", "generator")
pipe.connect("generator.replies", "validator.messages")
pipe.connect("validator.validation_error", "joiner")

result = pipe.run(
    data={
    "generator": {"generation_kwargs": {"response_format": {"type": "json_object"}}},
    "adapter": {"chat_message": [ChatMessage.from_user("Create json from Peter Parker")]}}
)

print(json.loads(result["validator"]["validated"][0].text))


>> {'first_name': 'Peter', 'last_name': 'Parker', 'nationality': 'American', 'name': 'Spider-Man', 'occupation':
>> 'Superhero', 'age': 23, 'location': 'New York City'}
```

Note that `BranchJoiner` can manage only one data type at a time. In this case, `BranchJoiner` is created for
passing `list[ChatMessage]`. This determines the type of data that `BranchJoiner` will receive from the upstream
connected components and also the type of data that `BranchJoiner` will send through its output.

In the code example, `BranchJoiner` receives a looped back `list[ChatMessage]` from the `JsonSchemaValidator` and
sends it down to the `OpenAIChatGenerator` for re-generation. We can have multiple loopback connections in the
pipeline. In this instance, the downstream component is only one (the `OpenAIChatGenerator`), but the pipeline could
have more than one downstream component.

<a id="branch.BranchJoiner.__init__"></a>

#### BranchJoiner.\_\_init\_\_

```python
def __init__(type_: type)
```

Creates a `BranchJoiner` component.

**Arguments**:

- `type_`: The expected data type of inputs and outputs.

<a id="branch.BranchJoiner.to_dict"></a>

#### BranchJoiner.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component into a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="branch.BranchJoiner.from_dict"></a>

#### BranchJoiner.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "BranchJoiner"
```

Deserializes a `BranchJoiner` instance from a dictionary.

**Arguments**:

- `data`: The dictionary containing serialized component data.

**Returns**:

A deserialized `BranchJoiner` instance.

<a id="branch.BranchJoiner.run"></a>

#### BranchJoiner.run

```python
def run(**kwargs) -> dict[str, Any]
```

Executes the `BranchJoiner`, selecting the first available input value and passing it downstream.

**Arguments**:

- `**kwargs`: The input data. Must be of the type declared by `type_` during initialization.

**Returns**:

A dictionary with a single key `value`, containing the first input received.

<a id="document_joiner"></a>

# Module document\_joiner

<a id="document_joiner.JoinMode"></a>

## JoinMode

Enum for join mode.

<a id="document_joiner.JoinMode.from_str"></a>

#### JoinMode.from\_str

```python
@staticmethod
def from_str(string: str) -> "JoinMode"
```

Convert a string to a JoinMode enum.

<a id="document_joiner.DocumentJoiner"></a>

## DocumentJoiner

Joins multiple lists of documents into a single list.

It supports different join modes:
- concatenate: Keeps the highest-scored document in case of duplicates.
- merge: Calculates a weighted sum of scores for duplicates and merges them.
- reciprocal_rank_fusion: Merges and assigns scores based on reciprocal rank fusion.
- distribution_based_rank_fusion: Merges and assigns scores based on scores distribution in each Retriever.

### Usage example:

```python
from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
docs = [Document(content="Paris"), Document(content="Berlin"), Document(content="London")]
embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedder.warm_up()
docs_embeddings = embedder.run(docs)
document_store.write_documents(docs_embeddings['documents'])

p = Pipeline()
p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
p.add_component(
        instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        name="text_embedder",
    )
p.add_component(instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever")
p.add_component(instance=DocumentJoiner(), name="joiner")
p.connect("bm25_retriever", "joiner")
p.connect("embedding_retriever", "joiner")
p.connect("text_embedder", "embedding_retriever")
query = "What is the capital of France?"
p.run(data={"query": query, "text": query, "top_k": 1})
```

<a id="document_joiner.DocumentJoiner.__init__"></a>

#### DocumentJoiner.\_\_init\_\_

```python
def __init__(join_mode: Union[str, JoinMode] = JoinMode.CONCATENATE,
             weights: Optional[list[float]] = None,
             top_k: Optional[int] = None,
             sort_by_score: bool = True)
```

Creates a DocumentJoiner component.

**Arguments**:

- `join_mode`: Specifies the join mode to use. Available modes:
- `concatenate`: Keeps the highest-scored document in case of duplicates.
- `merge`: Calculates a weighted sum of scores for duplicates and merges them.
- `reciprocal_rank_fusion`: Merges and assigns scores based on reciprocal rank fusion.
- `distribution_based_rank_fusion`: Merges and assigns scores based on scores
distribution in each Retriever.
- `weights`: Assign importance to each list of documents to influence how they're joined.
This parameter is ignored for
`concatenate` or `distribution_based_rank_fusion` join modes.
Weight for each list of documents must match the number of inputs.
- `top_k`: The maximum number of documents to return.
- `sort_by_score`: If `True`, sorts the documents by score in descending order.
If a document has no score, it is handled as if its score is -infinity.

<a id="document_joiner.DocumentJoiner.run"></a>

#### DocumentJoiner.run

```python
@component.output_types(documents=list[Document])
def run(documents: Variadic[list[Document]], top_k: Optional[int] = None)
```

Joins multiple lists of Documents into a single list depending on the `join_mode` parameter.

**Arguments**:

- `documents`: List of list of documents to be merged.
- `top_k`: The maximum number of documents to return. Overrides the instance's `top_k` if provided.

**Returns**:

A dictionary with the following keys:
- `documents`: Merged list of Documents

<a id="document_joiner.DocumentJoiner.to_dict"></a>

#### DocumentJoiner.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="document_joiner.DocumentJoiner.from_dict"></a>

#### DocumentJoiner.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "DocumentJoiner"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="list_joiner"></a>

# Module list\_joiner

<a id="list_joiner.ListJoiner"></a>

## ListJoiner

A component that joins multiple lists into a single flat list.

The ListJoiner receives multiple lists of the same type and concatenates them into a single flat list.
The output order respects the pipeline's execution sequence, with earlier inputs being added first.

Usage example:
```python
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack.components.joiners import ListJoiner


user_message = [ChatMessage.from_user("Give a brief answer the following question: {{query}}")]

feedback_prompt = """
    You are given a question and an answer.
    Your task is to provide a score and a brief feedback on the answer.
    Question: {{query}}
    Answer: {{response}}
    """
feedback_message = [ChatMessage.from_system(feedback_prompt)]

prompt_builder = ChatPromptBuilder(template=user_message)
feedback_prompt_builder = ChatPromptBuilder(template=feedback_message)
llm = OpenAIChatGenerator(model="gpt-4o-mini")
feedback_llm = OpenAIChatGenerator(model="gpt-4o-mini")

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.add_component("feedback_prompt_builder", feedback_prompt_builder)
pipe.add_component("feedback_llm", feedback_llm)
pipe.add_component("list_joiner", ListJoiner(list[ChatMessage]))

pipe.connect("prompt_builder.prompt", "llm.messages")
pipe.connect("prompt_builder.prompt", "list_joiner")
pipe.connect("llm.replies", "list_joiner")
pipe.connect("llm.replies", "feedback_prompt_builder.response")
pipe.connect("feedback_prompt_builder.prompt", "feedback_llm.messages")
pipe.connect("feedback_llm.replies", "list_joiner")

query = "What is nuclear physics?"
ans = pipe.run(data={"prompt_builder": {"template_variables":{"query": query}},
    "feedback_prompt_builder": {"template_variables":{"query": query}}})

print(ans["list_joiner"]["values"])
```

<a id="list_joiner.ListJoiner.__init__"></a>

#### ListJoiner.\_\_init\_\_

```python
def __init__(list_type_: Optional[type] = None)
```

Creates a ListJoiner component.

**Arguments**:

- `list_type_`: The expected type of the lists this component will join (e.g., list[ChatMessage]).
If specified, all input lists must conform to this type. If None, the component defaults to handling
lists of any type including mixed types.

<a id="list_joiner.ListJoiner.to_dict"></a>

#### ListJoiner.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="list_joiner.ListJoiner.from_dict"></a>

#### ListJoiner.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ListJoiner"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="list_joiner.ListJoiner.run"></a>

#### ListJoiner.run

```python
def run(values: Variadic[list[Any]]) -> dict[str, list[Any]]
```

Joins multiple lists into a single flat list.

**Arguments**:

- `values`: The list to be joined.

**Returns**:

Dictionary with 'values' key containing the joined list.

<a id="string_joiner"></a>

# Module string\_joiner

<a id="string_joiner.StringJoiner"></a>

## StringJoiner

Component to join strings from different components to a list of strings.

### Usage example

```python
from haystack.components.joiners import StringJoiner
from haystack.components.builders import PromptBuilder
from haystack.core.pipeline import Pipeline

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

string_1 = "What's Natural Language Processing?"
string_2 = "What is life?"

pipeline = Pipeline()
pipeline.add_component("prompt_builder_1", PromptBuilder("Builder 1: {{query}}"))
pipeline.add_component("prompt_builder_2", PromptBuilder("Builder 2: {{query}}"))
pipeline.add_component("string_joiner", StringJoiner())

pipeline.connect("prompt_builder_1.prompt", "string_joiner.strings")
pipeline.connect("prompt_builder_2.prompt", "string_joiner.strings")

print(pipeline.run(data={"prompt_builder_1": {"query": string_1}, "prompt_builder_2": {"query": string_2}}))

>> {"string_joiner": {"strings": ["Builder 1: What's Natural Language Processing?", "Builder 2: What is life?"]}}
```

<a id="string_joiner.StringJoiner.run"></a>

#### StringJoiner.run

```python
@component.output_types(strings=list[str])
def run(strings: Variadic[str])
```

Joins strings into a list of strings

**Arguments**:

- `strings`: strings from different components

**Returns**:

A dictionary with the following keys:
- `strings`: Merged list of strings
