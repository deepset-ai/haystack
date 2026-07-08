---
title: "Transformers"
id: integrations-transformers
description: "Transformers integration for Haystack"
slug: "/integrations-transformers"
---


## haystack_integrations.components.classifiers.transformers.zero_shot_document_classifier

### TransformersZeroShotDocumentClassifier

Performs zero-shot classification of documents based on given labels and adds the predicted label to their metadata.

The component uses a Hugging Face pipeline for zero-shot classification.
Provide the model and the set of labels to be used for categorization during initialization.
Additionally, you can configure the component to allow multiple labels to be true.

Classification is run on the document's content field by default. If you want it to run on another field, set the
`classification_field` to one of the document's metadata fields.

Available models for the task of zero-shot-classification include:
\- `valhalla/distilbart-mnli-12-3`
\- `cross-encoder/nli-distilroberta-base`
\- `cross-encoder/nli-deberta-v3-xsmall`

### Usage example

The following is a pipeline that classifies documents based on predefined classification labels
retrieved from a search pipeline:

```python
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.pipeline import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.classifiers.transformers import TransformersZeroShotDocumentClassifier

documents = [Document(id="0", content="Today was a nice day!"),
             Document(id="1", content="Yesterday was a bad day!")]

document_store = InMemoryDocumentStore()
retriever = InMemoryBM25Retriever(document_store=document_store)
document_classifier = TransformersZeroShotDocumentClassifier(
    model="cross-encoder/nli-deberta-v3-xsmall",
    labels=["positive", "negative"],
)

document_store.write_documents(documents)

pipeline = Pipeline()
pipeline.add_component(instance=retriever, name="retriever")
pipeline.add_component(instance=document_classifier, name="document_classifier")
pipeline.connect("retriever", "document_classifier")

queries = ["How was your day today?", "How was your day yesterday?"]
expected_predictions = ["positive", "negative"]

for idx, query in enumerate(queries):
    result = pipeline.run({"retriever": {"query": query, "top_k": 1}})
    assert result["document_classifier"]["documents"][0].to_dict()["id"] == str(idx)
    assert (result["document_classifier"]["documents"][0].to_dict()["classification"]["label"]
            == expected_predictions[idx])
```

#### __init__

```python
__init__(
    model: str,
    labels: list[str],
    multi_label: bool = False,
    classification_field: str | None = None,
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    huggingface_pipeline_kwargs: dict[str, Any] | None = None,
) -> None
```

Initializes the TransformersZeroShotDocumentClassifier.

See the Hugging Face [website](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads&search=nli)
for the full list of zero-shot classification models (NLI) models.

**Parameters:**

- **model** (<code>str</code>) – The name or path of a Hugging Face model for zero shot document classification.
- **labels** (<code>list\[str\]</code>) – The set of possible class labels to classify each document into, for example,
  ["positive", "negative"]. The labels depend on the selected model.
- **multi_label** (<code>bool</code>) – Whether or not multiple candidate labels can be true.
  If `False`, the scores are normalized such that
  the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
  independent and probabilities are normalized for each candidate by doing a softmax of the entailment
  score vs. the contradiction score.
- **classification_field** (<code>str | None</code>) – Name of document's meta field to be used for classification.
  If not set, `Document.content` is used by default.
- **device** (<code>ComponentDevice | None</code>) – The device on which the model is loaded. If `None`, the default device is automatically
  selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
- **token** (<code>Secret | None</code>) – The Hugging Face token to use as HTTP bearer authorization.
  Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
- **huggingface_pipeline_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary containing keyword arguments used to initialize the
  Hugging Face pipeline for text classification.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TransformersZeroShotDocumentClassifier
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TransformersZeroShotDocumentClassifier</code> – Deserialized component.

#### run

```python
run(documents: list[Document], batch_size: int = 1) -> dict[str, Any]
```

Classifies the documents based on the provided labels and adds them to their metadata.

The classification results are stored in the `classification` dict within
each document's metadata. If `multi_label` is set to `True`, the scores for each label are available under
the `details` key within the `classification` dictionary.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to process.
- **batch_size** (<code>int</code>) – Batch size used for processing the content in each document.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following key:
- `documents`: A list of documents with an added metadata field called `classification`.

## haystack_integrations.components.extractors.transformers.named_entity_extractor

### NamedEntityAnnotation

Describes a single NER annotation.

**Parameters:**

- **entity** (<code>str</code>) – Entity label.
- **start** (<code>int</code>) – Start index of the entity in the document.
- **end** (<code>int</code>) – End index of the entity in the document.
- **score** (<code>float | None</code>) – Score calculated by the model.

### TransformersNamedEntityExtractor

Annotates named entities in a collection of documents.

The component can be used with any token classification model from the
[Hugging Face model hub](https://huggingface.co/models). Annotations are
stored as metadata in the documents.

Usage example:

```python
from haystack import Document

from haystack_integrations.components.extractors.transformers import TransformersNamedEntityExtractor

documents = [
    Document(content="I'm Merlin, the happy pig!"),
    Document(content="My name is Clara and I live in Berkeley, California."),
]
extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")
results = extractor.run(documents=documents)["documents"]
annotations = [TransformersNamedEntityExtractor.get_stored_annotations(doc) for doc in results]
print(annotations)
```

#### __init__

```python
__init__(
    *,
    model: str,
    pipeline_kwargs: dict[str, Any] | None = None,
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    )
) -> None
```

Create a Named Entity extractor component.

**Parameters:**

- **model** (<code>str</code>) – Name of the model or a path to the model on
  the local disk.
- **pipeline_kwargs** (<code>dict\[str, Any\] | None</code>) – Keyword arguments passed to the pipeline. The
  pipeline can override these arguments.
- **device** (<code>ComponentDevice | None</code>) – The device on which the model is loaded. If `None`,
  the default device is automatically selected. If a
  device/device map is specified in `pipeline_kwargs`,
  it overrides this parameter.
- **token** (<code>Secret | None</code>) – The API token to download private models from Hugging Face.

#### warm_up

```python
warm_up() -> None
```

Initialize the component.

**Raises:**

- <code>ComponentError</code> – If the component fails to initialize successfully.

#### run

```python
run(documents: list[Document], batch_size: int = 1) -> dict[str, Any]
```

Annotate named entities in each document and store the annotations in the document's metadata.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to process.
- **batch_size** (<code>int</code>) – Batch size used for processing the documents.

**Returns:**

- <code>dict\[str, Any\]</code> – Processed documents.

**Raises:**

- <code>ComponentError</code> – If the model fails to process a document.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TransformersNamedEntityExtractor
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TransformersNamedEntityExtractor</code> – Deserialized component.

#### initialized

```python
initialized: bool
```

Returns if the extractor is ready to annotate text.

#### get_stored_annotations

```python
get_stored_annotations(
    document: Document,
) -> list[NamedEntityAnnotation] | None
```

Returns the document's named entity annotations stored in its metadata, if any.

**Parameters:**

- **document** (<code>Document</code>) – Document whose annotations are to be fetched.

**Returns:**

- <code>list\[NamedEntityAnnotation\] | None</code> – The stored annotations.

## haystack_integrations.components.generators.transformers.chat.chat_generator

### default_tool_parser

```python
default_tool_parser(text: str) -> list[ToolCall] | None
```

Default implementation for parsing tool calls from model output text.

Uses DEFAULT_TOOL_PATTERN to extract tool calls.

**Parameters:**

- **text** (<code>str</code>) – The text to parse for tool calls.

**Returns:**

- <code>list\[ToolCall\] | None</code> – A list containing a single ToolCall if a valid tool call is found, None otherwise.

### TransformersChatGenerator

Generates chat responses using models from Hugging Face that run locally.

Use this component with chat-based models,
such as `Qwen/Qwen3-0.6B` or `meta-llama/Llama-2-7b-chat-hf`.
LLMs running locally may need powerful hardware.

### Usage example

```python
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.transformers import TransformersChatGenerator

generator = TransformersChatGenerator(model="Qwen/Qwen3-0.6B")
messages = [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
print(generator.run(messages))
```

```
{'replies':
    [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
    "Natural Language Processing (NLP) is a subfield of artificial intelligence that deals
    with the interaction between computers and human language. It enables computers to understand, interpret, and
    generate human language in a valuable way. NLP involves various techniques such as speech recognition, text
    analysis, sentiment analysis, and machine translation. The ultimate goal is to make it easier for computers to
    process and derive meaning from human language, improving communication between humans and machines.")],
    _name=None,
    _meta={'finish_reason': 'stop', 'index': 0, 'model':
          'mistralai/Mistral-7B-Instruct-v0.2',
          'usage': {'completion_tokens': 90, 'prompt_tokens': 19, 'total_tokens': 109}})
          ]
}
```

#### __init__

```python
__init__(
    model: str = "Qwen/Qwen3-0.6B",
    task: (
        Literal["text-generation", "text2text-generation", "image-text-to-text"]
        | None
    ) = None,
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    chat_template: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    huggingface_pipeline_kwargs: dict[str, Any] | None = None,
    stop_words: list[str] | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    tools: ToolsType | None = None,
    tool_parsing_function: Callable[[str], list[ToolCall] | None] | None = None,
    async_executor: ThreadPoolExecutor | None = None,
    *,
    enable_thinking: bool = False
) -> None
```

Initializes the TransformersChatGenerator component.

**Parameters:**

- **model** (<code>str</code>) – The Hugging Face text generation model name or path,
  for example, `mistralai/Mistral-7B-Instruct-v0.2` or `TheBloke/OpenHermes-2.5-Mistral-7B-16k-AWQ`.
  The model must be a chat model supporting the ChatML messaging
  format.
  If the model is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
- **task** (<code>Literal['text-generation', 'text2text-generation', 'image-text-to-text'] | None</code>) – The task for the Hugging Face pipeline. Possible options:
- `text-generation`: Supported by decoder models, like GPT.
- `text2text-generation`: Deprecated as of Transformers v5; use `text-generation` instead.
  Previously supported by encoder-decoder models such as T5.
- `image-text-to-text`: Supported by vision-language models.
  If the task is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
  If not specified, the component calls the Hugging Face API to infer the task from the model name.
- **device** (<code>ComponentDevice | None</code>) – The device for loading the model. If `None`, automatically selects the default device.
  If a device or device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
- **token** (<code>Secret | None</code>) – The token to use as HTTP bearer authorization for remote files.
  If the token is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
- **chat_template** (<code>str | None</code>) – Specifies an optional Jinja template for formatting chat
  messages. Most high-quality chat models have their own templates, but for models without this
  feature or if you prefer a custom template, use this parameter.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary with keyword arguments to customize text generation.
  Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`.
  See Hugging Face's documentation for more information:
- - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
- - [GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
    The only `generation_kwargs` set by default is `max_new_tokens`, which is set to 512 tokens.
- **huggingface_pipeline_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary with keyword arguments to initialize the
  Hugging Face pipeline for text generation.
  These keyword arguments provide fine-grained control over the Hugging Face pipeline.
  In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
  For kwargs, see [Hugging Face documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task).
  In this dictionary, you can also include `model_kwargs` to specify the kwargs for [model initialization](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
- **stop_words** (<code>list\[str\] | None</code>) – A list of stop words. If the model generates a stop word, the generation stops.
  If you provide this parameter, don't specify the `stopping_criteria` in `generation_kwargs`.
  For some chat models, the output includes both the new text and the original prompt.
  In these cases, make sure your prompt has no stop words.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable for handling streaming responses.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
- **tool_parsing_function** (<code>Callable\\[[str\], list\[ToolCall\] | None\] | None</code>) – A callable that takes a string and returns a list of ToolCall objects or None.
  If None, the default_tool_parser will be used which extracts tool calls using a predefined pattern.
- **async_executor** (<code>ThreadPoolExecutor | None</code>) – Optional ThreadPoolExecutor to use for async calls. If not provided, a single-threaded executor will be
  initialized and used
- **enable_thinking** (<code>bool</code>) – Whether to enable thinking mode in the chat template for thinking-capable models.
  When enabled, the model generates intermediate reasoning before the final response. Defaults to False.

#### shutdown

```python
shutdown() -> None
```

Explicitly shutdown the executor if we own it.

#### warm_up

```python
warm_up() -> None
```

Initializes the component and warms up tools if provided.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TransformersChatGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>TransformersChatGenerator</code> – The deserialized component.

#### run

```python
run(
    messages: list[ChatMessage] | str,
    generation_kwargs: dict[str, Any] | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    tools: ToolsType | None = None,
) -> dict[str, list[ChatMessage]]
```

Invoke text generation inference based on the provided messages and generation parameters.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage objects representing the input messages. If a string is provided,
  it is converted to a list containing a ChatMessage with user role.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable for handling streaming responses.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter provided during initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: A list containing the generated responses as ChatMessage instances.

#### create_message

```python
create_message(
    text: str,
    index: int,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: str,
    generation_kwargs: dict[str, Any],
    parse_tool_calls: bool = False,
) -> ChatMessage
```

Create a ChatMessage instance from the provided text, populated with metadata.

**Parameters:**

- **text** (<code>str</code>) – The generated text.
- **index** (<code>int</code>) – The index of the generated text.
- **tokenizer** (<code>Union\[PreTrainedTokenizer, PreTrainedTokenizerFast\]</code>) – The tokenizer used for generation.
- **prompt** (<code>str</code>) – The prompt used for generation.
- **generation_kwargs** (<code>dict\[str, Any\]</code>) – The generation parameters.
- **parse_tool_calls** (<code>bool</code>) – Whether to attempt parsing tool calls from the text.

**Returns:**

- <code>ChatMessage</code> – A ChatMessage instance.

#### run_async

```python
run_async(
    messages: list[ChatMessage] | str,
    generation_kwargs: dict[str, Any] | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    tools: ToolsType | None = None,
) -> dict[str, list[ChatMessage]]
```

Asynchronously invokes text generation inference based on the provided messages and generation parameters.

This is the asynchronous version of the `run` method. It has the same parameters
and return values but can be used with `await` in an async code.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage objects representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable for handling streaming responses.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter provided during initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: A list containing the generated responses as ChatMessage instances.

## haystack_integrations.components.readers.transformers.extractive_reader

### TransformersExtractiveReader

Locates and extracts answers to a given query from Documents.

The TransformersExtractiveReader component performs extractive question answering.
It assigns a score to every possible answer span independently of other answer spans.
This fixes a common issue of other implementations which make comparisons across documents harder by normalizing
each document's answers independently.

Example usage:

```python
from haystack import Document

from haystack_integrations.components.readers.transformers import TransformersExtractiveReader

docs = [
    Document(content="Python is a popular programming language"),
    Document(content="python ist eine beliebte Programmiersprache"),
]

reader = TransformersExtractiveReader()

question = "What is a popular programming language?"
result = reader.run(query=question, documents=docs)
assert "Python" in result["answers"][0].data
```

#### __init__

```python
__init__(
    model: Path | str = "deepset/roberta-base-squad2-distilled",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    top_k: int = 20,
    score_threshold: float | None = None,
    max_seq_length: int = 384,
    stride: int = 128,
    max_batch_size: int | None = None,
    answers_per_seq: int | None = None,
    no_answer: bool = True,
    calibration_factor: float = 0.1,
    overlap_threshold: float | None = 0.01,
    model_kwargs: dict[str, Any] | None = None,
) -> None
```

Creates an instance of TransformersExtractiveReader.

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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TransformersExtractiveReader
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TransformersExtractiveReader</code> – Deserialized component.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### deduplicate_by_overlap

```python
deduplicate_by_overlap(
    answers: list[ExtractedAnswer], overlap_threshold: float | None
) -> list[ExtractedAnswer]
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

#### run

```python
run(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    score_threshold: float | None = None,
    max_seq_length: int | None = None,
    stride: int | None = None,
    max_batch_size: int | None = None,
    answers_per_seq: int | None = None,
    no_answer: bool | None = None,
    overlap_threshold: float | None = None,
) -> dict[str, Any]
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

- <code>dict\[str, Any\]</code> – List of answers sorted by (desc.) answer score.

## haystack_integrations.components.routers.transformers.text_router

### TransformersTextRouter

Routes the text strings to different connections based on a category label.

The labels are specific to each model and can be found it its description on Hugging Face.

### Usage example

```python
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.core.pipeline import Pipeline

from haystack_integrations.components.routers.transformers import TransformersTextRouter

p = Pipeline()
p.add_component(
    instance=TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection"),
    name="text_router"
)
p.add_component(
    instance=PromptBuilder(template="Answer the question: {{query}}\nAnswer:"),
    name="english_prompt_builder"
)
p.add_component(
    instance=PromptBuilder(template="Beantworte die Frage: {{query}}\nAntwort:"),
    name="german_prompt_builder"
)

p.add_component(
    instance=HuggingFaceLocalGenerator(model="DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1"),
    name="german_llm"
)
p.add_component(
    instance=HuggingFaceLocalGenerator(model="microsoft/Phi-3-mini-4k-instruct"),
    name="english_llm"
)

p.connect("text_router.en", "english_prompt_builder.query")
p.connect("text_router.de", "german_prompt_builder.query")
p.connect("english_prompt_builder.prompt", "english_llm.prompt")
p.connect("german_prompt_builder.prompt", "german_llm.prompt")

# English Example
print(p.run({"text_router": {"text": "What is the capital of Germany?"}}))

# German Example
print(p.run({"text_router": {"text": "Was ist die Hauptstadt von Deutschland?"}}))
```

#### __init__

```python
__init__(
    model: str,
    labels: list[str] | None = None,
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    huggingface_pipeline_kwargs: dict[str, Any] | None = None,
) -> None
```

Initializes the TransformersTextRouter component.

**Parameters:**

- **model** (<code>str</code>) – The name or path of a Hugging Face model for text classification.
- **labels** (<code>list\[str\] | None</code>) – The list of labels. If not provided, the component fetches the labels
  from the model configuration file hosted on the Hugging Face Hub using
  `transformers.AutoConfig.from_pretrained`.
- **device** (<code>ComponentDevice | None</code>) – The device for loading the model. If `None`, automatically selects the default device.
  If a device or device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
- **token** (<code>Secret | None</code>) – The API token used to download private models from Hugging Face.
  If `True`, uses either `HF_API_TOKEN` or `HF_TOKEN` environment variables.
  To generate these tokens, run `transformers-cli login`.
- **huggingface_pipeline_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments for initializing the Hugging Face
  text classification pipeline.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TransformersTextRouter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TransformersTextRouter</code> – Deserialized component.

#### run

```python
run(text: str) -> dict[str, str]
```

Routes the text strings to different connections based on a category label.

**Parameters:**

- **text** (<code>str</code>) – A string of text to route.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with the label as key and the text as value.

**Raises:**

- <code>TypeError</code> – If the input is not a str.

## haystack_integrations.components.routers.transformers.zero_shot_text_router

### TransformersZeroShotTextRouter

Routes the text strings to different connections based on a category label.

Specify the set of labels for categorization when initializing the component.

### Usage example

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.core.pipeline import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.routers.transformers import TransformersZeroShotTextRouter

document_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2")
docs = [
    Document(
        content="Germany, officially the Federal Republic of Germany, is a country in the western region of "
        "Central Europe. The nation's capital and most populous city is Berlin and its main financial centre "
        "is Frankfurt; the largest urban area is the Ruhr."
    ),
    Document(
        content="France, officially the French Republic, is a country located primarily in Western Europe. "
        "France is a unitary semi-presidential republic with its capital in Paris, the country's largest city "
        "and main cultural and commercial centre; other major urban areas include Marseille, Lyon, Toulouse, "
        "Lille, Bordeaux, Strasbourg, Nantes and Nice."
    )
]
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

p = Pipeline()
p.add_component(instance=TransformersZeroShotTextRouter(labels=["passage", "query"]), name="text_router")
p.add_component(
    instance=SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", prefix="passage: "),
    name="passage_embedder"
)
p.add_component(
    instance=SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", prefix="query: "),
    name="query_embedder"
)
p.add_component(
    instance=InMemoryEmbeddingRetriever(document_store=document_store),
    name="query_retriever"
)
p.add_component(
    instance=InMemoryEmbeddingRetriever(document_store=document_store),
    name="passage_retriever"
)

p.connect("text_router.passage", "passage_embedder.text")
p.connect("passage_embedder.embedding", "passage_retriever.query_embedding")
p.connect("text_router.query", "query_embedder.text")
p.connect("query_embedder.embedding", "query_retriever.query_embedding")

# Query Example
p.run({"text_router": {"text": "What is the capital of Germany?"}})

# Passage Example
p.run({
    "text_router":{
        "text": "The United Kingdom of Great Britain and Northern Ireland, commonly known as the "            "United Kingdom (UK) or Britain, is a country in Northwestern Europe, off the north-western coast of "            "the continental mainland."
    }
})
```

#### __init__

```python
__init__(
    labels: list[str],
    multi_label: bool = False,
    model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    huggingface_pipeline_kwargs: dict[str, Any] | None = None,
) -> None
```

Initializes the TransformersZeroShotTextRouter component.

**Parameters:**

- **labels** (<code>list\[str\]</code>) – The set of labels to use for classification. Can be a single label,
  a string of comma-separated labels, or a list of labels.
- **multi_label** (<code>bool</code>) – Indicates if multiple labels can be true.
  If `False`, label scores are normalized so their sum equals 1 for each sequence.
  If `True`, the labels are considered independent and probabilities are normalized for each candidate by
  doing a softmax of the entailment score vs. the contradiction score.
- **model** (<code>str</code>) – The name or path of a Hugging Face model for zero-shot text classification.
- **device** (<code>ComponentDevice | None</code>) – The device for loading the model. If `None`, automatically selects the default device.
  If a device or device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
- **token** (<code>Secret | None</code>) – The API token used to download private models from Hugging Face.
  If `True`, uses either `HF_API_TOKEN` or `HF_TOKEN` environment variables.
  To generate these tokens, run `transformers-cli login`.
- **huggingface_pipeline_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments for initializing the Hugging Face
  zero shot text classification.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TransformersZeroShotTextRouter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TransformersZeroShotTextRouter</code> – Deserialized component.

#### run

```python
run(text: str) -> dict[str, str]
```

Routes the text strings to different connections based on a category label.

**Parameters:**

- **text** (<code>str</code>) – A string of text to route.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with the label as key and the text as value.

**Raises:**

- <code>TypeError</code> – If the input is not a str.
