---
title: "Builders"
id: builders-api
description: "Extract the output of a Generator to an Answer format, and build prompts."
slug: "/builders-api"
---

<a id="answer_builder"></a>

## Module answer\_builder

<a id="answer_builder.AnswerBuilder"></a>

### AnswerBuilder

Converts a query and Generator replies into a `GeneratedAnswer` object.

AnswerBuilder parses Generator replies using custom regular expressions.
Check out the usage example below to see how it works.
Optionally, it can also take documents and metadata from the Generator to add to the `GeneratedAnswer` object.
AnswerBuilder works with both non-chat and chat Generators.

### Usage example


### Usage example with documents and reference pattern

```python
from haystack.components.builders import AnswerBuilder

builder = AnswerBuilder(pattern="Answer: (.*)")
builder.run(query="What's the answer?", replies=["This is an argument. Answer: This is the answer."])
```
```python
from haystack import Document
from haystack.components.builders import AnswerBuilder

replies = ["The capital of France is Paris [2]."]

docs = [
    Document(content="Berlin is the capital of Germany."),
    Document(content="Paris is the capital of France."),
    Document(content="Rome is the capital of Italy."),
]

builder = AnswerBuilder(reference_pattern="\[(\d+)\]", return_only_referenced_documents=False)
result = builder.run(query="What is the capital of France?", replies=replies, documents=docs)["answers"][0]

print(f"Answer: {result.data}")
print("References:")
for doc in result.documents:
    if doc.meta["referenced"]:
        print(f"[{doc.meta['source_index']}] {doc.content}")
print("Other sources:")
for doc in result.documents:
    if not doc.meta["referenced"]:
        print(f"[{doc.meta['source_index']}] {doc.content}")

# Answer: The capital of France is Paris
# References:
# [2] Paris is the capital of France.
# Other sources:
# [1] Berlin is the capital of Germany.
# [3] Rome is the capital of Italy.
```

<a id="answer_builder.AnswerBuilder.__init__"></a>

#### AnswerBuilder.\_\_init\_\_

```python
def __init__(pattern: Optional[str] = None,
             reference_pattern: Optional[str] = None,
             last_message_only: bool = False,
             *,
             return_only_referenced_documents: bool = True)
```

Creates an instance of the AnswerBuilder component.

**Arguments**:

- `pattern`: The regular expression pattern to extract the answer text from the Generator.
If not specified, the entire response is used as the answer.
The regular expression can have one capture group at most.
If present, the capture group text
is used as the answer. If no capture group is present, the whole match is used as the answer.
Examples:
    `[^\n]+$` finds "this is an answer" in a string "this is an argument.\nthis is an answer".
    `Answer: (.*)` finds "this is an answer" in a string "this is an argument. Answer: this is an answer".
- `reference_pattern`: The regular expression pattern used for parsing the document references.
If not specified, no parsing is done, and all documents are returned.
References need to be specified as indices of the input documents and start at [1].
Example: `\[(\d+)\]` finds "1" in a string "this is an answer[1]".
If this parameter is provided, documents metadata will contain a "referenced" key with a boolean value.
- `last_message_only`: If False (default value), all messages are used as the answer.
If True, only the last message is used as the answer.
- `return_only_referenced_documents`: To be used in conjunction with `reference_pattern`.
If True (default value), only the documents that were actually referenced in `replies` are returned.
If False, all documents are returned.
If `reference_pattern` is not provided, this parameter has no effect, and all documents are returned.

<a id="answer_builder.AnswerBuilder.run"></a>

#### AnswerBuilder.run

```python
@component.output_types(answers=list[GeneratedAnswer])
def run(query: str,
        replies: Union[list[str], list[ChatMessage]],
        meta: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[Document]] = None,
        pattern: Optional[str] = None,
        reference_pattern: Optional[str] = None)
```

Turns the output of a Generator into `GeneratedAnswer` objects using regular expressions.

**Arguments**:

- `query`: The input query used as the Generator prompt.
- `replies`: The output of the Generator. Can be a list of strings or a list of `ChatMessage` objects.
- `meta`: The metadata returned by the Generator. If not specified, the generated answer will contain no metadata.
- `documents`: The documents used as the Generator inputs. If specified, they are added to
the `GeneratedAnswer` objects.
Each Document.meta includes a "source_index" key, representing its 1-based position in the input list.
When `reference_pattern` is provided:
- "referenced" key is added to the Document.meta, indicating if the document was referenced in the output.
- `return_only_referenced_documents` init parameter controls if all or only referenced documents are
returned.
- `pattern`: The regular expression pattern to extract the answer text from the Generator.
If not specified, the entire response is used as the answer.
The regular expression can have one capture group at most.
If present, the capture group text
is used as the answer. If no capture group is present, the whole match is used as the answer.
    Examples:
        `[^\n]+$` finds "this is an answer" in a string "this is an argument.\nthis is an answer".
        `Answer: (.*)` finds "this is an answer" in a string
        "this is an argument. Answer: this is an answer".
- `reference_pattern`: The regular expression pattern used for parsing the document references.
If not specified, no parsing is done, and all documents are returned.
References need to be specified as indices of the input documents and start at [1].
Example: `\[(\d+)\]` finds "1" in a string "this is an answer[1]".

**Returns**:

A dictionary with the following keys:
- `answers`: The answers received from the output of the Generator.

<a id="prompt_builder"></a>

## Module prompt\_builder

<a id="prompt_builder.PromptBuilder"></a>

### PromptBuilder

Renders a prompt filling in any variables so that it can send it to a Generator.

The prompt uses Jinja2 template syntax.
The variables in the default template are used as PromptBuilder's input and are all optional.
If they're not provided, they're replaced with an empty string in the rendered prompt.
To try out different prompts, you can replace the prompt template at runtime by
providing a template for each pipeline run invocation.

### Usage examples

#### On its own

This example uses PromptBuilder to render a prompt template and fill it with `target_language`
and `snippet`. PromptBuilder returns a prompt with the string "Translate the following context to Spanish.
Context: I can't speak Spanish.; Translation:".
```python
from haystack.components.builders import PromptBuilder

template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
builder = PromptBuilder(template=template)
builder.run(target_language="spanish", snippet="I can't speak spanish.")
```

#### In a Pipeline

This is an example of a RAG pipeline where PromptBuilder renders a custom prompt template and fills it
with the contents of the retrieved documents and a query. The rendered prompt is then sent to a Generator.
```python
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

# in a real world use case documents could come from a retriever, web, or any other source
documents = [Document(content="Joe lives in Berlin"), Document(content="Joe is a software engineer")]
prompt_template = """
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Question: {{query}}
    Answer:
    """
p = Pipeline()
p.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
p.add_component(instance=OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY")), name="llm")
p.connect("prompt_builder", "llm")

question = "Where does Joe live?"
result = p.run({"prompt_builder": {"documents": documents, "query": question}})
print(result)
```

#### Changing the template at runtime (prompt engineering)

You can change the prompt template of an existing pipeline, like in this example:
```python
documents = [
    Document(content="Joe lives in Berlin", meta={"name": "doc1"}),
    Document(content="Joe is a software engineer", meta={"name": "doc1"}),
]
new_template = """
    You are a helpful assistant.
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        Document {{ loop.index }}:
        Document name: {{ doc.meta['name'] }}
        {{ doc.content }}
    {% endfor %}

    Question: {{ query }}
    Answer:
    """
p.run({
    "prompt_builder": {
        "documents": documents,
        "query": question,
        "template": new_template,
    },
})
```
To replace the variables in the default template when testing your prompt,
pass the new variables in the `variables` parameter.

#### Overwriting variables at runtime

To overwrite the values of variables, use `template_variables` during runtime:
```python
language_template = """
You are a helpful assistant.
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    Document {{ loop.index }}:
    Document name: {{ doc.meta['name'] }}
    {{ doc.content }}
{% endfor %}

Question: {{ query }}
Please provide your answer in {{ answer_language | default('English') }}
Answer:
"""
p.run({
    "prompt_builder": {
        "documents": documents,
        "query": question,
        "template": language_template,
        "template_variables": {"answer_language": "German"},
    },
})
```
Note that `language_template` introduces variable `answer_language` which is not bound to any pipeline variable.
If not set otherwise, it will use its default value 'English'.
This example overwrites its value to 'German'.
Use `template_variables` to overwrite pipeline variables (such as documents) as well.

<a id="prompt_builder.PromptBuilder.__init__"></a>

#### PromptBuilder.\_\_init\_\_

```python
def __init__(template: str,
             required_variables: Optional[Union[list[str],
                                                Literal["*"]]] = None,
             variables: Optional[list[str]] = None)
```

Constructs a PromptBuilder component.

**Arguments**:

- `template`: A prompt template that uses Jinja2 syntax to add variables. For example:
`"Summarize this document: {{ documents[0].content }}\nSummary:"`
It's used to render the prompt.
The variables in the default template are input for PromptBuilder and are all optional,
unless explicitly specified.
If an optional variable is not provided, it's replaced with an empty string in the rendered prompt.
- `required_variables`: List variables that must be provided as input to PromptBuilder.
If a variable listed as required is not provided, an exception is raised.
If set to "*", all variables found in the prompt are required. Optional.
- `variables`: List input variables to use in prompt templates instead of the ones inferred from the
`template` parameter. For example, to use more variables during prompt engineering than the ones present
in the default template, you can provide them here.

<a id="prompt_builder.PromptBuilder.to_dict"></a>

#### PromptBuilder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Returns a dictionary representation of the component.

**Returns**:

Serialized dictionary representation of the component.

<a id="prompt_builder.PromptBuilder.run"></a>

#### PromptBuilder.run

```python
@component.output_types(prompt=str)
def run(template: Optional[str] = None,
        template_variables: Optional[dict[str, Any]] = None,
        **kwargs)
```

Renders the prompt template with the provided variables.

It applies the template variables to render the final prompt. You can provide variables via pipeline kwargs.
In order to overwrite the default template, you can set the `template` parameter.
In order to overwrite pipeline kwargs, you can set the `template_variables` parameter.

**Arguments**:

- `template`: An optional string template to overwrite PromptBuilder's default template. If None, the default template
provided at initialization is used.
- `template_variables`: An optional dictionary of template variables to overwrite the pipeline variables.
- `kwargs`: Pipeline variables used for rendering the prompt.

**Raises**:

- `ValueError`: If any of the required template variables is not provided.

**Returns**:

A dictionary with the following keys:
- `prompt`: The updated prompt text after rendering the prompt template.

<a id="chat_prompt_builder"></a>

## Module chat\_prompt\_builder

<a id="chat_prompt_builder.ChatPromptBuilder"></a>

### ChatPromptBuilder

Renders a chat prompt from a template using Jinja2 syntax.

A template can be a list of `ChatMessage` objects, or a special string, as shown in the usage examples.

It constructs prompts using static or dynamic templates, which you can update for each pipeline run.

Template variables in the template are optional unless specified otherwise.
If an optional variable isn't provided, it defaults to an empty string. Use `variable` and `required_variables`
to define input types and required variables.

### Usage examples

#### Static ChatMessage prompt template

```python
template = [ChatMessage.from_user("Translate to {{ target_language }}. Context: {{ snippet }}; Translation:")]
builder = ChatPromptBuilder(template=template)
builder.run(target_language="spanish", snippet="I can't speak spanish.")
```

#### Overriding static ChatMessage template at runtime

```python
template = [ChatMessage.from_user("Translate to {{ target_language }}. Context: {{ snippet }}; Translation:")]
builder = ChatPromptBuilder(template=template)
builder.run(target_language="spanish", snippet="I can't speak spanish.")

msg = "Translate to {{ target_language }} and summarize. Context: {{ snippet }}; Summary:"
summary_template = [ChatMessage.from_user(msg)]
builder.run(target_language="spanish", snippet="I can't speak spanish.", template=summary_template)
```

#### Dynamic ChatMessage prompt template

```python
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack.utils import Secret

# no parameter init, we don't use any runtime template variables
prompt_builder = ChatPromptBuilder()
llm = OpenAIChatGenerator(api_key=Secret.from_token("<your-api-key>"), model="gpt-4o-mini")

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("prompt_builder.prompt", "llm.messages")

location = "Berlin"
language = "English"
system_message = ChatMessage.from_system("You are an assistant giving information to tourists in {{language}}")
messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]

res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "language": language},
                                    "template": messages}})
print(res)

>> {'llm': {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
"Berlin is the capital city of Germany and one of the most vibrant
and diverse cities in Europe. Here are some key things to know...Enjoy your time exploring the vibrant and dynamic
capital of Germany!")], _name=None, _meta={'model': 'gpt-4o-mini',
'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 27, 'completion_tokens': 681, 'total_tokens':
708}})]}}

messages = [system_message, ChatMessage.from_user("What's the weather forecast for {{location}} in the next
{{day_count}} days?")]

res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "day_count": "5"},
                                    "template": messages}})

print(res)
>> {'llm': {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
"Here is the weather forecast for Berlin in the next 5
days:\n\nDay 1: Mostly cloudy with a high of 22°C (72°F) and...so it's always a good idea to check for updates
closer to your visit.")], _name=None, _meta={'model': 'gpt-4o-mini',
'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 37, 'completion_tokens': 201,
'total_tokens': 238}})]}}
```

#### String prompt template
```python
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses.image_content import ImageContent

template = """
{% message role="system" %}
You are a helpful assistant.
{% endmessage %}

{% message role="user" %}
Hello! I am {{user_name}}. What's the difference between the following images?
{% for image in images %}
{{ image | templatize_part }}
{% endfor %}
{% endmessage %}
"""

images = [ImageContent.from_file_path("apple.jpg"), ImageContent.from_file_path("orange.jpg")]

builder = ChatPromptBuilder(template=template)
builder.run(user_name="John", images=images)
```

<a id="chat_prompt_builder.ChatPromptBuilder.__init__"></a>

#### ChatPromptBuilder.\_\_init\_\_

```python
def __init__(template: Optional[Union[list[ChatMessage], str]] = None,
             required_variables: Optional[Union[list[str],
                                                Literal["*"]]] = None,
             variables: Optional[list[str]] = None)
```

Constructs a ChatPromptBuilder component.

**Arguments**:

- `template`: A list of `ChatMessage` objects or a string template. The component looks for Jinja2 template syntax and
renders the prompt with the provided variables. Provide the template in either
the `init` method` or the `run` method.
- `required_variables`: List variables that must be provided as input to ChatPromptBuilder.
If a variable listed as required is not provided, an exception is raised.
If set to "*", all variables found in the prompt are required. Optional.
- `variables`: List input variables to use in prompt templates instead of the ones inferred from the
`template` parameter. For example, to use more variables during prompt engineering than the ones present
in the default template, you can provide them here.

<a id="chat_prompt_builder.ChatPromptBuilder.run"></a>

#### ChatPromptBuilder.run

```python
@component.output_types(prompt=list[ChatMessage])
def run(template: Optional[Union[list[ChatMessage], str]] = None,
        template_variables: Optional[dict[str, Any]] = None,
        **kwargs)
```

Renders the prompt template with the provided variables.

It applies the template variables to render the final prompt. You can provide variables with pipeline kwargs.
To overwrite the default template, you can set the `template` parameter.
To overwrite pipeline kwargs, you can set the `template_variables` parameter.

**Arguments**:

- `template`: An optional list of `ChatMessage` objects or string template to overwrite ChatPromptBuilder's default
template.
If `None`, the default template provided at initialization is used.
- `template_variables`: An optional dictionary of template variables to overwrite the pipeline variables.
- `kwargs`: Pipeline variables used for rendering the prompt.

**Raises**:

- `ValueError`: If `chat_messages` is empty or contains elements that are not instances of `ChatMessage`.

**Returns**:

A dictionary with the following keys:
- `prompt`: The updated list of `ChatMessage` objects after rendering the templates.

<a id="chat_prompt_builder.ChatPromptBuilder.to_dict"></a>

#### ChatPromptBuilder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Returns a dictionary representation of the component.

**Returns**:

Serialized dictionary representation of the component.

<a id="chat_prompt_builder.ChatPromptBuilder.from_dict"></a>

#### ChatPromptBuilder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ChatPromptBuilder"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize and create the component.

**Returns**:

The deserialized component.

