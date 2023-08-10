- Title: LLM SUpport in Haystack 2.0
- Decision driver: @ZanSara
- Start Date: 2023-08-10
- Proposal PR: #5540
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/5330

# Summary

In this proposal we outline our migration strategy of the `PromptNode` stack of functionality into Haystack 2.0 components.

This proposal, however, does not discuss Agents or Agent-related features and functionality, which are going to be addressed into a separate proposal.

# Motivation

This proposal is part of a larger effort of migrating existing Haystack 1.x components into their 2.0 counterpart.

# Basic example

In Haystack 2.0 components should be smaller than their 1.x counterpart. Therefore, we plan to split the functionality of `PromptNode` into a few smaller components.

As this proposal concerns LLM support, not specifically Agents, the main usecase in question is RAG (Retrieval-Augmented Generation).

## Example: RAG Pipeline

Using the components discussed in the Detailed Design section, a Haystack 2.0 RAG pipeline may look like this:

```mermaid
graph TD;

IN{IN} -- "questions (List[str])" --> Retriever
IN{IN} -- "questions (List[str])" --> Prompts
Retriever -- "documents (List[List[Doc]])"  --> Prompts
Prompts -- "prompts (List[str])" --> GPT4
GPT4 -- "replies (List[List[str]])" --> RepliesToAnswersConverter
RepliesToAnswersConverter -- "answers (List[List[Answer]])" --> OUT{OUT}
```

While the code for such pipeline may look like:

```python
from haystack.preview.components import MemoryRetriever, Prompts, ChatGPT, RepliesToAnswersConverter
from haystack.preview.document_stores import MemoryDocumentStore
from haystack.preview.pipeline import Pipeline

pipe = Pipeline()
pipe.add_store("store", MemoryDocumentStore())
pipe.add_component("retriever", MemoryRetriever(), store="store")
pipe.add_component("prompts", Prompts("deepset/question-answering"))
pipe.add_component("llm", GPT4(api_key="..."))
pipe.add_component("replies_converter", RepliesToAnswersConverter())

pipe.connect("retriever", "prompt")
pipe.connect("prompt", "llm")
pipe.connect("llm", "replies_converter")

questions = ["Why?", "Why not?"]
results = pipe.run({
	"retriever": {"queries": questions},
	"prompt": {"questions": questions},
})

assert results == {
	"replies_converter": {
    "answers": [[Answer("Because of this.")], [Answer("Because of that.")]]
  }
}
```

# Detailed design

Haystack’s `PromptNode` is a very complex component that includes under its name several functionalities: loading prompt templates through the `PromptTemplate` class, rendering such prompt template with the variables from the invocation context, choosing which LLM backend to use, sending the prompt to the LLM using the correct invocation layer, interpreting the results, parsing them into objects, and putting them back in the pipeline in a way other components can understand.

in Haystack 2.0 we unpack these functionalities into a few separate components, to clarify what is happening, how it works, and provide additional flexibility.

The main functionalities we identified are the following:

1. Fetching the prompt from different sources
2. Rendering the prompt using variables
3. Invoke the LLM
4. Parse the output
5. History/Memory management

We leave the discussion about History/Memory to a separate proposal, as it concerns mostly Agents, and focus on the other points.

## LLM invocation

In Haystack 1.x, `PromptNode` uses `InvocationLayer` to query different LLMs under a unified API. In that design, users do not need to know which invocation layer is used for the model they select, as `PromptNode` takes responsibility of selecting it.

Such invocation layers can be ported to 2.0 as standalone components. In this way we will have one component for each LLM backed that we support.

Each component should be named after the class of models it supports. For example we should have `GPT4`, `ChatGPT`, etc. For components whose name may be confusing, or that include models that are not LLMs, we can add a LLM suffix: like `FalconLLM`, `CohereLLM`, `HuggingFaceLLM`.

Note that having separate components for each LLM makes easy to deprecate them when we realize they are dropping out of favor or become severely outdated. It also makes very easy for external contributors to make their own external components to support rarer LLMs, without having to add them to Haystack’s core.

All these LLM clients will have a near-identical I/O:

```python
@component
class ChatGPT:

    @component.output_types(replies=List[List[str]])
    def run(self, prompts: List[str], ... chatgpt specific params...):
        ...
        return {'replies': [...]}
```

Note how the component takes a list of prompts and LLM parameters only, but no variables nor templates, and returns only strings. This is because input rendering and output parsing are delegated to separate components, which description follows.

### Returning metadata

In the example above we made the LLM return only a list of replies, as strings. However, in order to be able to parse the output into meaningful objects (see “Output parsing”) we may need additional metadata from these clients.

1. Do we already have any such situation?
2. Can we foresee any other?

If the answer to any of the above is yes, a simple, maybe temporary solution would be to add a second output, called for example `'replies_meta'` . Any component that need such meta to parse the output would then request this second output along with the first and zip the two lists together to reconstruct the original output of the LLM.

### Returning streams

In some cases users may want to see the output of the LLM as it’s being generated, so as a stream that prints word by word in the terminal.

To achieve such effect, we should make LLM components return a more generic type `replies: List[Iterator[str]]`. In this way, users will have the ability of returning a generator or iterator in place of a regular string, and therefore allow the receiving components to unroll these replies as they wish. Components that need the full reply will unroll the iterator internally (like `RepliesToAnswersConverter`), while if the generator goes in the output, the users can unroll it in a loop printing the content word by word. We can also have specialized components to visualize the stream within a pipeline: something like:

```mermaid
graph TD;

IN(...previous components...) --> GPT4
GPT4 -- "replies (List[Iterator[str]])" --> S[StreamPrinter\n<i><small>Prints the input stream to the console\nwhile unrolling the iterator\n.]
S -- "replies (List[List[str]])" --> RepliesToAnswersConverter
RepliesToAnswersConverter -- "answers (List[List[Answer]])" --> OUT{OUT}

```

### How many clients we will have?

Basing on the list of current invocation layers in Haystack 1.x, the list might look like:

1. `Claude`
2. `ChatGPT` (supporting GPT4 as well)
3. `Cohere`
4. `HuggingFaceInferenceLLM`
5. `HuggingFaceLocalLLM`
6. `GPT3`
6. `AzureGPT3`?
7. `Bard` ?
8. `SagemakerLLM`

Plus one more for any other inference hosting/library that may appear in the future.

## Prompt Builder

In Haystack 1.x, prompts fetching and rendering is carried out by `PromptTemplate`. In 2.0, we rather make a separate `Prompts` component to handle this process.

Due to the dynamic nature of prompt templates, the `Prompts.run()`  method takes `kwargs`, which contains all the variables that will be filled in the template. However, for this component to work with Canals, we need to know in advance which values this dict will contain: therefore, we need the users to specify in the `__init__` of the component which parameters to expect in the template.

Keep in mind that such parameters names **cannot be changed at runtime**.

We foresee `Prompts` to output a list of prompts, not necessarily just one, as currently  some of our existing prompt templates produce several output prompts when rendered.

Draft I/O for `Prompts`:

```python
@component
class Prompts:

    def __init__(self, template_variables: Union[str, Path]):
        self.template_variables = template_variables
		component.set_input_parameters(**{var: Any for var in template_variables})

  	@component.output_types(prompts=List[str])
    def run(self, template: str, **kwargs):
        self.template_text = # Load the template

        variables = # extracts the variables from the template text
        if variables != self.template_variables:
            raise ValueError()

        # Render the template using the variables
        return {"prompts": prompts}
```

### **Why we need to specify the variables in `__init__`?**

The design above derives from one Canals limitation: component’s sockets need to be all known the latest at `__init__` time, in order for the connections to be made and validated. Therefore, we need to know all the prompt variables before building the pipelines, because the prompt variables are inputs of the `run()` method.

However, earlier iterations of Canals did support so-called “true variadic” components: components that do not need to know what they will be connected to, and build the input sockets at need. Such components of course lack input validation, but enable usecases like the above.

If we decide that Canals should support again such components, we would be able to rewrite `Prompts` to take a prompt as its input parameter and just accept any other incoming input, on the assumption that users knows that they’re doing.

For example:

```python
@component
class Prompts:

	@variadic_input
  	@component.output_types(prompts=List[str])
    def run(self, template: Union[str, Path], **kwargs):
	    # ... loads the template ...
        # ... render the prompts ...
        return {"prompts": prompts}
```

### Why a separate `Prompts` component at all?

`PromptNode` used to take the prompt template and the variables to render it directly, and then forward the result to the LLM.

**PRO: Allows using various templating and guidance libraries, or none at all**

The key advantage of `Prompts` is ability to use any tool from the ever growing list of LLM prompting template libs. These include but are not limited to: https://github.com/microsoft/guidance,
https://shreyar.github.io/guardrails/rail/prompt/
and many many more. See list at https://www.promptingguide.ai/tools If someone has invested a lot in guidance and considers using a framework like Haystack or LangChain this will be one of the biggest selling points: an ability to use `Prompts` for a specific prompt tooling lib.

On top of that, it also allows users to skip the template rendering step altogether and send prompts directly to the LLM, which may be beneficial in some context (for example, if users just want to chat with the LLM without RAG).

**PRO: Abstracts away duplicate work (which may be expensive)**

On top of this, there is also the question of duplication of work. In a situation where a Pipeline sends the same prompt to several LLMs, the v2 version of such pipeline can easily become very complicated. Take this usecase:

```mermaid
graph TD;

LegalLLM -- "replies (List[List[str]])" --> JoinLists
CodingLLM -- "replies (List[List[str]])" --> JoinLists
MedicalLLM -- "replies (List[List[str]])" --> JoinLists
JoinLists -- "list (List[List[str]])" --> RepliesToAnswers
RepliesToAnswers -- "answers (List[List[Answer]])" --> OUT{OUT}
```

How to send the prompts to the LLMs? If the LLMs expect the template and the variables, the pipeline may look as:

```mermaid
graph TD;

IN{IN} -- "questions (List[str])" --> Retriever
IN{IN} -- "questions (List[str])" --> LegalLLM
IN{IN} -- "questions (List[str])" --> CodingLLM
IN{IN} -- "questions (List[str])" --> MedicalLLM
Retriever -- "documents (List[List[Doc]])"  --> LegalLLM
Retriever -- "documents (List[List[Doc]])"  --> CodingLLM
Retriever -- "documents (List[List[Doc]])"  --> MedicalLLM
LegalLLM -- "replies (List[List[str]])" --> JoinLists
CodingLLM -- "replies (List[List[str]])" --> JoinLists
MedicalLLM -- "replies (List[List[str]])" --> JoinLists
JoinLists -- "list (List[List[str]])" --> RepliesToAnswers
RepliesToAnswers -- "answers (List[List[Answer]])" --> OUT{OUT}
```

Assuming the prompt requires only questions and documents, or the number of connections to make will grow. Also, each of the LLM component will perform the exact same rendering step under the hood, so it results in duplication of efforts.

Compare with a pipeline that uses a `Prompts` :

```mermaid
graph TD;

IN{IN} -- "questions (List[str])" --> Retriever
IN{IN} -- "questions (List[str])" --> Prompts
Retriever -- "documents (List[List[Doc]])"  --> Prompts
Prompts -- "prompts (List[str])" --> LegalLLM
Prompts -- "prompts (List[str])" --> CodingLLM
Prompts -- "prompts (List[str])" --> MedicalLLM
LegalLLM -- "replies (List[List[str]])" --> JoinLists
CodingLLM -- "replies (List[List[str]])" --> JoinLists
MedicalLLM -- "replies (List[List[str]])" --> JoinLists
JoinLists -- "list (List[List[str]])" --> RepliesToAnswers
RepliesToAnswers -- "answers (List[List[Answer]])" --> OUT{OUT}
```

**CON: One additional component = more complexity**

The drawback is that `Prompts` is an additional component, so we must evaluate if this additional flexibility is worth the additional complexity. To do so, we need to understand how likely it is that the prompt will need to be sent to different pipelines and under which circumstances.

## Output parsing

LLMs clients output strings, but many components expect other object types, and LLMs may produce output in a parsable format that can be directly converted into objects. Output parsers transform these strings into objects of the user’s choosing.

In Haystack 1.x, this task was assigned to the subclasses of `BaseOutputParser`. In 2.0 we’re going to have a very similar situation, with the difference that such classes are components.

The most straightforward component in this category is `RepliesToAnswersConverter`. It takes the string replies of an LLM and produce `Answer` objects.  One additional output parser could be `RepliesToAnswersWithReferencesConverter`, which also connects answers to the documents used to produce them. As the need for additional output parsers arises, we will progressively add more.

Draft I/O for `RepliesToAnswersConverter` (note: this may end up being almost the entire component’s implementation):

```python
@component
class RepliesToAnswersConverter:

    @component.output_types(answers=List[List[Answer]])
    def run(self, replies: List[List[str]]):
        return {"answers": Answer(answer=answer) for answers in replies for answer in answers}
```

# Drawbacks

Possible drawbacks of this design:

1. Users now need to use three components instead of a single, large one.
2. We lose the capability to change the prompt for the LLM at runtime.

# Alternatives

1. Porting the existing `PromptNode` to v2: would be a massive effort and make v2 inherit some design decision that, with time, proved unnecessary and/or clumsy to use, like the “hiding” of invocation layer that makes it quite hard for external contributors to add support for other LLMs to `PromptNode`, or it’s imperfect layer selection algorithm.

# Adoption strategy

Follows the same strategy outlines for all other Proposal relative to the Haystack 2.0 migration

# How we teach this

We need brand new tutorials and examples of pipelines using these components.
