<a id="base"></a>

# Module base

<a id="base.BaseGenerator"></a>

## BaseGenerator

```python
class BaseGenerator(BaseComponent)
```

Abstract class for Generators

<a id="base.BaseGenerator.predict"></a>

#### predict

```python
@abstractmethod
def predict(query: str, documents: List[Document], top_k: Optional[int]) -> Dict
```

Abstract method to generate answers.

**Arguments**:

- `query`: Query
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `top_k`: Number of returned answers

**Returns**:

Generated answers plus additional infos in a dict

<a id="transformers"></a>

# Module transformers

<a id="transformers.RAGenerator"></a>

## RAGenerator

```python
class RAGenerator(BaseGenerator)
```

Implementation of Facebook's Retrieval-Augmented Generator (https://arxiv.org/abs/2005.11401) based on
HuggingFace's transformers (https://huggingface.co/transformers/model_doc/rag.html).

Instead of "finding" the answer within a document, these models **generate** the answer.
In that sense, RAG follows a similar approach as GPT-3 but it comes with two huge advantages
for real-world applications:
a) it has a manageable model size
b) the answer generation is conditioned on retrieved documents,
i.e. the model can easily adjust to domain documents even after training has finished
(in contrast: GPT-3 relies on the web data seen during training)

**Example**

```python
|     query = "who got the first nobel prize in physics?"
|
|     # Retrieve related documents from retriever
|     retrieved_docs = retriever.retrieve(query=query)
|
|     # Now generate answer from query and retrieved documents
|     generator.predict(
|        query=query,
|        documents=retrieved_docs,
|        top_k=1
|     )
|
|     # Answer
|
|     {'query': 'who got the first nobel prize in physics',
|      'answers':
|          [{'query': 'who got the first nobel prize in physics',
|            'answer': ' albert einstein',
|            'meta': { 'doc_ids': [...],
|                      'doc_scores': [80.42758 ...],
|                      'doc_probabilities': [40.71379089355469, ...
|                      'content': ['Albert Einstein was a ...]
|                      'titles': ['"Albert Einstein"', ...]
|      }}]}
```

<a id="transformers.RAGenerator.predict"></a>

#### predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
```

Generate the answer to the input query. The generation will be conditioned on the supplied documents.

These document can for example be retrieved via the Retriever.

**Arguments**:

- `query`: Query
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `top_k`: Number of returned answers

**Returns**:

Generated answers plus additional infos in a dict like this:
```python
|     {'query': 'who got the first nobel prize in physics',
|      'answers':
|          [{'query': 'who got the first nobel prize in physics',
|            'answer': ' albert einstein',
|            'meta': { 'doc_ids': [...],
|                      'doc_scores': [80.42758 ...],
|                      'doc_probabilities': [40.71379089355469, ...
|                      'content': ['Albert Einstein was a ...]
|                      'titles': ['"Albert Einstein"', ...]
|      }}]}
```

<a id="transformers.Seq2SeqGenerator"></a>

## Seq2SeqGenerator

```python
class Seq2SeqGenerator(BaseGenerator)
```

A generic sequence-to-sequence generator based on HuggingFace's transformers.

Text generation is supported by so called auto-regressive language models like GPT2,
XLNet, XLM, Bart, T5 and others. In fact, any HuggingFace language model that extends
GenerationMixin can be used by Seq2SeqGenerator.

Moreover, as language models prepare model input in their specific encoding, each model
specified with model_name_or_path parameter in this Seq2SeqGenerator should have an
accompanying model input converter that takes care of prefixes, separator tokens etc.
By default, we provide model input converters for a few well-known seq2seq language models (e.g. ELI5).
It is the responsibility of Seq2SeqGenerator user to ensure an appropriate model input converter
is either already registered or specified on a per-model basis in the Seq2SeqGenerator constructor.

For mode details on custom model input converters refer to _BartEli5Converter


See https://huggingface.co/transformers/main_classes/model.html?transformers.generation_utils.GenerationMixin#transformers.generation_utils.GenerationMixin
as well as https://huggingface.co/blog/how-to-generate

For a list of all text-generation models see https://huggingface.co/models?pipeline_tag=text-generation

**Example**

```python
|     query = "Why is Dothraki language important?"
|
|     # Retrieve related documents from retriever
|     retrieved_docs = retriever.retrieve(query=query)
|
|     # Now generate answer from query and retrieved documents
|     generator.predict(
|        query=query,
|        documents=retrieved_docs,
|        top_k=1
|     )
|
|     # Answer
|
|     {'query': 'who got the first nobel prize in physics',
|      'answers':
|          [{'query': 'who got the first nobel prize in physics',
|            'answer': ' albert einstein',
|            'meta': { 'doc_ids': [...],
|                      'doc_scores': [80.42758 ...],
|                      'doc_probabilities': [40.71379089355469, ...
|                      'content': ['Albert Einstein was a ...]
|                      'titles': ['"Albert Einstein"', ...]
|      }}]}
```

<a id="transformers.Seq2SeqGenerator.predict"></a>

#### predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
```

Generate the answer to the input query. The generation will be conditioned on the supplied documents.

These document can be retrieved via the Retriever or supplied directly via predict method.

**Arguments**:

- `query`: Query
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `top_k`: Number of returned answers

**Returns**:

Generated answers

