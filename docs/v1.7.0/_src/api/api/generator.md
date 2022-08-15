<a id="base"></a>

# Module base

<a id="base.BaseGenerator"></a>

## BaseGenerator

```python
class BaseGenerator(BaseComponent)
```

Abstract class for Generators

<a id="base.BaseGenerator.predict"></a>

#### BaseGenerator.predict

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

<a id="base.BaseGenerator.predict_batch"></a>

#### BaseGenerator.predict\_batch

```python
def predict_batch(queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int] = None, batch_size: Optional[int] = None)
```

Generate the answer to the input queries. The generation will be conditioned on the supplied documents.

These documents can for example be retrieved via the Retriever.

- If you provide a list containing a single query...

    - ... and a single list of Documents, the query will be applied to each Document individually.
    - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers
      will be aggregated per Document list.

- If you provide a list of multiple queries...

    - ... and a single list of Documents, each query will be applied to each Document individually.
    - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents
      and the Answers will be aggregated per query-Document pair.

**Arguments**:

- `queries`: List of queries.
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
Can be a single list of Documents or a list of lists of Documents.
- `top_k`: Number of returned answers per query.
- `batch_size`: Not applicable.

**Returns**:

Generated answers plus additional infos in a dict like this:
```python
|     {'queries': 'who got the first nobel prize in physics',
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

<a id="transformers.RAGenerator.__init__"></a>

#### RAGenerator.\_\_init\_\_

```python
def __init__(model_name_or_path: str = "facebook/rag-token-nq", model_version: Optional[str] = None, retriever: Optional[DensePassageRetriever] = None, generator_type: str = "token", top_k: int = 2, max_length: int = 200, min_length: int = 2, num_beams: int = 2, embed_title: bool = True, prefix: Optional[str] = None, use_gpu: bool = True, progress_bar: bool = True)
```

Load a RAG model from Transformers along with passage_embedding_model.

See https://huggingface.co/transformers/model_doc/rag.html for more details

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g.
'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
See https://huggingface.co/models for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `retriever`: `DensePassageRetriever` used to embedded passages for the docs passed to `predict()`. This is optional and is only needed if the docs you pass don't already contain embeddings in `Document.embedding`.
- `generator_type`: Which RAG generator implementation to use ("token" or "sequence")
- `top_k`: Number of independently generated text to return
- `max_length`: Maximum length of generated text
- `min_length`: Minimum length of generated text
- `num_beams`: Number of beams for beam search. 1 means no beam search.
- `embed_title`: Embedded the title of passage while generating embedding
- `prefix`: The prefix used by the generator's tokenizer.
- `use_gpu`: Whether to use GPU. Falls back on CPU if no GPU is available.

<a id="transformers.RAGenerator.predict"></a>

#### RAGenerator.predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
```

Generate the answer to the input query. The generation will be conditioned on the supplied documents.

These documents can for example be retrieved via the Retriever.

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

This generator supports all [Text2Text](https://huggingface.co/models?pipeline_tag=text2text-generation) models
from the Hugging Face hub. If the primary interface for the model specified by `model_name_or_path` constructor
parameter is AutoModelForSeq2SeqLM from Hugging Face, then you can use it in this Generator.

Moreover, as language models prepare model input in their specific encoding, each model
specified with model_name_or_path parameter in this Seq2SeqGenerator should have an
accompanying model input converter that takes care of prefixes, separator tokens etc.
By default, we provide model input converters for a few well-known seq2seq language models (e.g. ELI5).
It is the responsibility of Seq2SeqGenerator user to ensure an appropriate model input converter
is either already registered or specified on a per-model basis in the Seq2SeqGenerator constructor.

For mode details on custom model input converters refer to _BartEli5Converter

For a list of all text2text-generation models, see
the [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=text2text-generation)


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

<a id="transformers.Seq2SeqGenerator.__init__"></a>

#### Seq2SeqGenerator.\_\_init\_\_

```python
def __init__(model_name_or_path: str, input_converter: Optional[Callable] = None, top_k: int = 1, max_length: int = 200, min_length: int = 2, num_beams: int = 8, use_gpu: bool = True, progress_bar: bool = True)
```

**Arguments**:

- `model_name_or_path`: a HF model name for auto-regressive language model like GPT2, XLNet, XLM, Bart, T5 etc
- `input_converter`: an optional Callable to prepare model input for the underlying language model
specified in model_name_or_path parameter. The required __call__ method signature for
the Callable is:
__call__(tokenizer: PreTrainedTokenizer, query: str, documents: List[Document],
top_k: Optional[int] = None) -> BatchEncoding:
- `top_k`: Number of independently generated text to return
- `max_length`: Maximum length of generated text
- `min_length`: Minimum length of generated text
- `num_beams`: Number of beams for beam search. 1 means no beam search.
- `use_gpu`: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.

<a id="transformers.Seq2SeqGenerator.predict"></a>

#### Seq2SeqGenerator.predict

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

