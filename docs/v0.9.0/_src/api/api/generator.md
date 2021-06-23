<a name="base"></a>
# Module base

<a name="base.BaseGenerator"></a>
## BaseGenerator Objects

```python
class BaseGenerator(BaseComponent)
```

Abstract class for Generators

<a name="base.BaseGenerator.predict"></a>
#### predict

```python
 | @abstractmethod
 | predict(query: str, documents: List[Document], top_k: Optional[int]) -> Dict
```

Abstract method to generate answers.

**Arguments**:

- `query`: Query
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `top_k`: Number of returned answers

**Returns**:

Generated answers plus additional infos in a dict

<a name="transformers"></a>
# Module transformers

<a name="transformers.RAGenerator"></a>
## RAGenerator Objects

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
|                      'texts': ['Albert Einstein was a ...]
|                      'titles': ['"Albert Einstein"', ...]
|      }}]}
```

<a name="transformers.RAGenerator.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str = "facebook/rag-token-nq", model_version: Optional[str] = None, retriever: Optional[DensePassageRetriever] = None, generator_type: RAGeneratorType = RAGeneratorType.TOKEN, top_k: int = 2, max_length: int = 200, min_length: int = 2, num_beams: int = 2, embed_title: bool = True, prefix: Optional[str] = None, use_gpu: bool = True)
```

Load a RAG model from Transformers along with passage_embedding_model.
See https://huggingface.co/transformers/model_doc/rag.html for more details

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g.
                           'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
                           See https://huggingface.co/models for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `retriever`: `DensePassageRetriever` used to embedded passage
- `generator_type`: Which RAG generator implementation to use? RAG-TOKEN or RAG-SEQUENCE
- `top_k`: Number of independently generated text to return
- `max_length`: Maximum length of generated text
- `min_length`: Minimum length of generated text
- `num_beams`: Number of beams for beam search. 1 means no beam search.
- `embed_title`: Embedded the title of passage while generating embedding
- `prefix`: The prefix used by the generator's tokenizer.
- `use_gpu`: Whether to use GPU (if available)

<a name="transformers.RAGenerator.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
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
|                      'texts': ['Albert Einstein was a ...]
|                      'titles': ['"Albert Einstein"', ...]
|      }}]}
```

<a name="transformers.Seq2SeqGenerator"></a>
## Seq2SeqGenerator Objects

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
|     {'answers': [" The Dothraki language is a constructed fictional language. It's important because George R.R. Martin wrote it."],
|      'query': 'Why is Dothraki language important?'}
|
```

<a name="transformers.Seq2SeqGenerator.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str, input_converter: Optional[Callable] = None, top_k: int = 1, max_length: int = 200, min_length: int = 2, num_beams: int = 8, use_gpu: bool = True)
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
- `use_gpu`: Whether to use GPU (if available)

<a name="transformers.Seq2SeqGenerator.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
```

Generate the answer to the input query. The generation will be conditioned on the supplied documents.
These document can be retrieved via the Retriever or supplied directly via predict method.

**Arguments**:

- `query`: Query
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `top_k`: Number of returned answers

**Returns**:

Generated answers

