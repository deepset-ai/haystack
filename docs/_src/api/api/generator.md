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
|     question = "who got the first nobel prize in physics?"
|
|     # Retrieve related documents from retriever
|     retrieved_docs = retriever.retrieve(query=question)
|
|     # Now generate answer from question and retrieved documents
|     generator.predict(
|        question=question,
|        documents=retrieved_docs,
|        top_k=1
|     )
|
|     # Answer
|
|     {'question': 'who got the first nobel prize in physics',
|      'answers':
|          [{'question': 'who got the first nobel prize in physics',
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
 | __init__(model_name_or_path: str = "facebook/rag-token-nq", retriever: Optional[DensePassageRetriever] = None, generator_type: RAGeneratorType = RAGeneratorType.TOKEN, top_k_answers: int = 2, max_length: int = 200, min_length: int = 2, num_beams: int = 2, embed_title: bool = True, prefix: Optional[str] = None, use_gpu: bool = True)
```

Load a RAG model from Transformers along with passage_embedding_model.
See https://huggingface.co/transformers/model_doc/rag.html for more details

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g.
'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
See https://huggingface.co/models for full list of available models.
- `retriever`: `DensePassageRetriever` used to embedded passage
- `generator_type`: Which RAG generator implementation to use? RAG-TOKEN or RAG-SEQUENCE
- `top_k_answers`: Number of independently generated text to return
- `max_length`: Maximum length of generated text
- `min_length`: Minimum length of generated text
- `num_beams`: Number of beams for beam search. 1 means no beam search.
- `embed_title`: Embedded the title of passage while generating embedding
- `prefix`: The prefix used by the generator's tokenizer.
- `use_gpu`: Whether to use GPU (if available)

<a name="transformers.RAGenerator.predict"></a>
#### predict

```python
 | predict(question: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
```

Generate the answer to the input question. The generation will be conditioned on the supplied documents.
These document can for example be retrieved via the Retriever.

**Arguments**:

- `question`: Question
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `top_k`: Number of returned answers

**Returns**:

Generated answers plus additional infos in a dict like this:

```python
|     {'question': 'who got the first nobel prize in physics',
|      'answers':
|          [{'question': 'who got the first nobel prize in physics',
|            'answer': ' albert einstein',
|            'meta': { 'doc_ids': [...],
|                      'doc_scores': [80.42758 ...],
|                      'doc_probabilities': [40.71379089355469, ...
|                      'texts': ['Albert Einstein was a ...]
|                      'titles': ['"Albert Einstein"', ...]
|      }}]}
```

<a name="base"></a>
# Module base

<a name="base.BaseGenerator"></a>
## BaseGenerator Objects

```python
class BaseGenerator(ABC)
```

Abstract class for Generators

<a name="base.BaseGenerator.predict"></a>
#### predict

```python
 | @abstractmethod
 | predict(question: str, documents: List[Document], top_k: Optional[int]) -> Dict
```

Abstract method to generate answers.

**Arguments**:

- `question`: Question
- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `top_k`: Number of returned answers

**Returns**:

Generated answers plus additional infos in a dict

