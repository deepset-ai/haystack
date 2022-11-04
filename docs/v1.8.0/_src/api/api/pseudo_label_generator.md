<a id="pseudo_label_generator"></a>

# Module pseudo\_label\_generator

<a id="pseudo_label_generator.PseudoLabelGenerator"></a>

## PseudoLabelGenerator

```python
class PseudoLabelGenerator(BaseComponent)
```

PseudoLabelGenerator is a component that creates Generative Pseudo Labeling (GPL) training data for the
training of dense retrievers.

GPL is an unsupervised domain adaptation method for the training of dense retrievers. It is based on question
generation and pseudo labelling with powerful cross-encoders. To train a domain-adapted model, it needs access
to an unlabeled target corpus, usually through DocumentStore and a Retriever to mine for negatives.

For more details, see [GPL](https://github.com/UKPLab/gpl).

For example:


```python
|   document_store = DocumentStore(...)
|   retriever = Retriever(...)
|   qg = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1")
|   plg = PseudoLabelGenerator(qg, retriever)
|   output, output_id = psg.run(documents=document_store.get_all_documents())
|
```

**Notes**:

  
  While the NLP researchers trained the default question
  [generation](https://huggingface.co/doc2query/msmarco-t5-base-v1) and the cross
  [encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) models on
  the English language corpus, we can also use the language-specific question generation and
  cross-encoder models in the target language of our choice to apply GPL to documents in languages
  other than English.
  
  As of this writing, the German language question
  [generation](https://huggingface.co/ml6team/mt5-small-german-query-generation) and the cross
  [encoder](https://huggingface.co/ml6team/cross-encoder-mmarco-german-distilbert-base) models are
  already available, as well as question [generation](https://huggingface.co/doc2query/msmarco-14langs-mt5-base-v1)
  and the cross [encoder](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
  models trained on fourteen languages.

<a id="pseudo_label_generator.PseudoLabelGenerator.__init__"></a>

#### PseudoLabelGenerator.\_\_init\_\_

```python
def __init__(question_producer: Union[QuestionGenerator, List[Dict[str, str]]], retriever: BaseRetriever, cross_encoder_model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_questions_per_document: int = 3, top_k: int = 50, batch_size: int = 16, progress_bar: bool = True, use_auth_token: Optional[Union[str, bool]] = None)
```

Loads the cross-encoder model and prepares PseudoLabelGenerator.

**Arguments**:

- `question_producer` (`Union[QuestionGenerator, List[Dict[str, str]]]`): The question producer used to generate questions or a list of already produced
questions/document pairs in a Dictionary format {"question": "question text ...", "document": "document text ..."}.
- `retriever` (`BaseRetriever`): The Retriever used to query document stores.
- `cross_encoder_model_name_or_path` (`str (optional)`): The path to the cross encoder model, defaults to
`cross-encoder/ms-marco-MiniLM-L-6-v2`.
- `max_questions_per_document` (`int`): The max number of questions generated per document, defaults to 3.
- `top_k` (`int (optional)`): The number of answers retrieved for each question, defaults to 50.
- `batch_size` (`int (optional)`): The number of documents to process at a time.
- `progress_bar` (`bool (optional)`): Whether to show a progress bar, defaults to True.
- `use_auth_token` (`Union[str, bool] (optional)`): The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a id="pseudo_label_generator.PseudoLabelGenerator.generate_questions"></a>

#### PseudoLabelGenerator.generate\_questions

```python
def generate_questions(documents: List[Document], batch_size: Optional[int] = None) -> List[Dict[str, str]]
```

It takes a list of documents and generates a list of question-document pairs.

**Arguments**:

- `documents` (`List[Document]`): A list of documents to generate questions from.
- `batch_size` (`Optional[int]`): The number of documents to process at a time.

**Returns**:

A list of question-document pairs.

<a id="pseudo_label_generator.PseudoLabelGenerator.mine_negatives"></a>

#### PseudoLabelGenerator.mine\_negatives

```python
def mine_negatives(question_doc_pairs: List[Dict[str, str]], batch_size: Optional[int] = None) -> List[Dict[str, str]]
```

Given a list of question and positive document pairs, this function returns a list of question/positive document/negative document

dictionaries.

**Arguments**:

- `question_doc_pairs` (`List[Dict[str, str]]`): A list of question/positive document pairs.
- `batch_size` (`int (optional)`): The number of queries to run in a batch.

**Returns**:

A list of dictionaries, where each dictionary contains the question, positive document,
and negative document.

<a id="pseudo_label_generator.PseudoLabelGenerator.generate_margin_scores"></a>

#### PseudoLabelGenerator.generate\_margin\_scores

```python
def generate_margin_scores(mined_negatives: List[Dict[str, str]], batch_size: Optional[int] = None) -> List[Dict]
```

Given a list of mined negatives, this function predicts the score margin between the positive and negative document using

the cross-encoder.

The function returns a list of examples, where each example is a dictionary with the following keys:

* question: The question string.
* pos_doc: Positive document string (the document containing the answer).
* neg_doc: Negative document string (the document that doesn't contain the answer).
* score: The margin between the score for question-positive document pair and the score for question-negative document pair.

**Arguments**:

- `mined_negatives` (`List[Dict[str, str]]`): The list of mined negatives.
- `batch_size` (`int (optional)`): The number of mined negative lists to run in a batch.

**Returns**:

A list of dictionaries, each of which has the following keys:
- question: The question string
- pos_doc: Positive document string
- neg_doc: Negative document string
- score: The score margin

<a id="pseudo_label_generator.PseudoLabelGenerator.generate_pseudo_labels"></a>

#### PseudoLabelGenerator.generate\_pseudo\_labels

```python
def generate_pseudo_labels(documents: List[Document], batch_size: Optional[int] = None) -> Tuple[dict, str]
```

Given a list of documents, this function generates a list of question-document pairs, mines for negatives, and

scores a positive/negative margin with cross-encoder. The output is the training data for the
adaptation of dense retriever models.

**Arguments**:

- `documents` (`List[Document]`): List[Document] = The list of documents to mine negatives from.
- `batch_size` (`Optional[int]`): The number of documents to process in a batch.

**Returns**:

A dictionary with a single key 'gpl_labels' representing a list of dictionaries, where each
dictionary contains the following keys:
- question: The question string.
- pos_doc: Positive document for the given question.
- neg_doc: Negative document for the given question.
- score: The margin between the score for question-positive document pair and the score for question-negative document pair.

