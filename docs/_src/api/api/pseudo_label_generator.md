<a id="pseudo_label_generator"></a>

# Module pseudo\_label\_generator

<a id="pseudo_label_generator.PseudoLabelGenerator"></a>

## PseudoLabelGenerator

```python
class PseudoLabelGenerator(BaseComponent)
```

The PseudoLabelGenerator is a component that creates Generative Pseudo Labeling (GPL) training data for the
training of dense retrievers.

GPL is an unsupervised domain adaptation method for the training of dense retrievers. It is based on question
generation and pseudo labelling with powerful cross-encoders. To train a domain-adapted model, it needs access
to an unlabeled target corpus, usually via DocumentStore and a retriever to mine for negatives.

For more details see [https://github.com/UKPLab/gpl](https://github.com/UKPLab/gpl)

For example:

```python
|   document_store = DocumentStore(...)
|   retriever = Retriever(...)
|   qg = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1")
|   plg = PseudoLabelGenerator(qg, retriever)
|   output, output_id = psg.run(documents=document_store.get_all_documents())
|
```

<a id="pseudo_label_generator.PseudoLabelGenerator.__init__"></a>

#### PseudoLabelGenerator.\_\_init\_\_

```python
def __init__(question_producer: Union[QuestionGenerator, List[Dict[str, str]]], retriever: BaseRetriever, cross_encoder_model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_questions_per_document: int = 3, top_k: int = 50, batch_size: int = 16, progress_bar: bool = True)
```

Loads the cross encoder model and prepares PseudoLabelGenerator.

**Arguments**:

- `question_producer` (`Union[QuestionGenerator, List[Dict[str, str]]]`): The question producer used to generate questions or a list of already produced
questions/document pairs in Dict format {"question": "question text ...", "document": "document text ..."}.
- `retriever` (`BaseRetriever`): The retriever used to query document stores
- `cross_encoder_model_name_or_path` (`str (optional)`): The path to the cross encoder model, defaults to
cross-encoder/ms-marco-MiniLM-L-6-v2
- `max_questions_per_document` (`int`): The max number of questions generated per document, defaults to 3
- `top_k` (`int (optional)`): The number of answers retrieved for each question, defaults to 50
- `batch_size` (`int (optional)`): Number of documents to process at a time

<a id="pseudo_label_generator.PseudoLabelGenerator.generate_questions"></a>

#### PseudoLabelGenerator.generate\_questions

```python
def generate_questions(documents: List[Document], batch_size: Optional[int] = None) -> List[Dict[str, str]]
```

It takes a list of documents and generates a list of question-document pairs.

**Arguments**:

- `documents` (`List[Document]`): A list of documents to generate questions from
- `batch_size` (`Optional[int]`): Number of documents to process at a time.

**Returns**:

A list of question-document pairs.

<a id="pseudo_label_generator.PseudoLabelGenerator.mine_negatives"></a>

#### PseudoLabelGenerator.mine\_negatives

```python
def mine_negatives(question_doc_pairs: List[Dict[str, str]], batch_size: Optional[int] = None) -> List[Dict[str, str]]
```

Given a list of question and pos_doc pairs, this function returns a list of question/pos_doc/neg_doc

dictionaries.

**Arguments**:

- `question_doc_pairs` (`List[Dict[str, str]]`): A list of question/pos_doc pairs
- `batch_size` (`int (optional)`): The number of queries to run in a batch

**Returns**:

A list of dictionaries, where each dictionary contains the question, positive document,
and negative document.

<a id="pseudo_label_generator.PseudoLabelGenerator.generate_margin_scores"></a>

#### PseudoLabelGenerator.generate\_margin\_scores

```python
def generate_margin_scores(mined_negatives: List[Dict[str, str]], batch_size: Optional[int] = None) -> List[Dict]
```

Given a list of mined negatives, predict the score margin between the positive and negative document using

the cross encoder.

The function returns a list of examples, where each example is a dictionary with the following keys:

* question: the question string
* pos_doc: the positive document string
* neg_doc: the negative document string
* score: the score margin

**Arguments**:

- `mined_negatives` (`List[Dict[str, str]]`): List of mined negatives
- `batch_size` (`int (optional)`): The number of mined negative lists to run in a batch

**Returns**:

A list of dictionaries, each of which has the following keys:
- question: The question string
- pos_doc: The positive document string
- neg_doc: The negative document string
- score: The score margin

<a id="pseudo_label_generator.PseudoLabelGenerator.generate_pseudo_labels"></a>

#### PseudoLabelGenerator.generate\_pseudo\_labels

```python
def generate_pseudo_labels(documents: List[Document], batch_size: Optional[int] = None) -> Tuple[dict, str]
```

Given a list of documents, generate a list of question-document pairs, mine for negatives, and

score positive/negative margin with cross-encoder. The output is the training data for the
adaptation of dense retriever models.

**Arguments**:

- `documents` (`List[Document]`): List[Document] = List of documents to mine negatives from
- `batch_size` (`Optional[int]`): The number of documents to process in a batch

**Returns**:

A dictionary with a single key 'gpl_labels' representing a list of dictionaries, where each
dictionary contains the following keys:
- question: the question
- pos_doc: the positive document for the given question
- neg_doc: the negative document for the given question
- score: the margin score (a float)

