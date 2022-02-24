<a id="evaluator"></a>

# Module evaluator

<a id="evaluator.EvalDocuments"></a>

## EvalDocuments

```python
class EvalDocuments(BaseComponent)
```

This is a pipeline node that should be placed after a node that returns a List of Document, e.g., Retriever or
Ranker, in order to assess its performance. Performance metrics are stored in this class and updated as each
sample passes through it. To view the results of the evaluation, call EvalDocuments.print(). Note that results
from this Node may differ from that when calling Retriever.eval() since that is a closed domain evaluation. Have
a look at our evaluation tutorial for more info about open vs closed domain eval (
https://haystack.deepset.ai/tutorials/evaluation).

EvalDocuments node is deprecated and will be removed in a future version.
Please use pipeline.eval() instead.

<a id="evaluator.EvalDocuments.run"></a>

#### run

```python
def run(documents: List[Document], labels: List[Label], top_k: Optional[int] = None)
```

Run this node on one sample and its labels

<a id="evaluator.EvalDocuments.print"></a>

#### print

```python
def print()
```

Print the evaluation results

<a id="evaluator.EvalAnswers"></a>

## EvalAnswers

```python
class EvalAnswers(BaseComponent)
```

This is a pipeline node that should be placed after a Reader in order to assess the performance of the Reader
individually or to assess the extractive QA performance of the whole pipeline. Performance metrics are stored in
this class and updated as each sample passes through it. To view the results of the evaluation, call EvalAnswers.print().
Note that results from this Node may differ from that when calling Reader.eval()
since that is a closed domain evaluation. Have a look at our evaluation tutorial for more info about
open vs closed domain eval (https://haystack.deepset.ai/tutorials/evaluation).

EvalAnswers node is deprecated and will be removed in a future version.
Please use pipeline.eval() instead.

<a id="evaluator.EvalAnswers.run"></a>

#### run

```python
def run(labels: List[Label], answers: List[Answer], correct_retrieval: bool)
```

Run this node on one sample and its labels

<a id="evaluator.EvalAnswers.print"></a>

#### print

```python
def print(mode)
```

Print the evaluation results

<a id="evaluator.semantic_answer_similarity"></a>

#### semantic\_answer\_similarity

```python
def semantic_answer_similarity(predictions: List[List[str]], gold_labels: List[List[str]], sas_model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> Tuple[List[float], List[float]]
```

Computes Transformer-based similarity of predicted answer to gold labels to derive a more meaningful metric than EM or F1.

Returns per QA pair a) the similarity of the most likely prediction (top 1) to all available gold labels
                    b) the highest similarity of all predictions to gold labels

**Arguments**:

- `predictions`: Predicted answers as list of multiple preds per question
- `gold_labels`: Labels as list of multiple possible answers per question
- `sas_model_name_or_path`: SentenceTransformers semantic textual similarity model, should be path or string
pointing to downloadable models.

**Returns**:

top_1_sas, top_k_sas

