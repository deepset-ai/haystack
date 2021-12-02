<a name="evaluator"></a>
# Module evaluator

<a name="evaluator.EvalDocuments"></a>
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

<a name="evaluator.EvalDocuments.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(debug: bool = False, open_domain: bool = True, top_k: int = 10)
```

**Arguments**:

- `open_domain`: When True, a document is considered correctly retrieved so long as the answer string can be found within it.
                    When False, correct retrieval is evaluated based on document_id.
- `debug`: When True, a record of each sample and its evaluation will be stored in EvalDocuments.log
- `top_k`: calculate eval metrics for top k results, e.g., recall@k

<a name="evaluator.EvalDocuments.run"></a>
#### run

```python
 | run(documents: List[Document], labels: List[Label], top_k: Optional[int] = None)
```

Run this node on one sample and its labels

<a name="evaluator.EvalDocuments.print"></a>
#### print

```python
 | print()
```

Print the evaluation results

<a name="evaluator.EvalAnswers"></a>
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

<a name="evaluator.EvalAnswers.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(skip_incorrect_retrieval: bool = True, open_domain: bool = True, sas_model: str = None, debug: bool = False)
```

**Arguments**:

- `skip_incorrect_retrieval`: When set to True, this eval will ignore the cases where the retriever returned no correct documents
- `open_domain`: When True, extracted answers are evaluated purely on string similarity rather than the position of the extracted answer
- `sas_model`: Name or path of "Semantic Answer Similarity (SAS) model". When set, the model will be used to calculate similarity between predictions and labels and generate the SAS metric.
                  The SAS metric correlates better with human judgement of correct answers as it does not rely on string overlaps.
                  Example: Prediction = "30%", Label = "thirty percent", EM and F1 would be overly pessimistic with both being 0, while SAS paints a more realistic picture.
                  More info in the paper: https://arxiv.org/abs/2108.06130
                  Models:
                  - You can use Bi Encoders (sentence transformers) or cross encoders trained on Semantic Textual Similarity (STS) data.
                    Not all cross encoders can be used because of different return types.
                    If you use custom cross encoders please make sure they work with sentence_transformers.CrossEncoder class
                  - Good default for multiple languages: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                  - Large, powerful, but slow model for English only: "cross-encoder/stsb-roberta-large"
                  - Large model for German only: "deepset/gbert-large-sts"
- `debug`: When True, a record of each sample and its evaluation will be stored in EvalAnswers.log

<a name="evaluator.EvalAnswers.run"></a>
#### run

```python
 | run(labels: List[Label], answers: List[Answer], correct_retrieval: bool)
```

Run this node on one sample and its labels

<a name="evaluator.EvalAnswers.print"></a>
#### print

```python
 | print(mode)
```

Print the evaluation results

<a name="evaluator.semantic_answer_similarity"></a>
#### semantic\_answer\_similarity

```python
semantic_answer_similarity(predictions: List[List[str]], gold_labels: List[List[str]], sas_model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> Tuple[List[float],List[float]]
```

Computes Transformer-based similarity of predicted answer to gold labels to derive a more meaningful metric than EM or F1.
Returns per QA pair a) the similarity of the most likely prediction (top 1) to all available gold labels
                    b) the highest similarity of all predictions to gold labels

**Arguments**:

- `predictions`: Predicted answers as list of multiple preds per question
- `gold_labels`: Labels as list of multiple possible answers per question
- `sas_model_name_or_path`: SentenceTransformers semantic textual similarity model, should be path or string
                                 pointing to downloadable models.


:return top_1_sas, top_k_sas

