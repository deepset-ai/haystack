<a name="base"></a>
# Module base

<a name="base.BaseRanker"></a>
## BaseRanker Objects

```python
class BaseRanker(BaseComponent)
```

<a name="base.BaseRanker.timing"></a>
#### timing

```python
 | timing(fn, attr_name)
```

Wrapper method used to time functions.

<a name="base.BaseRanker.eval"></a>
#### eval

```python
 | eval(label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label", top_k: int = 10, open_domain: bool = False, return_preds: bool = False) -> dict
```

Performs evaluation of the Ranker.
Ranker is evaluated in the same way as a Retriever based on whether it finds the correct document given the query string and at which
position in the ranking of documents the correct document is.

|  Returns a dict containing the following metrics:

    - "recall": Proportion of questions for which correct document is among retrieved documents
    - "mrr": Mean of reciprocal rank. Rewards retrievers that give relevant documents a higher rank.
      Only considers the highest ranked relevant document.
    - "map": Mean of average precision for each question. Rewards retrievers that give relevant
      documents a higher rank. Considers all retrieved relevant documents. If ``open_domain=True``,
      average precision is normalized by the number of retrieved relevant documents per query.
      If ``open_domain=False``, average precision is normalized by the number of all relevant documents
      per query.

**Arguments**:

- `label_index`: Index/Table in DocumentStore where labeled questions are stored
- `doc_index`: Index/Table in DocumentStore where documents that are used for evaluation are stored
- `top_k`: How many documents to return per query
- `open_domain`: If ``True``, retrieval will be evaluated by checking if the answer string to a question is
                    contained in the retrieved docs (common approach in open-domain QA).
                    If ``False``, retrieval uses a stricter evaluation that checks if the retrieved document ids
                    are within ids explicitly stated in the labels.
- `return_preds`: Whether to add predictions in the returned dictionary. If True, the returned dictionary
                     contains the keys "predictions" and "metrics".

<a name="farm"></a>
# Module farm

<a name="farm.FARMRanker"></a>
## FARMRanker Objects

```python
class FARMRanker(BaseRanker)
```

Transformer based model for Document Re-ranking using the TextPairClassifier of FARM framework (https://github.com/deepset-ai/FARM).
While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.

|  With a FARMRanker, you can:

 - directly get predictions via predict()
 - fine-tune the model on TextPair data via train()

<a name="farm.FARMRanker.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: Union[str, Path], model_version: Optional[str] = None, batch_size: int = 50, use_gpu: bool = True, top_k: int = 10, num_processes: Optional[int] = None, max_seq_len: int = 256, progress_bar: bool = True)
```

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
See https://huggingface.co/models for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `batch_size`: Number of samples the model receives in one batch for inference.
                   Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                   to a value so only a single batch is used.
- `use_gpu`: Whether to use GPU (if available)
- `top_k`: The maximum number of documents to return
- `num_processes`: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                      multiprocessing. Set to None to let Inferencer determine optimum number. If you
                      want to debug the Language Model, you might need to disable multiprocessing!
- `max_seq_len`: Max sequence length of one input text for the model
- `progress_bar`: Whether to show a tqdm progress bar or not.
                     Can be helpful to disable in production deployments to keep the logs clean.

<a name="farm.FARMRanker.train"></a>
#### train

```python
 | train(data_dir: str, train_filename: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, batch_size: int = 10, n_epochs: int = 2, learning_rate: float = 1e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None)
```

Fine-tune a model on a TextPairClassification dataset. Options:

- Take a plain language model (e.g. `bert-base-cased`) and train it for TextPairClassification
- Take a TextPairClassification model and fine-tune it for your domain

**Arguments**:

- `data_dir`: Path to directory containing your training data in SQuAD style
- `train_filename`: Filename of training data
- `dev_filename`: Filename of dev / eval data
- `test_filename`: Filename of test data
- `dev_split`: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here
                  that gets split off from training data for eval.
- `use_gpu`: Whether to use GPU (if available)
- `batch_size`: Number of samples the model receives in one batch for training
- `n_epochs`: Number of iterations on the whole training data set
- `learning_rate`: Learning rate of the optimizer
- `max_seq_len`: Maximum text length (in tokens). Everything longer gets cut down.
- `warmup_proportion`: Proportion of training steps until maximum learning rate is reached.
                          Until that point LR is increasing linearly. After that it's decreasing again linearly.
                          Options for different schedules are available in FARM.
- `evaluate_every`: Evaluate the model every X steps on the hold-out eval dataset
- `save_dir`: Path to store the final model
- `num_processes`: The number of processes for `multiprocessing.Pool` during preprocessing.
                      Set to value of 1 to disable multiprocessing. When set to 1, you cannot split away a dev set from train set.
                      Set to None to use all CPU cores minus one.
- `use_amp`: Optimization level of NVIDIA's automatic mixed precision (AMP). The higher the level, the faster the model.
                Available options:
                None (Don't use AMP)
                "O0" (Normal FP32 training)
                "O1" (Mixed Precision => Recommended)
                "O2" (Almost FP16)
                "O3" (Pure FP16).
                See details on: https://nvidia.github.io/apex/amp.html

**Returns**:

None

<a name="farm.FARMRanker.update_parameters"></a>
#### update\_parameters

```python
 | update_parameters(max_seq_len: Optional[int] = None)
```

Hot update parameters of a loaded Ranker. It may not to be safe when processing concurrent requests.

<a name="farm.FARMRanker.save"></a>
#### save

```python
 | save(directory: Path)
```

Saves the Ranker model so that it can be reused at a later point in time.

**Arguments**:

- `directory`: Directory where the Ranker model should be saved

<a name="farm.FARMRanker.predict_batch"></a>
#### predict\_batch

```python
 | predict_batch(query_doc_list: List[dict], top_k: int = None, batch_size: int = None)
```

Use loaded Ranker model to, for a list of queries, rank each query's supplied list of Document.

Returns list of dictionary of query and list of document sorted by (desc.) similarity with query

**Arguments**:

- `query_doc_list`: List of dictionaries containing queries with their retrieved documents
- `top_k`: The maximum number of answers to return for each query
- `batch_size`: Number of samples the model receives in one batch for inference

**Returns**:

List of dictionaries containing query and ranked list of Document

<a name="farm.FARMRanker.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None)
```

Use loaded ranker model to re-rank the supplied list of Document.

Returns list of Document sorted by (desc.) TextPairClassification similarity with the query.

**Arguments**:

- `query`: Query string
- `documents`: List of Document to be re-ranked
- `top_k`: The maximum number of documents to return

**Returns**:

List of Document

