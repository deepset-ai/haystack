<a name="base"></a>
# Module base

<a name="base.BaseClassifier"></a>
## BaseClassifier Objects

```python
class BaseClassifier(BaseComponent)
```

<a name="base.BaseClassifier.timing"></a>
#### timing

```python
 | timing(fn, attr_name)
```

Wrapper method used to time functions.

<a name="farm"></a>
# Module farm

<a name="farm.FARMClassifier"></a>
## FARMClassifier Objects

```python
class FARMClassifier(BaseClassifier)
```

This node classifies documents and adds the output from the classification step to the document's meta data.
The meta field of the document is a dictionary with the following format:
'meta': {'name': '450_Baelor.txt', 'classification': {'label': 'neutral', 'probability' = 0.9997646, ...} }

|  With a FARMClassifier, you can:
 - directly get predictions via predict()
 - fine-tune the model on text classification training data via train()

Usage example:
...
retriever = ElasticsearchRetriever(document_store=document_store)
classifier = FARMClassifier(model_name_or_path="deepset/bert-base-german-cased-sentiment-Germeval17")
p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=classifier, name="Classifier", inputs=["Retriever"])

res = p.run(
    query="Who is the father of Arya Stark?",
    params={"Retriever": {"top_k": 10}, "Classifier": {"top_k": 5}}
)

print(res["documents"][0].to_dict()["meta"]["classification"]["label"])
__Note that print_documents() does not output the content of the classification field in the meta data__

__document_dicts = [doc.to_dict() for doc in res["documents"]]__

__res["documents"] = document_dicts__

__print_documents(res, max_text_len=100)__


<a name="farm.FARMClassifier.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: Union[str, Path], model_version: Optional[str] = None, batch_size: int = 50, use_gpu: bool = True, top_k: int = 10, num_processes: Optional[int] = None, max_seq_len: int = 256, progress_bar: bool = True)
```

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g. 'deepset/bert-base-german-cased-sentiment-Germeval17'.
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

<a name="farm.FARMClassifier.train"></a>
#### train

```python
 | train(data_dir: str, train_filename: str, label_list: List[str], delimiter: str, metric: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, batch_size: int = 10, n_epochs: int = 2, learning_rate: float = 1e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None)
```

Fine-tune a model on a TextClassification dataset.
The dataset needs to be in tabular format (CSV, TSV, etc.), with columns called "label" and "text" in no specific order.
Options:

- Take a plain language model (e.g. `bert-base-cased`) and train it for TextClassification
- Take a TextClassification model and fine-tune it for your domain

**Arguments**:

- `data_dir`: Path to directory containing your training data
- `label_list`: list of labels in the training dataset, e.g., ["0", "1"]
- `delimiter`: delimiter that separates columns in the training dataset, e.g., "\t"
- `metric`: evaluation metric to be used while training, e.g., "f1_macro"
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

<a name="farm.FARMClassifier.update_parameters"></a>
#### update\_parameters

```python
 | update_parameters(max_seq_len: Optional[int] = None)
```

Hot update parameters of a loaded FARMClassifier. It may not to be safe when processing concurrent requests.

<a name="farm.FARMClassifier.save"></a>
#### save

```python
 | save(directory: Path)
```

Saves the FARMClassifier model so that it can be reused at a later point in time.

**Arguments**:

- `directory`: Directory where the FARMClassifier model should be saved

<a name="farm.FARMClassifier.predict_batch"></a>
#### predict\_batch

```python
 | predict_batch(query_doc_list: List[dict], top_k: int = None, batch_size: int = None)
```

Use loaded FARMClassifier model to, for a list of queries, classify each query's supplied list of Document.

Returns list of dictionary of query and list of document sorted by (desc.) similarity with query

**Arguments**:

- `query_doc_list`: List of dictionaries containing queries with their retrieved documents
- `top_k`: The maximum number of answers to return for each query
- `batch_size`: Number of samples the model receives in one batch for inference

**Returns**:

List of dictionaries containing query and list of Document with class probabilities in meta field

<a name="farm.FARMClassifier.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]
```

Use loaded classification model to classify the supplied list of Document.

Returns list of Document enriched with class label and probability, which are stored in Document.meta["classification"]

**Arguments**:

- `query`: Query string (is not used at the moment)
- `documents`: List of Document to be classified
- `top_k`: The maximum number of documents to return

**Returns**:

List of Document with class probabilities in meta field
