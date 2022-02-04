<a name="base"></a>
# Module base

<a name="base.BaseReader"></a>
## BaseReader

```python
class BaseReader(BaseComponent)
```

<a name="base.BaseReader.run_batch"></a>
#### run\_batch

```python
 | run_batch(query_doc_list: List[Dict], top_k: Optional[int] = None)
```

A unoptimized implementation of running Reader queries in batch

<a name="base.BaseReader.timing"></a>
#### timing

```python
 | timing(fn, attr_name)
```

Wrapper method used to time functions.

<a name="farm"></a>
# Module farm

<a name="farm.FARMReader"></a>
## FARMReader

```python
class FARMReader(BaseReader)
```

Transformer based model for extractive Question Answering using the FARM framework (https://github.com/deepset-ai/FARM).
While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.

|  With a FARMReader, you can:

 - directly get predictions via predict()
 - fine-tune the model on QA data via train()

<a name="farm.FARMReader.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str, model_version: Optional[str] = None, context_window_size: int = 150, batch_size: int = 50, use_gpu: bool = True, no_ans_boost: float = 0.0, return_no_answer: bool = False, top_k: int = 10, top_k_per_candidate: int = 3, top_k_per_sample: int = 1, num_processes: Optional[int] = None, max_seq_len: int = 256, doc_stride: int = 128, progress_bar: bool = True, duplicate_filtering: int = 0, use_confidence_scores: bool = True, proxies=None, local_files_only=False, force_download=False, use_auth_token: Optional[Union[str,bool]] = None, **kwargs)
```

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
See https://huggingface.co/models for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `context_window_size`: The size, in characters, of the window around the answer span that is used when
                            displaying the context around the answer.
- `batch_size`: Number of samples the model receives in one batch for inference.
                   Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                   to a value so only a single batch is used.
- `use_gpu`: Whether to use GPU (if available)
- `no_ans_boost`: How much the no_answer logit is boosted/increased.
If set to 0 (default), the no_answer logit is not changed.
If a negative number, there is a lower chance of "no_answer" being predicted.
If a positive number, there is an increased chance of "no_answer"
- `return_no_answer`: Whether to include no_answer predictions in the results.
- `top_k`: The maximum number of answers to return
- `top_k_per_candidate`: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
Note that this is not the number of "final answers" you will receive
(see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
and that FARM includes no_answer in the sorted list of predictions.
- `top_k_per_sample`: How many answers to extract from each small text passage that the model can process at once
(one "candidate doc" is usually split into many smaller "passages").
You usually want a very small value here, as it slows down inference
and you don't gain much of quality by having multiple answers from one passage.
Note that this is not the number of "final answers" you will receive
(see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
and that FARM includes no_answer in the sorted list of predictions.
- `num_processes`: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                      multiprocessing. Set to None to let Inferencer determine optimum number. If you
                      want to debug the Language Model, you might need to disable multiprocessing!
- `max_seq_len`: Max sequence length of one input text for the model
- `doc_stride`: Length of striding window for splitting long texts (used if ``len(text) > max_seq_len``)
- `progress_bar`: Whether to show a tqdm progress bar or not.
                     Can be helpful to disable in production deployments to keep the logs clean.
- `duplicate_filtering`: Answers are filtered based on their position. Both start and end position of the answers are considered.
                            The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.
- `use_confidence_scores`: Sets the type of score that is returned with every predicted answer.
                              `True` => a scaled confidence / relevance score between [0, 1].
                              This score can also be further calibrated on your dataset via self.eval()
                              (see https://haystack.deepset.ai/components/reader#confidence-scores) .
                              `False` => an unscaled, raw score [-inf, +inf] which is the sum of start and end logit
                              from the model for the predicted span.
- `proxies`: Dict of proxy servers to use for downloading external models. Example: {'http': 'some.proxy:1234', 'http://hostname': 'my.proxy:3111'}
- `local_files_only`: Whether to force checking for local files only (and forbid downloads)
- `force_download`: Whether fo force a (re-)download even if the model exists locally in the cache.
- `use_auth_token`: API token used to download private models from Huggingface. If this parameter is set to `True`,
                        the local token will be used, which must be previously created via `transformer-cli login`. 
                        Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a name="farm.FARMReader.train"></a>
#### train

```python
 | train(data_dir: str, train_filename: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, batch_size: int = 10, n_epochs: int = 2, learning_rate: float = 1e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None, checkpoint_root_dir: Path = Path("model_checkpoints"), checkpoint_every: Optional[int] = None, checkpoints_to_keep: int = 3, caching: bool = False, cache_path: Path = Path("cache/data_silo"))
```

Fine-tune a model on a QA dataset. Options:

- Take a plain language model (e.g. `bert-base-cased`) and train it for QA (e.g. on SQuAD data)
- Take a QA model (e.g. `deepset/bert-base-cased-squad2`) and fine-tune it for your domain (e.g. using your labels collected via the haystack annotation tool)

Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.

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
- `checkpoint_root_dir`: the Path of directory where all train checkpoints are saved. For each individual
       checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
- `checkpoint_every`: save a train checkpoint after this many steps of training.
- `checkpoints_to_keep`: maximum number of train checkpoints to save.
:param caching: whether or not to use caching for preprocessed dataset
- `cache_path`: Path to cache the preprocessed dataset

**Returns**:

None

<a name="farm.FARMReader.distil_from"></a>
#### distil\_from

```python
 | distil_from(teacher_model: "FARMReader", data_dir: str, train_filename: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, student_batch_size: int = 10, teacher_batch_size: Optional[int] = None, n_epochs: int = 2, learning_rate: float = 1e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None, checkpoint_root_dir: Path = Path("model_checkpoints"), checkpoint_every: Optional[int] = None, checkpoints_to_keep: int = 3, caching: bool = False, cache_path: Path = Path("cache/data_silo"), distillation_loss_weight: float = 0.5, distillation_loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "kl_div", temperature: float = 1.0)
```

Fine-tune a model on a QA dataset using distillation. You need to provide a teacher model that is already finetuned on the dataset
and a student model that will be trained using the teacher's logits. The idea of this is to increase the accuracy of a lightweight student model
using a more complex teacher.

**Example**
```python
student = FARMReader(model_name_or_path="prajjwal1/bert-medium")
teacher = FARMReader(model_name_or_path="deepset/bert-large-uncased-whole-word-masking-squad2")

student.distil_from(teacher, data_dir="squad2", train_filename="train.json", test_filename="dev.json",
                    learning_rate=3e-5, distillation_loss_weight=1.0, temperature=5)
```

Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.

**Arguments**:

- `teacher_model`: Model whose logits will be used to improve accuracy
- `data_dir`: Path to directory containing your training data in SQuAD style
- `train_filename`: Filename of training data
- `dev_filename`: Filename of dev / eval data
- `test_filename`: Filename of test data
- `dev_split`: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here
                  that gets split off from training data for eval.
- `use_gpu`: Whether to use GPU (if available)
- `student_batch_size`: Number of samples the student model receives in one batch for training
- `student_batch_size`: Number of samples the teacher model receives in one batch for distillation
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
- `checkpoint_root_dir`: the Path of directory where all train checkpoints are saved. For each individual
       checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
- `checkpoint_every`: save a train checkpoint after this many steps of training.
- `checkpoints_to_keep`: maximum number of train checkpoints to save.
:param caching: whether or not to use caching for preprocessed dataset and teacher logits
- `cache_path`: Path to cache the preprocessed dataset and teacher logits
- `distillation_loss_weight`: The weight of the distillation loss. A higher weight means the teacher outputs are more important.
- `distillation_loss`: Specifies how teacher and model logits should be compared. Can either be a string ("mse" for mean squared error or "kl_div" for kl divergence loss) or a callable loss function (needs to have named paramters student_logits and teacher_logits)
- `temperature`: The temperature for distillation. A higher temperature will result in less certainty of teacher outputs. A lower temperature means more certainty. A temperature of 1.0 does not change the certainty of the model.

**Returns**:

None

<a name="farm.FARMReader.update_parameters"></a>
#### update\_parameters

```python
 | update_parameters(context_window_size: Optional[int] = None, no_ans_boost: Optional[float] = None, return_no_answer: Optional[bool] = None, max_seq_len: Optional[int] = None, doc_stride: Optional[int] = None)
```

Hot update parameters of a loaded Reader. It may not to be safe when processing concurrent requests.

<a name="farm.FARMReader.save"></a>
#### save

```python
 | save(directory: Path)
```

Saves the Reader model so that it can be reused at a later point in time.

**Arguments**:

- `directory`: Directory where the Reader model should be saved

<a name="farm.FARMReader.predict_batch"></a>
#### predict\_batch

```python
 | predict_batch(query_doc_list: List[dict], top_k: int = None, batch_size: int = None)
```

Use loaded QA model to find answers for a list of queries in each query's supplied list of Document.

Returns list of dictionaries containing answers sorted by (desc.) score

**Arguments**:

- `query_doc_list`: List of dictionaries containing queries with their retrieved documents
- `top_k`: The maximum number of answers to return for each query
- `batch_size`: Number of samples the model receives in one batch for inference

**Returns**:

List of dictionaries containing query and answers

<a name="farm.FARMReader.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None)
```

Use loaded QA model to find answers for a query in the supplied list of Document.

Returns dictionaries containing answers sorted by (desc.) score.
Example:
 ```python
    |{
    |    'query': 'Who is the father of Arya Stark?',
    |    'answers':[Answer(
    |                 'answer': 'Eddard,',
    |                 'context': "She travels with her father, Eddard, to King's Landing when he is",
    |                 'score': 0.9787139466668613,
    |                 'offsets_in_context': [Span(start=29, end=35],
    |                 'offsets_in_context': [Span(start=347, end=353],
    |                 'document_id': '88d1ed769d003939d3a0d28034464ab2'
    |                 ),...
    |              ]
    |}
```

**Arguments**:

- `query`: Query string
- `documents`: List of Document in which to search for the answer
- `top_k`: The maximum number of answers to return

**Returns**:

Dict containing query and answers

<a name="farm.FARMReader.eval_on_file"></a>
#### eval\_on\_file

```python
 | eval_on_file(data_dir: str, test_filename: str, device: Optional[str] = None)
```

Performs evaluation on a SQuAD-formatted file.
Returns a dict containing the following metrics:
    - "EM": exact match score
    - "f1": F1-Score
    - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

**Arguments**:

- `data_dir`: The directory in which the test set can be found
:type data_dir: Path or str
- `test_filename`: The name of the file containing the test data in SQuAD format.
:type test_filename: str
- `device`: The device on which the tensors should be processed. Choose from "cpu" and "cuda" or use the Reader's device by default.
:type device: str

<a name="farm.FARMReader.eval"></a>
#### eval

```python
 | eval(document_store: BaseDocumentStore, device: Optional[str] = None, label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold-label", calibrate_conf_scores: bool = False)
```

Performs evaluation on evaluation documents in the DocumentStore.
Returns a dict containing the following metrics:
      - "EM": Proportion of exact matches of predicted answers with their corresponding correct answers
      - "f1": Average overlap between predicted answers and their corresponding correct answers
      - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

**Arguments**:

- `document_store`: DocumentStore containing the evaluation documents
- `device`: The device on which the tensors should be processed. Choose from "cpu" and "cuda" or use the Reader's device by default.
- `label_index`: Index/Table name where labeled questions are stored
- `doc_index`: Index/Table name where documents that are used for evaluation are stored
- `label_origin`: Field name where the gold labels are stored
- `calibrate_conf_scores`: Whether to calibrate the temperature for temperature scaling of the confidence scores

<a name="farm.FARMReader.calibrate_confidence_scores"></a>
#### calibrate\_confidence\_scores

```python
 | calibrate_confidence_scores(document_store: BaseDocumentStore, device: Optional[str] = None, label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label")
```

Calibrates confidence scores on evaluation documents in the DocumentStore.

**Arguments**:

- `document_store`: DocumentStore containing the evaluation documents
- `device`: The device on which the tensors should be processed. Choose from "cpu" and "cuda" or use the Reader's device by default.
- `label_index`: Index/Table name where labeled questions are stored
- `doc_index`: Index/Table name where documents that are used for evaluation are stored
- `label_origin`: Field name where the gold labels are stored

<a name="farm.FARMReader.predict_on_texts"></a>
#### predict\_on\_texts

```python
 | predict_on_texts(question: str, texts: List[str], top_k: Optional[int] = None)
```

Use loaded QA model to find answers for a question in the supplied list of Document.
Returns dictionaries containing answers sorted by (desc.) score.
Example:
 ```python
    |{
    |    'question': 'Who is the father of Arya Stark?',
    |    'answers':[
    |                 {'answer': 'Eddard,',
    |                 'context': " She travels with her father, Eddard, to King's Landing when he is ",
    |                 'offset_answer_start': 147,
    |                 'offset_answer_end': 154,
    |                 'score': 0.9787139466668613,
    |                 'document_id': '1337'
    |                 },...
    |              ]
    |}
```

**Arguments**:

- `question`: Question string
- `documents`: List of documents as string type
- `top_k`: The maximum number of answers to return

**Returns**:

Dict containing question and answers

<a name="farm.FARMReader.convert_to_onnx"></a>
#### convert\_to\_onnx

```python
 | @classmethod
 | convert_to_onnx(cls, model_name: str, output_path: Path, convert_to_float16: bool = False, quantize: bool = False, task_type: str = "question_answering", opset_version: int = 11)
```

Convert a PyTorch BERT model to ONNX format and write to ./onnx-export dir. The converted ONNX model
can be loaded with in the `FARMReader` using the export path as `model_name_or_path` param.

Usage:

    `from haystack.reader.farm import FARMReader
    from pathlib import Path
    onnx_model_path = Path("roberta-onnx-model")
    FARMReader.convert_to_onnx(model_name="deepset/bert-base-cased-squad2", output_path=onnx_model_path)
    reader = FARMReader(onnx_model_path)`

**Arguments**:

- `model_name`: transformers model name
- `output_path`: Path to output the converted model
- `convert_to_float16`: Many models use float32 precision by default. With the half precision of float16,
                           inference is faster on Nvidia GPUs with Tensor core like T4 or V100. On older GPUs,
                           float32 could still be be more performant.
- `quantize`: convert floating point number to integers
- `task_type`: Type of task for the model. Available options: "question_answering" or "embeddings".
- `opset_version`: ONNX opset version

<a name="transformers"></a>
# Module transformers

<a name="transformers.TransformersReader"></a>
## TransformersReader

```python
class TransformersReader(BaseReader)
```

Transformer based model for extractive Question Answering using the HuggingFace's transformers framework
(https://github.com/huggingface/transformers).
While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
With this reader, you can directly get predictions via predict()

<a name="transformers.TransformersReader.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str = "distilbert-base-uncased-distilled-squad", model_version: Optional[str] = None, tokenizer: Optional[str] = None, context_window_size: int = 70, use_gpu: bool = True, top_k: int = 10, top_k_per_candidate: int = 4, return_no_answers: bool = True, max_seq_len: int = 256, doc_stride: int = 128)
```

Load a QA model from Transformers.
Available models include:

- ``'distilbert-base-uncased-distilled-squad`'``
- ``'bert-large-cased-whole-word-masking-finetuned-squad``'
- ``'bert-large-uncased-whole-word-masking-finetuned-squad``'

See https://huggingface.co/models for full list of available QA models

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
See https://huggingface.co/models for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `tokenizer`: Name of the tokenizer (usually the same as model)
- `context_window_size`: Num of chars (before and after the answer) to return as "context" for each answer.
                            The context usually helps users to understand if the answer really makes sense.
- `use_gpu`: Whether to use GPU (if available).
- `top_k`: The maximum number of answers to return
- `top_k_per_candidate`: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
Note that this is not the number of "final answers" you will receive
(see `top_k` in TransformersReader.predict() or Finder.get_answers() for that)
and that no_answer can be included in the sorted list of predictions.
- `return_no_answers`: If True, the HuggingFace Transformers model could return a "no_answer" (i.e. when there is an unanswerable question)
If False, it cannot return a "no_answer". Note that `no_answer_boost` is unfortunately not available with TransformersReader.
If you would like to set no_answer_boost, use a `FARMReader`.
- `max_seq_len`: max sequence length of one input text for the model
- `doc_stride`: length of striding window for splitting long texts (used if len(text) > max_seq_len)

<a name="transformers.TransformersReader.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None)
```

Use loaded QA model to find answers for a query in the supplied list of Document.

Returns dictionaries containing answers sorted by (desc.) score.
Example:

 ```python
    |{
    |    'query': 'Who is the father of Arya Stark?',
    |    'answers':[
    |                 {'answer': 'Eddard,',
    |                 'context': " She travels with her father, Eddard, to King's Landing when he is ",
    |                 'offset_answer_start': 147,
    |                 'offset_answer_end': 154,
    |                 'score': 0.9787139466668613,
    |                 'document_id': '1337'
    |                 },...
    |              ]
    |}
```

**Arguments**:

- `query`: Query string
- `documents`: List of Document in which to search for the answer
- `top_k`: The maximum number of answers to return

**Returns**:

Dict containing query and answers

<a name="table"></a>
# Module table

<a name="table.TableReader"></a>
## TableReader

```python
class TableReader(BaseReader)
```

Transformer-based model for extractive Question Answering on Tables with TaPas
using the HuggingFace's transformers framework (https://github.com/huggingface/transformers).
With this reader, you can directly get predictions via predict()

**Example**:

```python
from haystack import Document
from haystack.reader import TableReader
import pandas as pd

table_reader = TableReader(model_name_or_path="google/tapas-base-finetuned-wtq")
data = {
    "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
    "age": ["57", "46", "60"],
    "number of movies": ["87", "53", "69"],
    "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
}
table = pd.DataFrame(data)
document = Document(content=table, content_type="table")
query = "When was DiCaprio born?"
prediction = table_reader.predict(query=query, documents=[document])
answer = prediction["answers"][0].answer  # "10 june 1996"
```

<a name="table.TableReader.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str = "google/tapas-base-finetuned-wtq", model_version: Optional[str] = None, tokenizer: Optional[str] = None, use_gpu: bool = True, top_k: int = 10, max_seq_len: int = 256)
```

Load a TableQA model from Transformers.
Available models include:

- ``'google/tapas-base-finetuned-wtq`'``
- ``'google/tapas-base-finetuned-wikisql-supervised``'

See https://huggingface.co/models?pipeline_tag=table-question-answering
for full list of available TableQA models.

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g.
See https://huggingface.co/models?pipeline_tag=table-question-answering for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `tokenizer`: Name of the tokenizer (usually the same as model)
- `use_gpu`: Whether to use GPU or CPU. Falls back on CPU if no GPU is available.
- `top_k`: The maximum number of answers to return
- `max_seq_len`: Max sequence length of one input table for the model. If the number of tokens of
                    query + table exceed max_seq_len, the table will be truncated by removing rows until the
                    input size fits the model.

<a name="table.TableReader.predict"></a>
#### predict

```python
 | predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
```

Use loaded TableQA model to find answers for a query in the supplied list of Documents
of content_type ``'table'``.

Returns dictionary containing query and list of Answer objects sorted by (desc.) score.
WARNING: The answer scores are not reliable, as they are always extremely high, even if
         a question cannot be answered by a given table.

**Arguments**:

- `query`: Query string
- `documents`: List of Document in which to search for the answer. Documents should be
                  of content_type ``'table'``.
- `top_k`: The maximum number of answers to return

**Returns**:

Dict containing query and answers

