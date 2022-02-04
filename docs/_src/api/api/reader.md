<a id="base"></a>

# Module base

<a id="base.BaseReader"></a>

## BaseReader

```python
class BaseReader(BaseComponent)
```

<a id="base.BaseReader.run_batch"></a>

#### run\_batch

```python
def run_batch(query_doc_list: List[Dict], top_k: Optional[int] = None)
```

A unoptimized implementation of running Reader queries in batch

<a id="base.BaseReader.timing"></a>

#### timing

```python
def timing(fn, attr_name)
```

Wrapper method used to time functions.

<a id="farm"></a>

# Module farm

<a id="farm.FARMReader"></a>

## FARMReader

```python
class FARMReader(BaseReader)
```

Transformer based model for extractive Question Answering using the FARM framework (https://github.com/deepset-ai/FARM).
While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.

|  With a FARMReader, you can:

 - directly get predictions via predict()
 - fine-tune the model on QA data via train()

<a id="farm.FARMReader.train"></a>

#### train

```python
def train(data_dir: str, train_filename: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, batch_size: int = 10, n_epochs: int = 2, learning_rate: float = 1e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None, checkpoint_root_dir: Path = Path("model_checkpoints"), checkpoint_every: Optional[int] = None, checkpoints_to_keep: int = 3, caching: bool = False, cache_path: Path = Path("cache/data_silo"))
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
- `caching`: whether or not to use caching for preprocessed dataset
- `cache_path`: Path to cache the preprocessed dataset
- `processor`: The processor to use for preprocessing. If None, the default SquadProcessor is used.

**Returns**:

None

<a id="farm.FARMReader.distil_prediction_layer_from"></a>

#### distil\_prediction\_layer\_from

```python
def distil_prediction_layer_from(teacher_model: "FARMReader", data_dir: str, train_filename: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, student_batch_size: int = 10, teacher_batch_size: Optional[int] = None, n_epochs: int = 2, learning_rate: float = 3e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None, checkpoint_root_dir: Path = Path("model_checkpoints"), checkpoint_every: Optional[int] = None, checkpoints_to_keep: int = 3, caching: bool = False, cache_path: Path = Path("cache/data_silo"), distillation_loss_weight: float = 0.5, distillation_loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "kl_div", temperature: float = 1.0)
```

Fine-tune a model on a QA dataset using logit-based distillation. You need to provide a teacher model that is already finetuned on the dataset

and a student model that will be trained using the teacher's logits. The idea of this is to increase the accuracy of a lightweight student model.
using a more complex teacher.
Originally proposed in: https://arxiv.org/pdf/1503.02531.pdf
This can also be considered as the second stage of distillation finetuning as described in the TinyBERT paper:
https://arxiv.org/pdf/1909.10351.pdf
**Example**
```python
student = FARMReader(model_name_or_path="prajjwal1/bert-medium")
teacher = FARMReader(model_name_or_path="deepset/bert-large-uncased-whole-word-masking-squad2")
student.distil_prediction_layer_from(teacher, data_dir="squad2", train_filename="train.json", test_filename="dev.json",
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
- `caching`: whether or not to use caching for preprocessed dataset and teacher logits
- `cache_path`: Path to cache the preprocessed dataset and teacher logits
- `distillation_loss_weight`: The weight of the distillation loss. A higher weight means the teacher outputs are more important.
- `distillation_loss`: Specifies how teacher and model logits should be compared. Can either be a string ("mse" for mean squared error or "kl_div" for kl divergence loss) or a callable loss function (needs to have named parameters student_logits and teacher_logits)
- `temperature`: The temperature for distillation. A higher temperature will result in less certainty of teacher outputs. A lower temperature means more certainty. A temperature of 1.0 does not change the certainty of the model.
- `tinybert_loss`: Whether to use the TinyBERT loss function for distillation. This requires the student to be a TinyBERT model and the teacher to be a finetuned version of bert-base-uncased.
- `tinybert_epochs`: Number of epochs to train the student model with the TinyBERT loss function. After this many epochs, the student model is trained with the regular distillation loss function.
- `tinybert_learning_rate`: Learning rate to use when training the student model with the TinyBERT loss function.
- `tinybert_train_filename`: Filename of training data to use when training the student model with the TinyBERT loss function. To best follow the original paper, this should be an augmented version of the training data created using the augment_squad.py script. If not specified, the training data from the original training is used.
- `processor`: The processor to use for preprocessing. If None, the default SquadProcessor is used.

**Returns**:

None

<a id="farm.FARMReader.distil_intermediate_layers_from"></a>

#### distil\_intermediate\_layers\_from

```python
def distil_intermediate_layers_from(teacher_model: "FARMReader", data_dir: str, train_filename: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, batch_size: int = 10, n_epochs: int = 5, learning_rate: float = 5e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None, checkpoint_root_dir: Path = Path("model_checkpoints"), checkpoint_every: Optional[int] = None, checkpoints_to_keep: int = 3, caching: bool = False, cache_path: Path = Path("cache/data_silo"), distillation_loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "mse", temperature: float = 1.0, processor: Optional[Processor] = None)
```

The first stage of distillation finetuning as described in the TinyBERT paper:

https://arxiv.org/pdf/1909.10351.pdf
**Example**
```python
student = FARMReader(model_name_or_path="prajjwal1/bert-medium")
teacher = FARMReader(model_name_or_path="huawei-noah/TinyBERT_General_6L_768D")
student.distil_intermediate_layers_from(teacher, data_dir="squad2", train_filename="train.json", test_filename="dev.json",
                    learning_rate=3e-5, distillation_loss_weight=1.0, temperature=5)
```

Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.

**Arguments**:

- `teacher_model`: Model whose logits will be used to improve accuracy
- `data_dir`: Path to directory containing your training data in SQuAD style
- `train_filename`: Filename of training data. To best follow the original paper, this should be an augmented version of the training data created using the augment_squad.py script
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
- `caching`: whether or not to use caching for preprocessed dataset and teacher logits
- `cache_path`: Path to cache the preprocessed dataset and teacher logits
- `distillation_loss_weight`: The weight of the distillation loss. A higher weight means the teacher outputs are more important.
- `distillation_loss`: Specifies how teacher and model logits should be compared. Can either be a string ("mse" for mean squared error or "kl_div" for kl divergence loss) or a callable loss function (needs to have named parameters student_logits and teacher_logits)
- `temperature`: The temperature for distillation. A higher temperature will result in less certainty of teacher outputs. A lower temperature means more certainty. A temperature of 1.0 does not change the certainty of the model.
- `processor`: The processor to use for preprocessing. If None, the default SquadProcessor is used.

**Returns**:

None

<a id="farm.FARMReader.update_parameters"></a>

#### update\_parameters

```python
def update_parameters(context_window_size: Optional[int] = None, no_ans_boost: Optional[float] = None, return_no_answer: Optional[bool] = None, max_seq_len: Optional[int] = None, doc_stride: Optional[int] = None)
```

Hot update parameters of a loaded Reader. It may not to be safe when processing concurrent requests.

<a id="farm.FARMReader.save"></a>

#### save

```python
def save(directory: Path)
```

Saves the Reader model so that it can be reused at a later point in time.

**Arguments**:

- `directory`: Directory where the Reader model should be saved

<a id="farm.FARMReader.predict_batch"></a>

#### predict\_batch

```python
def predict_batch(query_doc_list: List[dict], top_k: int = None, batch_size: int = None)
```

Use loaded QA model to find answers for a list of queries in each query's supplied list of Document.

Returns list of dictionaries containing answers sorted by (desc.) score

**Arguments**:

- `query_doc_list`: List of dictionaries containing queries with their retrieved documents
- `top_k`: The maximum number of answers to return for each query
- `batch_size`: Number of samples the model receives in one batch for inference

**Returns**:

List of dictionaries containing query and answers

<a id="farm.FARMReader.predict"></a>

#### predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None)
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

<a id="farm.FARMReader.eval_on_file"></a>

#### eval\_on\_file

```python
def eval_on_file(data_dir: str, test_filename: str, device: Optional[str] = None)
```

Performs evaluation on a SQuAD-formatted file.

Returns a dict containing the following metrics:
    - "EM": exact match score
    - "f1": F1-Score
    - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

**Arguments**:

- `data_dir` (`Path or str`): The directory in which the test set can be found
- `test_filename` (`str`): The name of the file containing the test data in SQuAD format.
- `device` (`str`): The device on which the tensors should be processed. Choose from "cpu" and "cuda" or use the Reader's device by default.

<a id="farm.FARMReader.eval"></a>

#### eval

```python
def eval(document_store: BaseDocumentStore, device: Optional[str] = None, label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold-label", calibrate_conf_scores: bool = False)
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

<a id="farm.FARMReader.calibrate_confidence_scores"></a>

#### calibrate\_confidence\_scores

```python
def calibrate_confidence_scores(document_store: BaseDocumentStore, device: Optional[str] = None, label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label")
```

Calibrates confidence scores on evaluation documents in the DocumentStore.

**Arguments**:

- `document_store`: DocumentStore containing the evaluation documents
- `device`: The device on which the tensors should be processed. Choose from "cpu" and "cuda" or use the Reader's device by default.
- `label_index`: Index/Table name where labeled questions are stored
- `doc_index`: Index/Table name where documents that are used for evaluation are stored
- `label_origin`: Field name where the gold labels are stored

<a id="farm.FARMReader.predict_on_texts"></a>

#### predict\_on\_texts

```python
def predict_on_texts(question: str, texts: List[str], top_k: Optional[int] = None)
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

<a id="farm.FARMReader.convert_to_onnx"></a>

#### convert\_to\_onnx

```python
@classmethod
def convert_to_onnx(cls, model_name: str, output_path: Path, convert_to_float16: bool = False, quantize: bool = False, task_type: str = "question_answering", opset_version: int = 11)
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

<a id="transformers"></a>

# Module transformers

<a id="transformers.TransformersReader"></a>

## TransformersReader

```python
class TransformersReader(BaseReader)
```

Transformer based model for extractive Question Answering using the HuggingFace's transformers framework
(https://github.com/huggingface/transformers).
While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
With this reader, you can directly get predictions via predict()

<a id="transformers.TransformersReader.predict"></a>

#### predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None)
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

<a id="table"></a>

# Module table

<a id="table.TableReader"></a>

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

<a id="table.TableReader.predict"></a>

#### predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
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

<a id="table.RCIReader"></a>

## RCIReader

```python
class RCIReader(BaseReader)
```

Table Reader model based on Glass et al. (2021)'s Row-Column-Intersection model.
See the original paper for more details:
Glass, Michael, et al. (2021): "Capturing Row and Column Semantics in Transformer Based Question Answering over Tables"
(https://aclanthology.org/2021.naacl-main.96/)

Each row and each column is given a score with regard to the query by two separate models. The score of each cell
is then calculated as the sum of the corresponding row score and column score. Accordingly, the predicted answer is
the cell with the highest score.

Pros and Cons of RCIReader compared to TableReader:
+ Provides meaningful confidence scores
+ Allows larger tables as input
- Does not support aggregation over table cells
- Slower

<a id="table.RCIReader.predict"></a>

#### predict

```python
def predict(query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict
```

Use loaded RCI models to find answers for a query in the supplied list of Documents

of content_type ``'table'``.

Returns dictionary containing query and list of Answer objects sorted by (desc.) score.
The existing RCI models on the HF model hub don"t allow aggregation, therefore, the answer will always be
composed of a single cell.

**Arguments**:

- `query`: Query string
- `documents`: List of Document in which to search for the answer. Documents should be
of content_type ``'table'``.
- `top_k`: The maximum number of answers to return

**Returns**:

Dict containing query and answers

