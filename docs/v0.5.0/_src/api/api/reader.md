<a name="farm"></a>
# farm

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
 | __init__(model_name_or_path: Union[str, Path], context_window_size: int = 150, batch_size: int = 50, use_gpu: bool = True, no_ans_boost: Optional[float] = None, top_k_per_candidate: int = 3, top_k_per_sample: int = 1, num_processes: Optional[int] = None, max_seq_len: int = 256, doc_stride: int = 128)
```

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
See https://huggingface.co/models for full list of available models.
- `context_window_size`: The size, in characters, of the window around the answer span that is used when
displaying the context around the answer.
- `batch_size`: Number of samples the model receives in one batch for inference.
Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
to a value so only a single batch is used.
- `use_gpu`: Whether to use GPU (if available)
- `no_ans_boost`: How much the no_answer logit is boosted/increased.
If set to None (default), disables returning "no answer" predictions.
If a negative number, there is a lower chance of "no_answer" being predicted.
If a positive number, there is an increased chance of "no_answer"
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

<a name="farm.FARMReader.train"></a>
#### train

```python
 | train(data_dir: str, train_filename: str, dev_filename: Optional[str] = None, test_filename: Optional[str] = None, use_gpu: Optional[bool] = None, batch_size: int = 10, n_epochs: int = 2, learning_rate: float = 1e-5, max_seq_len: Optional[int] = None, warmup_proportion: float = 0.2, dev_split: float = 0, evaluate_every: int = 300, save_dir: Optional[str] = None, num_processes: Optional[int] = None, use_amp: str = None)
```

Fine-tune a model on a QA dataset. Options:

- Take a plain language model (e.g. `bert-base-cased`) and train it for QA (e.g. on SQuAD data)
- Take a QA model (e.g. `deepset/bert-base-cased-squad2`) and fine-tune it for your domain (e.g. using your labels collected via the haystack annotation tool)

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
 | predict_batch(question_doc_list: List[dict], top_k_per_question: int = None, batch_size: int = None)
```

Use loaded QA model to find answers for a list of questions in each question's supplied list of Document.

Returns list of dictionaries containing answers sorted by (desc.) probability

**Arguments**:

- `question_doc_list`: List of dictionaries containing questions with their retrieved documents
- `top_k_per_question`: The maximum number of answers to return for each question
- `batch_size`: Number of samples the model receives in one batch for inference

**Returns**:

List of dictionaries containing question and answers

<a name="farm.FARMReader.predict"></a>
#### predict

```python
 | predict(question: str, documents: List[Document], top_k: Optional[int] = None)
```

Use loaded QA model to find answers for a question in the supplied list of Document.

Returns dictionaries containing answers sorted by (desc.) probability.
Example:

{'question': 'Who is the father of Arya Stark?',
'answers': [
{'answer': 'Eddard,',
'context': " She travels with her father, Eddard, to King's Landing when he is ",
'offset_answer_start': 147,
'offset_answer_end': 154,
'probability': 0.9787139466668613,
'score': None,
'document_id': '1337'
},
...
]
}

**Arguments**:

- `question`: Question string
- `documents`: List of Document in which to search for the answer
- `top_k`: The maximum number of answers to return

**Returns**:

Dict containing question and answers

<a name="farm.FARMReader.eval_on_file"></a>
#### eval\_on\_file

```python
 | eval_on_file(data_dir: str, test_filename: str, device: str)
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
- `device`: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
:type device: str

<a name="farm.FARMReader.eval"></a>
#### eval

```python
 | eval(document_store: BaseDocumentStore, device: str, label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold_label")
```

Performs evaluation on evaluation documents in the DocumentStore.
Returns a dict containing the following metrics:
- "EM": Proportion of exact matches of predicted answers with their corresponding correct answers
- "f1": Average overlap between predicted answers and their corresponding correct answers
- "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

**Arguments**:

- `document_store`: DocumentStore containing the evaluation documents
- `device`: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
- `label_index`: Index/Table name where labeled questions are stored
- `doc_index`: Index/Table name where documents that are used for evaluation are stored

<a name="farm.FARMReader.predict_on_texts"></a>
#### predict\_on\_texts

```python
 | predict_on_texts(question: str, texts: List[str], top_k: Optional[int] = None)
```

Use loaded QA model to find answers for a question in the supplied list of Document.
Returns dictionaries containing answers sorted by (desc.) probability.
Example:

{
'question': 'Who is the father of Arya Stark?',
'answers':[
{'answer': 'Eddard,',
'context': " She travels with her father, Eddard, to King's Landing when he is ",
'offset_answer_start': 147,
'offset_answer_end': 154,
'probability': 0.9787139466668613,
'score': None,
'document_id': '1337'
},...
]
}

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
# transformers

<a name="transformers.TransformersReader"></a>
## TransformersReader

```python
class TransformersReader(BaseReader)
```

Transformer based model for extractive Question Answering using the HuggingFace's transformers framework
(https://github.com/huggingface/transformers).
While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.

|  With the reader, you can:

    - directly get predictions via predict()

<a name="transformers.TransformersReader.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str = "distilbert-base-uncased-distilled-squad", tokenizer: Optional[str] = None, context_window_size: int = 70, use_gpu: int = 0, top_k_per_candidate: int = 4, return_no_answers: bool = True, max_seq_len: int = 256, doc_stride: int = 128)
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
- `tokenizer`: Name of the tokenizer (usually the same as model)
- `context_window_size`: Num of chars (before and after the answer) to return as "context" for each answer.
The context usually helps users to understand if the answer really makes sense.
- `use_gpu`: If < 0, then use cpu. If >= 0, this is the ordinal of the gpu to use
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
 | predict(question: str, documents: List[Document], top_k: Optional[int] = None)
```

Use loaded QA model to find answers for a question in the supplied list of Document.

Returns dictionaries containing answers sorted by (desc.) probability.
Example:

{'question': 'Who is the father of Arya Stark?',
'answers': [
{'answer': 'Eddard,',
'context': " She travels with her father, Eddard, to King's Landing when he is ",
'offset_answer_start': 147,
'offset_answer_end': 154,
'probability': 0.9787139466668613,
'score': None,
'document_id': '1337'
},
...
]
}

**Arguments**:

- `question`: Question string
- `documents`: List of Document in which to search for the answer
- `top_k`: The maximum number of answers to return

**Returns**:

Dict containing question and answers

<a name="base"></a>
# base

