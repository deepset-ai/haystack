<a id="entity"></a>

# Module entity

<a id="entity.EntityExtractor"></a>

## EntityExtractor

```python
class EntityExtractor(BaseComponent)
```

This node is used to extract entities out of documents.

The most common use case for this would be as a named entity extractor.
The default model used is dslim/bert-base-NER.
This node can be placed in a querying pipeline to perform entity extraction on retrieved documents only,
or it can be placed in an indexing pipeline so that all documents in the document store have extracted entities.
The entities extracted by this Node will populate Document.entities

**Arguments**:

- `model_name_or_path`: The name of the model to use for entity extraction.
- `model_version`: The version of the model to use for entity extraction.
- `use_gpu`: Whether to use the GPU or not.
- `progress_bar`: Whether to show a progress bar or not.
- `batch_size`: The batch size to use for entity extraction.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.
- `aggregation_strategy`: The strategy to fuse (or not) tokens based on the model prediction.
“none”: Will not do any aggregation and simply return raw results from the model.
“simple”: Will attempt to group entities following the default schema.
          (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being
          [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”}, {“word”: “E”, “entity”: “TAG2”}]
          Notice that two consecutive B tags will end up as different entities.
          On word based languages, we might end up splitting words undesirably: Imagine Microsoft being tagged
          as [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}].
          Look at the options FIRST, MAX, and AVERAGE for ways to mitigate this example and disambiguate words
          (on languages that support that meaning, which is basically tokens separated by a space).
          These mitigations will only work on real words, “New york” might still be tagged with two different entities.
“first”: (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with
         different tags. Words will simply use the tag of the first token of the word when there is ambiguity.
“average”: (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with
           different tags. The scores will be averaged across tokens, and then the label with the maximum score is chosen.
“max”: (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with
       different tags. Word entity will simply be the token with the maximum score.
- `add_prefix_space`: Do this if you do not want the first word to be treated differently. This is relevant for
model types such as "bloom", "gpt2", and "roberta".
Explained in more detail here:
https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer

<a id="entity.EntityExtractor.run"></a>

#### EntityExtractor.run

```python
def run(
    documents: Optional[Union[List[Document], List[dict]]] = None
) -> Tuple[Dict, str]
```

This is the method called when this node is used in a pipeline

<a id="entity.EntityExtractor.preprocess"></a>

#### EntityExtractor.preprocess

```python
def preprocess(sentence: Union[str, List[str]],
               offset_mapping: Optional[torch.Tensor] = None)
```

Preprocessing step to tokenize the provided text.

**Arguments**:

- `sentence`: Text to tokenize. This works with a list of texts or a single text.
- `offset_mapping`: Only needed if a slow tokenizer is used. Will be used in the postprocessing step to
determine the original character positions of the detected entities.

<a id="entity.EntityExtractor.forward"></a>

#### EntityExtractor.forward

```python
def forward(model_inputs: Dict[str, Any])
```

Forward step

**Arguments**:

- `model_inputs`: Dictionary of inputs to be given to the model.

<a id="entity.EntityExtractor.postprocess"></a>

#### EntityExtractor.postprocess

```python
def postprocess(model_outputs: Dict[str, Any])
```

Aggregate each of the items in `model_outputs` based on which text document they originally came from.

Then we pass the grouped `model_outputs` to `self.extractor_pipeline.postprocess` to take advantage of the
advanced postprocessing features available in the HuggingFace TokenClassificationPipeline object.

**Arguments**:

- `model_outputs`: Dictionary of model outputs

<a id="entity.EntityExtractor.extract"></a>

#### EntityExtractor.extract

```python
def extract(text: Union[str, List[str]], batch_size: int = 1)
```

This function can be called to perform entity extraction when using the node in isolation.

**Arguments**:

- `text`: Text to extract entities from. Can be a str or a List of str.
- `batch_size`: 

<a id="entity.EntityExtractor.extract_batch"></a>

#### EntityExtractor.extract\_batch

```python
def extract_batch(texts: Union[List[str], List[List[str]]],
                  batch_size: Optional[int] = None)
```

This function allows the extraction of entities out of a list of strings or a list of lists of strings.

The only difference between this function and `self.extract` is that it has additional logic to handle a
list of lists of strings.

**Arguments**:

- `texts`: List of str or list of lists of str to extract entities from.
- `batch_size`: Number of texts to make predictions on at a time.

<a id="entity.EntityExtractor.train"></a>

#### EntityExtractor.train

```python
def train(do_eval: bool,
          do_test: bool,
          fp16: bool,
          resume_from_checkpoint: str,
          output_dir: str,
          lr: float,
          batch_size: int,
          epochs: int,
          pad_to_max_length: bool = False,
          train_file: Optional[str] = None,
          validation_file: Optional[str] = None,
          test_file: Optional[str] = None,
          preprocessing_num_workers: Optional[int] = None,
          overwrite_cache: bool = False,
          dataset_name: Optional[str] = None,
          dataset_config_name: Optional[str] = None,
          text_column_name: Optional[str] = None,
          label_column_name: Optional[str] = None,
          cache_dir: str = None,
          label_all_tokens: bool = False,
          return_entity_level_metrics: bool = False,
          max_seq_length: int = None,
          task_name: str = "ner",
          push_to_hub: bool = False)
```

Run NER training which was adapted from

https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

**Arguments**:

- `do_eval`: 
- `do_test`: 
- `fp16`: 
- `resume_from_checkpoint`: 
- `output_dir`: 
- `lr`: 
- `batch_size`: 
- `epochs`: 
- `pad_to_max_length`: Whether to pad all samples to model maximum sentence length.
If False, this function will pad the samples dynamically when batching to the maximum length in the batch.
This dynamic behavior is more efficient on GPUs but performs very poorly on TPUs.
- `train_file`: 
- `validation_file`: 
- `test_file`: 
- `preprocessing_num_workers`: The number of processes to use for the preprocessing.
- `overwrite_cache`: If True overwrite the cached training, evaluation and test datasets.
- `dataset_name`: 
- `dataset_config_name`: 
- `text_column_name`: 
- `label_column_name`: 
- `cache_dir`: Location to store datasets loaded from huggingface
- `label_all_tokens`: Whether to put the label for one word on all tokens of generated by that word or just
on the one (in which case the other tokens will have a padding index).
- `return_entity_level_metrics`: 
- `max_seq_length`: The maximum total input sequence length after tokenization. If set, sequences longer
than this will be truncated, sequences shorter will be padded.
- `task_name`: 
- `push_to_hub`: If True push the model to the HuggingFace model hub.

<a id="entity.EntityExtractor.get_raw_datasets"></a>

#### EntityExtractor.get\_raw\_datasets

```python
def get_raw_datasets(dataset_name: Optional[str] = None,
                     dataset_config_name: Optional[str] = None,
                     cache_dir: Optional[str] = None,
                     train_file: Optional[str] = None,
                     validation_file: Optional[str] = None,
                     test_file: Optional[str] = None) -> DatasetDict
```

Retrieve the datasets. You can either provide your own CSV/JSON/TXT training and evaluation files

or provide the name of one of the public datasets available on HuggingFace at https://huggingface.co/datasets/.

For CSV/JSON files, this function will use the column called 'text' or the first column if no column called
'text' is found.

**Arguments**:

- `dataset_name`: The name of a dataset available on HuggingFace
- `dataset_config_name`: The name of the dataset configuration file on HuggingFace
- `cache_dir`: The directory to read and write data. This defaults to "~/.cache/huggingface/datasets".
- `train_file`: The path to the file with the training data.
- `validation_file`: The path to the file with the validation data.
- `test_file`: The path to the file with the test data.

<a id="entity.NERDataProcessor"></a>

## NERDataProcessor

```python
class NERDataProcessor()
```

<a id="entity.NERDataProcessor.get_labels"></a>

#### NERDataProcessor.get\_labels

```python
def get_labels(raw_datasets, features, label_column_name)
```

If the labels are of type ClassLabel, they are already integers, and we have the map stored somewhere.

Otherwise, we have to get the list of labels manually.

**Arguments**:

- `raw_datasets`: 
- `features`: 
- `label_column_name`: 

<a id="entity.NERDataProcessor.tokenize_and_align_labels"></a>

#### NERDataProcessor.tokenize\_and\_align\_labels

```python
def tokenize_and_align_labels(examples, padding, max_seq_length,
                              text_column_name, label_column_name, label_to_id,
                              label_all_tokens, b_to_i_label)
```

Tokenize all texts and align the labels with them.

**Arguments**:

- `examples`: 
- `padding`: 
- `max_seq_length`: The maximum total input sequence length after tokenization. If set, sequences longer
than this will be truncated, sequences shorter will be padded.
- `text_column_name`: The column name of text to input in the file (a csv or JSON file).
- `label_column_name`: The column name of label to input in the file (a csv or JSON file).
- `label_to_id`: 
- `label_all_tokens`: 
- `b_to_i_label`: 

<a id="entity.NERDataProcessor.preprocess_datasets"></a>

#### NERDataProcessor.preprocess\_datasets

```python
def preprocess_datasets(raw_datasets: DatasetDict,
                        training_args: TrainingArguments,
                        text_column_name: str,
                        label_column_name: str,
                        pad_to_max_length: bool = False,
                        max_seq_length: Optional[int] = None,
                        label_all_tokens: bool = False,
                        preprocessing_num_workers: Optional[int] = None,
                        overwrite_cache: bool = False)
```

Preprocess the raw datasets

**Arguments**:

- `raw_datasets`: 
- `training_args`: 
- `text_column_name`: The column name of text to input in the file (a csv or JSON file).
- `label_column_name`: The column name of label to input in the file (a csv or JSON file).
- `pad_to_max_length`: Whether to pad all samples to model maximum sentence length.
If False, this function will pad the samples dynamically when batching to the maximum length in the batch.
This dynamic behavior is more efficient on GPUs but performs very poorly on TPUs.
- `max_seq_length`: The maximum total input sequence length after tokenization. If set, sequences longer
than this will be truncated, sequences shorter will be padded.
- `label_all_tokens`: Whether to put the label for one word on all tokens of generated by that word or just
on the one (in which case the other tokens will have a padding index).
- `preprocessing_num_workers`: The number of processes to use for the preprocessing.
- `overwrite_cache`: If True overwrite the cached training, evaluation and test datasets.

<a id="entity.NERDataProcessor.get_unique_label_list"></a>

#### NERDataProcessor.get\_unique\_label\_list

```python
@staticmethod
def get_unique_label_list(labels)
```

In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the

unique labels.

**Arguments**:

- `labels`: List of labels in a dataset

<a id="entity.simplify_ner_for_qa"></a>

#### simplify\_ner\_for\_qa

```python
def simplify_ner_for_qa(output)
```

Returns a simplified version of the output dictionary

with the following structure:
[
    {
        answer: { ... }
        entities: [ { ... }, {} ]
    }
]
The entities included are only the ones that overlap with
the answer itself.

**Arguments**:

- `output`: Output from a query pipeline

<a id="entity.TokenClassificationDataset"></a>

## TokenClassificationDataset

```python
class TokenClassificationDataset(Dataset)
```

Token Classification Dataset

This is a wrapper class to create a Pytorch dataset object from a `transformers.tokenization_utils_base.BatchEncoding`
object.

**Arguments**:

- `model_inputs`: The output of a HuggingFace tokenizer that are needed to evaluate the forward pass of a token
classification model.

