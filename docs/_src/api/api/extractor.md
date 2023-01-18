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
- `num_workers`: Number of workers to be used in the Pytorch Dataloader
- `flatten_entities_in_meta_data`: If True this converts all entities predicted for a document from a list of
dictionaries into a single list for each key in the dictionary.

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
def forward(model_inputs: Dict[str, Any]) -> Dict[str, Any]
```

Forward step

**Arguments**:

- `model_inputs`: Dictionary of inputs to be given to the model.

<a id="entity.EntityExtractor.postprocess"></a>

#### EntityExtractor.postprocess

```python
def postprocess(model_outputs: Dict[str, Any]) -> List[List[Dict]]
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
- `batch_size`: Number of texts to make predictions on at a time.

<a id="entity.EntityExtractor.extract_batch"></a>

#### EntityExtractor.extract\_batch

```python
def extract_batch(texts: Union[List[str], List[List[str]]],
                  batch_size: int = 1) -> List[List[Dict]]
```

This function allows the extraction of entities out of a list of strings or a list of lists of strings.

The only difference between this function and `self.extract` is that it has additional logic to handle a
list of lists of strings.

**Arguments**:

- `texts`: List of str or list of lists of str to extract entities from.
- `batch_size`: Number of texts to make predictions on at a time.

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

This is a wrapper class to create a Pytorch dataset object from the data attribute of a
`transformers.tokenization_utils_base.BatchEncoding` object.

**Arguments**:

- `model_inputs`: The data attribute of the output from a HuggingFace tokenizer which is needed to evaluate the
forward pass of a token classification model.
