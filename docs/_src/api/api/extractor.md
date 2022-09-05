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
- `use_gpu`: Whether to use the GPU or not.
- `batch_size`: The batch size to use for entity extraction.
- `progress_bar`: Whether to show a progress bar or not.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.

<a id="entity.EntityExtractor.run"></a>

#### EntityExtractor.run

```python
def run(
    documents: Optional[Union[List[Document], List[dict]]] = None
) -> Tuple[Dict, str]
```

This is the method called when this node is used in a pipeline

<a id="entity.EntityExtractor.extract"></a>

#### EntityExtractor.extract

```python
def extract(text)
```

This function can be called to perform entity extraction when using the node in isolation.

<a id="entity.EntityExtractor.extract_batch"></a>

#### EntityExtractor.extract\_batch

```python
def extract_batch(texts: Union[List[str], List[List[str]]],
                  batch_size: Optional[int] = None)
```

This function allows to extract entities out of a list of strings or a list of lists of strings.

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

