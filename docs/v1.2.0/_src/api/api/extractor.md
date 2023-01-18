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

<a id="entity.EntityExtractor.run"></a>

#### run

```python
def run(documents: Optional[Union[List[Document], List[dict]]] = None) -> Tuple[Dict, str]
```

This is the method called when this node is used in a pipeline

<a id="entity.EntityExtractor.extract"></a>

#### extract

```python
def extract(text)
```

This function can be called to perform entity extraction when using the node in isolation.

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
