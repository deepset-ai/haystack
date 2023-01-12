<a id="file_type"></a>

# Module file\_type

<a id="file_type.FileTypeClassifier"></a>

## FileTypeClassifier

```python
class FileTypeClassifier(BaseComponent)
```

Route files in an Indexing Pipeline to corresponding file converters.

<a id="file_type.FileTypeClassifier.__init__"></a>

#### FileTypeClassifier.\_\_init\_\_

```python
def __init__(supported_types: List[str] = DEFAULT_TYPES)
```

Node that sends out files on a different output edge depending on their extension.

**Arguments**:

- `supported_types`: the file types that this node can distinguish.
Note that it's limited to a maximum of 10 outgoing edges, which
correspond each to a file extension. Such extension are, by default
`txt`, `pdf`, `md`, `docx`, `html`. Lists containing more than 10
elements will not be allowed. Lists with duplicate elements will
also be rejected.

<a id="file_type.FileTypeClassifier.run"></a>

#### FileTypeClassifier.run

```python
def run(file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]])
```

Sends out files on a different output edge depending on their extension.

**Arguments**:

- `file_paths`: paths to route on different edges.
