<a id="file_type"></a>

# Module file\_type

<a id="file_type.FileTypeClassifier"></a>

## FileTypeClassifier

```python
class FileTypeClassifier(BaseComponent)
```

Route files in an Indexing Pipeline to corresponding file converters.

<a id="file_type.FileTypeClassifier.run"></a>

#### run

```python
def run(file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]])
```

Sends out files on a different output edge depending on their extension.

**Arguments**:

- `file_paths`: paths to route on different edges.

