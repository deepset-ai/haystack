from typing import Dict, Any, Optional, Callable

from pathlib import Path

from canals.pipeline import unmarshal_pipelines as canals_unmarshal_pipelines

from haystack.preview.pipeline import Pipeline


def load_pipelines(path: Path, _reader: Optional[Callable[..., Any]] = None):
    """
    Loads a dictionary of Pipeline objects from a file. The file can be in any format supported by the function
    passed to `_reader`.

    :param path: the path to the file.
    :param _reader: a function that reads the file and returns a dictionary.

    :returns: a dictionary of Pipeline objects.
    """
    with open(path, "r", encoding="utf-8") as handle:
        schema = _reader(handle)
    return unmarshal_pipelines(schema=schema)


def unmarshal_pipelines(schema: Dict[str, Any]) -> Dict[str, Pipeline]:
    """
    Given a dictionary with a representation of pipelines, returns a dictionary of Pipeline objects.

    This representation includes components, connections, stores and connections to stores.

    The pipelines unmarshalled from the output of this function are equal to the original pipelines.

    :param schema: a dictionary loaded from JSON, YAML or other formats.
    :returns: a dictionary of Pipeline objects.
    """
    pipelines = canals_unmarshal_pipelines(schema=schema)
    for name, pipeline in pipelines.items():
        pipeline._stores = _unmarshal_stores(stores=schema["pipelines"][name]["stores"])


def _unmarshal_stores(stores: Dict[str, Dict[str, Any]]):
    """
    Unmarshals the stores of a pipeline.
    """
    return {store_name: _unmarshal_store(store_data=store_data) for store_name, store_data in stores.items()}


def _unmarshal_store(store_data: Dict[str, Any]):
    """
    Unmarshals a store of a pipeline.
    """
    store_type = store_data["type"]
    store_init_parameters = store_data["init_parameters"]
    return globals()[store_type](**store_init_parameters)
