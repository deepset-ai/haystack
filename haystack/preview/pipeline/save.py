from typing import Dict, Any, Optional, Callable

from pathlib import Path

from canals.pipeline import marshal_pipelines as canals_marshal_pipelines

from haystack.preview.pipeline import Pipeline


def save_pipelines(pipelines: Dict[str, Pipeline], path: Path, _writer: Optional[Callable[..., Any]] = None):
    """
    Saves a dictionary of Pipeline objects to a file. The file will be in the format supported by the function
    passed to `_writer`.

    :param pipelines: a dictionary of Pipeline objects.
    :param path: the path to the file.
    :param _writer: a function that writes the file.
    """
    schema = marshal_pipelines(pipelines=pipelines)
    with open(path, "w", encoding="utf-8") as handle:
        _writer(schema, handle)


def marshal_pipelines(pipelines: Dict[str, Pipeline]) -> Dict[str, Any]:
    """
    Given a dictionary of Pipeline objects, returns a dictionary with a representation of those pipelines
    that can be serialized to JSON.

    This representation includes components, connections, stores and connections to stores.

    The pipelines unmarshalled from the output of this function are equal to the original pipelines.

    :param pipelines: a dictionary of Pipeline objects.
    :returns: a dictionary that can be written out to JSON, YAML or other formats.
    """
    marshalled = canals_marshal_pipelines(pipelines=pipelines)
    for pipeline_name, pipeline in pipelines.items():
        marshalled["pipelines"][pipeline_name]["stores"] = _marshal_stores(pipeline)
        marshalled["pipelines"][pipeline_name]["components"] = _marshal_store_connections(
            pipeline=pipeline, components=marshalled["pipelines"][pipeline_name]["components"]
        )
    return marshalled


def _marshal_stores(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Marshals the stores of a pipeline.
    """
    return {
        store_name: {"type": store.__class__.__name__, "init_parameters": store.init_parameters}
        for store_name, store in pipeline._stores.items()
    }


def _marshal_store_connections(pipeline: Pipeline, components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Marshals the connection between stores and components.
    """
    marshalled_components = {}
    for component_name, component_data in components.items():
        if component_name in pipeline._store_connections:
            component_data["store"] = pipeline._store_connections[component_name]
            marshalled_components[component_name] = component_data
    return marshalled_components
