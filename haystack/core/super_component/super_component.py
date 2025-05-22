# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import functools
from pathlib import Path
from types import new_class
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack import logging
from haystack.core.component.component import component
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.core.pipeline.pipeline import Pipeline
from haystack.core.pipeline.utils import parse_connect_string
from haystack.core.serialization import default_from_dict, default_to_dict, generate_qualified_class_name
from haystack.core.super_component.utils import _delegate_default, _is_compatible

logger = logging.getLogger(__name__)


class InvalidMappingTypeError(Exception):
    """Raised when input or output mappings have invalid types or type conflicts."""

    pass


class InvalidMappingValueError(Exception):
    """Raised when input or output mappings have invalid values or missing components/sockets."""

    pass


@component
class _SuperComponent:
    def __init__(
        self,
        pipeline: Union[Pipeline, AsyncPipeline],
        input_mapping: Optional[Dict[str, List[str]]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Creates a SuperComponent with optional input and output mappings.

        :param pipeline: The pipeline instance or async pipeline instance to be wrapped
        :param input_mapping: A dictionary mapping component input names to pipeline input socket paths.
            If not provided, a default input mapping will be created based on all pipeline inputs.
        :param output_mapping: A dictionary mapping pipeline output socket paths to component output names.
            If not provided, a default output mapping will be created based on all pipeline outputs.
        :raises InvalidMappingError: Raised if any mapping is invalid or type conflicts occur
        :raises ValueError: Raised if no pipeline is provided
        """
        if pipeline is None:
            raise ValueError("Pipeline must be provided to SuperComponent.")

        self.pipeline: Union[Pipeline, AsyncPipeline] = pipeline
        self._warmed_up = False

        # Determine input types based on pipeline and mapping
        pipeline_inputs = self.pipeline.inputs()
        resolved_input_mapping = (
            input_mapping if input_mapping is not None else self._create_input_mapping(pipeline_inputs)
        )
        self._validate_input_mapping(pipeline_inputs, resolved_input_mapping)
        input_types = self._resolve_input_types_from_mapping(pipeline_inputs, resolved_input_mapping)
        # Set input types on the component
        for input_name, info in input_types.items():
            component.set_input_type(self, name=input_name, **info)

        self.input_mapping: Dict[str, List[str]] = resolved_input_mapping
        self._original_input_mapping = input_mapping

        # Set output types based on pipeline and mapping
        leaf_pipeline_outputs = self.pipeline.outputs()
        all_possible_pipeline_outputs = self.pipeline.outputs(include_components_with_connected_outputs=True)

        resolved_output_mapping = (
            output_mapping if output_mapping is not None else self._create_output_mapping(leaf_pipeline_outputs)
        )
        self._validate_output_mapping(all_possible_pipeline_outputs, resolved_output_mapping)
        output_types = self._resolve_output_types_from_mapping(all_possible_pipeline_outputs, resolved_output_mapping)
        # Set output types on the component
        component.set_output_types(self, **output_types)
        self.output_mapping: Dict[str, str] = resolved_output_mapping
        self._original_output_mapping = output_mapping

    def warm_up(self) -> None:
        """
        Warms up the SuperComponent by warming up the wrapped pipeline.
        """
        if not self._warmed_up:
            self.pipeline.warm_up()
            self._warmed_up = True

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Runs the wrapped pipeline with the provided inputs.

        Steps:
        1. Maps the inputs from kwargs to pipeline component inputs
        2. Runs the pipeline
        3. Maps the pipeline outputs to the SuperComponent's outputs

        :param kwargs: Keyword arguments matching the SuperComponent's input names
        :returns:
            Dictionary containing the SuperComponent's output values
        """
        filtered_inputs = {param: value for param, value in kwargs.items() if value != _delegate_default}
        pipeline_inputs = self._map_explicit_inputs(input_mapping=self.input_mapping, inputs=filtered_inputs)
        include_outputs_from = self._get_include_outputs_from()
        pipeline_outputs = self.pipeline.run(data=pipeline_inputs, include_outputs_from=include_outputs_from)
        return self._map_explicit_outputs(pipeline_outputs, self.output_mapping)

    def _get_include_outputs_from(self) -> set[str]:
        # Collecting the component names from output_mapping
        return {self._split_component_path(path)[0] for path in self.output_mapping.keys()}

    async def run_async(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Runs the wrapped pipeline with the provided inputs async.

        Steps:
        1. Maps the inputs from kwargs to pipeline component inputs
        2. Runs the pipeline async
        3. Maps the pipeline outputs to the SuperComponent's outputs

        :param kwargs: Keyword arguments matching the SuperComponent's input names
        :returns:
            Dictionary containing the SuperComponent's output values
        :raises TypeError:
            If the pipeline is not an AsyncPipeline
        """
        if not isinstance(self.pipeline, AsyncPipeline):
            raise TypeError("Pipeline is not an AsyncPipeline. run_async is not supported.")

        filtered_inputs = {param: value for param, value in kwargs.items() if value != _delegate_default}
        pipeline_inputs = self._map_explicit_inputs(input_mapping=self.input_mapping, inputs=filtered_inputs)
        pipeline_outputs = await self.pipeline.run_async(data=pipeline_inputs)
        return self._map_explicit_outputs(pipeline_outputs, self.output_mapping)

    @staticmethod
    def _split_component_path(path: str) -> Tuple[str, str]:
        """
        Splits a component path into a component name and a socket name.

        :param path: A string in the format "component_name.socket_name".
        :returns:
            A tuple containing (component_name, socket_name).
        :raises InvalidMappingValueError:
            If the path format is incorrect.
        """
        comp_name, socket_name = parse_connect_string(path)
        if socket_name is None:
            raise InvalidMappingValueError(f"Invalid path format: '{path}'. Expected 'component_name.socket_name'.")
        return comp_name, socket_name

    def _validate_input_mapping(
        self, pipeline_inputs: Dict[str, Dict[str, Any]], input_mapping: Dict[str, List[str]]
    ) -> None:
        """
        Validates the input mapping to ensure that specified components and sockets exist in the pipeline.

        :param pipeline_inputs: A dictionary containing pipeline input specifications.
        :param input_mapping: A dictionary mapping wrapper input names to pipeline socket paths.
        :raises InvalidMappingTypeError:
            If the input mapping is of invalid type or contains invalid types.
        :raises InvalidMappingValueError:
            If the input mapping contains nonexistent components or sockets.
        """
        if not isinstance(input_mapping, dict):
            raise InvalidMappingTypeError("input_mapping must be a dictionary")

        for wrapper_input_name, pipeline_input_paths in input_mapping.items():
            if not isinstance(pipeline_input_paths, list):
                raise InvalidMappingTypeError(f"Input paths for '{wrapper_input_name}' must be a list of strings.")
            for path in pipeline_input_paths:
                comp_name, socket_name = self._split_component_path(path)
                if comp_name not in pipeline_inputs:
                    raise InvalidMappingValueError(f"Component '{comp_name}' not found in pipeline inputs.")
                if socket_name not in pipeline_inputs[comp_name]:
                    raise InvalidMappingValueError(
                        f"Input socket '{socket_name}' not found in component '{comp_name}'."
                    )

    def _resolve_input_types_from_mapping(
        self, pipeline_inputs: Dict[str, Dict[str, Any]], input_mapping: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Resolves and validates input types based on the provided input mapping.

        This function ensures that all mapped pipeline inputs are compatible, consolidating types
        when multiple mappings exist. It also determines whether an input is mandatory or has a default value.

        :param pipeline_inputs: A dictionary containing pipeline input specifications.
        :param input_mapping: A dictionary mapping SuperComponent inputs to pipeline socket paths.
        :returns:
            A dictionary specifying the resolved input types and their properties.
        :raises InvalidMappingTypeError:
            If the input mapping contains incompatible types.
        """
        aggregated_inputs: Dict[str, Dict[str, Any]] = {}
        for wrapper_input_name, pipeline_input_paths in input_mapping.items():
            for path in pipeline_input_paths:
                comp_name, socket_name = self._split_component_path(path)
                socket_info = pipeline_inputs[comp_name][socket_name]

                # Add to aggregated inputs
                existing_socket_info = aggregated_inputs.get(wrapper_input_name)
                if existing_socket_info is None:
                    aggregated_inputs[wrapper_input_name] = {"type": socket_info["type"]}
                    if not socket_info["is_mandatory"]:
                        aggregated_inputs[wrapper_input_name]["default"] = _delegate_default
                    continue

                is_compatible, common_type = _is_compatible(existing_socket_info["type"], socket_info["type"])

                if not is_compatible:
                    raise InvalidMappingTypeError(
                        f"Type conflict for input '{socket_name}' from component '{comp_name}'. "
                        f"Existing type: {existing_socket_info['type']}, new type: {socket_info['type']}."
                    )

                # Use the common type for the aggregated input
                aggregated_inputs[wrapper_input_name]["type"] = common_type

                # If any socket requires mandatory inputs then the aggregated input is also considered mandatory.
                # So we use the type of the mandatory input and remove the default value if it exists.
                if socket_info["is_mandatory"]:
                    aggregated_inputs[wrapper_input_name].pop("default", None)

        return aggregated_inputs

    @staticmethod
    def _create_input_mapping(pipeline_inputs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Create an input mapping from pipeline inputs.

        :param pipeline_inputs: Dictionary of pipeline input specifications
        :returns:
            Dictionary mapping SuperComponent input names to pipeline socket paths
        """
        input_mapping: Dict[str, List[str]] = {}
        for comp_name, inputs_dict in pipeline_inputs.items():
            for socket_name in inputs_dict.keys():
                existing_socket_info = input_mapping.get(socket_name)
                if existing_socket_info is None:
                    input_mapping[socket_name] = [f"{comp_name}.{socket_name}"]
                    continue
                input_mapping[socket_name].append(f"{comp_name}.{socket_name}")
        return input_mapping

    def _validate_output_mapping(
        self, pipeline_outputs: Dict[str, Dict[str, Any]], output_mapping: Dict[str, str]
    ) -> None:
        """
        Validates the output mapping to ensure that specified components and sockets exist in the pipeline.

        :param pipeline_outputs: A dictionary containing pipeline output specifications.
        :param output_mapping: A dictionary mapping pipeline socket paths to wrapper output names.
        :raises InvalidMappingTypeError:
            If the output mapping is of invalid type or contains invalid types.
        :raises InvalidMappingValueError:
            If the output mapping contains nonexistent components or sockets.
        """
        for pipeline_output_path, wrapper_output_name in output_mapping.items():
            if not isinstance(wrapper_output_name, str):
                raise InvalidMappingTypeError("Output names in output_mapping must be strings.")
            comp_name, socket_name = self._split_component_path(pipeline_output_path)
            if comp_name not in pipeline_outputs:
                raise InvalidMappingValueError(f"Component '{comp_name}' not found among pipeline outputs.")
            if socket_name not in pipeline_outputs[comp_name]:
                raise InvalidMappingValueError(f"Output socket '{socket_name}' not found in component '{comp_name}'.")

    def _resolve_output_types_from_mapping(
        self, pipeline_outputs: Dict[str, Dict[str, Any]], output_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Resolves and validates output types based on the provided output mapping.

        This function ensures that all mapped pipeline outputs are correctly assigned to
        the corresponding SuperComponent outputs while preventing duplicate output names.

        :param pipeline_outputs: A dictionary containing pipeline output specifications.
        :param output_mapping: A dictionary mapping pipeline output socket paths to SuperComponent output names.
        :returns:
            A dictionary mapping SuperComponent output names to their resolved types.
        :raises InvalidMappingValueError:
            If the output mapping contains duplicate output names.
        """
        resolved_outputs = {}
        for pipeline_output_path, wrapper_output_name in output_mapping.items():
            comp_name, socket_name = self._split_component_path(pipeline_output_path)
            if wrapper_output_name in resolved_outputs:
                raise InvalidMappingValueError(f"Duplicate output name '{wrapper_output_name}' in output_mapping.")
            resolved_outputs[wrapper_output_name] = pipeline_outputs[comp_name][socket_name]["type"]
        return resolved_outputs

    @staticmethod
    def _create_output_mapping(pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Create an output mapping from pipeline outputs.

        :param pipeline_outputs: Dictionary of pipeline output specifications
        :returns:
            Dictionary mapping pipeline socket paths to SuperComponent output names
        :raises InvalidMappingValueError:
            If there are output name conflicts between components
        """
        output_mapping = {}
        used_output_names: set[str] = set()
        for comp_name, outputs_dict in pipeline_outputs.items():
            for socket_name in outputs_dict.keys():
                if socket_name in used_output_names:
                    raise InvalidMappingValueError(
                        f"Output name conflict: '{socket_name}' is produced by multiple components. "
                        "Please provide an output_mapping to resolve this conflict."
                    )
                used_output_names.add(socket_name)
                output_mapping[f"{comp_name}.{socket_name}"] = socket_name
        return output_mapping

    def _map_explicit_inputs(
        self, input_mapping: Dict[str, List[str]], inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map inputs according to explicit input mapping.

        :param input_mapping: Mapping configuration for inputs
        :param inputs: Input arguments provided to wrapper
        :return: Dictionary of mapped pipeline inputs
        """
        pipeline_inputs: Dict[str, Dict[str, Any]] = {}
        for wrapper_input_name, pipeline_input_paths in input_mapping.items():
            if wrapper_input_name not in inputs:
                continue

            for socket_path in pipeline_input_paths:
                comp_name, input_name = self._split_component_path(socket_path)
                if comp_name not in pipeline_inputs:
                    pipeline_inputs[comp_name] = {}
                pipeline_inputs[comp_name][input_name] = inputs[wrapper_input_name]

        return pipeline_inputs

    def _map_explicit_outputs(
        self, pipeline_outputs: Dict[str, Dict[str, Any]], output_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Map outputs according to explicit output mapping.

        :param pipeline_outputs: Raw outputs from pipeline execution
        :param output_mapping: Output mapping configuration
        :return: Dictionary of mapped outputs
        """
        outputs: Dict[str, Any] = {}
        for pipeline_output_path, wrapper_output_name in output_mapping.items():
            comp_name, socket_name = self._split_component_path(pipeline_output_path)
            if comp_name in pipeline_outputs and socket_name in pipeline_outputs[comp_name]:
                outputs[wrapper_output_name] = pipeline_outputs[comp_name][socket_name]
        return outputs

    def _to_super_component_dict(self) -> Dict[str, Any]:
        """
        Convert to a SuperComponent dictionary representation.

        :return: Dictionary containing serialized SuperComponent data
        """
        serialized_pipeline = self.pipeline.to_dict()
        serialized = default_to_dict(
            self,
            pipeline=serialized_pipeline,
            input_mapping=self._original_input_mapping,
            output_mapping=self._original_output_mapping,
        )
        serialized["type"] = generate_qualified_class_name(SuperComponent)
        return serialized


@component
class SuperComponent(_SuperComponent):
    """
    A class for creating super components that wrap around a Pipeline.

    This component allows for remapping of input and output socket names between the wrapped pipeline and the
    SuperComponent's input and output names. This is useful for creating higher-level components that abstract
    away the details of the wrapped pipeline.

    ### Usage example

    ```python
    from haystack import Pipeline, SuperComponent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.dataclasses.chat_message import ChatMessage
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.dataclasses import Document

    document_store = InMemoryDocumentStore()
    documents = [
        Document(content="Paris is the capital of France."),
        Document(content="London is the capital of England."),
    ]
    document_store.write_documents(documents)

    prompt_template = [
        ChatMessage.from_user(
        '''
        According to the following documents:
        {% for document in documents %}
        {{document.content}}
        {% endfor %}
        Answer the given question: {{query}}
        Answer:
        '''
        )
    ]

    prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables="*")

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", OpenAIChatGenerator())
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    # Create a super component with simplified input/output mapping
    wrapper = SuperComponent(
        pipeline=pipeline,
        input_mapping={
            "query": ["retriever.query", "prompt_builder.query"],
        },
        output_mapping={"llm.replies": "replies"}
    )

    # Run the pipeline with simplified interface
    result = wrapper.run(query="What is the capital of France?")
    print(result)
    {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>,
     _content=[TextContent(text='The capital of France is Paris.')],...)
    ```

    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the SuperComponent into a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return self._to_super_component_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuperComponent":
        """
        Deserializes the SuperComponent from a dictionary.

        :param data: The dictionary to deserialize from.
        :returns:
            The deserialized SuperComponent.
        """
        pipeline = Pipeline.from_dict(data["init_parameters"]["pipeline"])
        data["init_parameters"]["pipeline"] = pipeline
        return default_from_dict(cls, data)

    def show(self, server_url: str = "https://mermaid.ink", params: Optional[dict] = None, timeout: int = 30) -> None:
        """
        Display an image representing this SuperComponent's underlying pipeline in a Jupyter notebook.

        This function generates a diagram of the Pipeline using a Mermaid server and displays it directly in
        the notebook.

        :param server_url:
            The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
            See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
            info on how to set up your own Mermaid server.

        :param params:
            Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
            Supported keys:
                - format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
                - type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
                - theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
                - bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
                - width: Width of the output image (integer).
                - height: Height of the output image (integer).
                - scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
                - fit: Whether to fit the diagram size to the page (PDF only, boolean).
                - paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
                - landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.

        :param timeout:
            Timeout in seconds for the request to the Mermaid server.

        :raises PipelineDrawingError:
            If the function is called outside of a Jupyter notebook or if there is an issue with rendering.
        """
        self.pipeline.show(server_url=server_url, params=params, timeout=timeout)

    def draw(
        self, path: Path, server_url: str = "https://mermaid.ink", params: Optional[dict] = None, timeout: int = 30
    ) -> None:
        """
        Save an image representing this SuperComponent's underlying pipeline to the specified file path.

        This function generates a diagram of the Pipeline using the Mermaid server and saves it to the provided path.

        :param path:
            The file path where the generated image will be saved.
        :param server_url:
            The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
            See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
            info on how to set up your own Mermaid server.
        :param params:
            Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
            Supported keys:
                - format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
                - type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
                - theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
                - bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
                - width: Width of the output image (integer).
                - height: Height of the output image (integer).
                - scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
                - fit: Whether to fit the diagram size to the page (PDF only, boolean).
                - paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
                - landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.

        :param timeout:
            Timeout in seconds for the request to the Mermaid server.

        :raises PipelineDrawingError:
            If there is an issue with rendering or saving the image.
        """
        self.pipeline.draw(path=path, server_url=server_url, params=params, timeout=timeout)


def super_component(cls: Any):
    """
    Decorator that converts a class into a SuperComponent.

    This decorator:
    1. Creates a new class that inherits from SuperComponent
    2. Copies all methods and attributes from the original class
    3. Adds initialization logic to properly set up the SuperComponent

    The decorated class should define:
    - pipeline: A Pipeline or AsyncPipeline instance in the __init__ method
    - input_mapping: Dictionary mapping component inputs to pipeline inputs (optional)
    - output_mapping: Dictionary mapping pipeline outputs to component outputs (optional)
    """
    logger.debug("Registering {cls} as a super_component", cls=cls)

    # Store the original __init__ method
    original_init = cls.__init__

    # Create a new __init__ method that will initialize both the original class and SuperComponent
    def init_wrapper(self, *args, **kwargs):
        # Call the original __init__ to set up pipeline and mappings
        original_init(self, *args, **kwargs)

        # Verify required attributes
        if not hasattr(self, "pipeline"):
            raise ValueError(f"Class {cls.__name__} decorated with @super_component must define a 'pipeline' attribute")

        # Initialize SuperComponent
        _SuperComponent.__init__(
            self,
            pipeline=self.pipeline,
            input_mapping=getattr(self, "input_mapping", None),
            output_mapping=getattr(self, "output_mapping", None),
        )

    # Preserve original init's signature for IDEs/docs/tools
    init_wrapper = functools.wraps(original_init)(init_wrapper)

    # Function to copy namespace from the original class
    def copy_class_namespace(namespace):
        """Copy all attributes from the original class except special ones."""
        for key, val in dict(cls.__dict__).items():
            # Skip special attributes that should be recreated
            if key in ("__dict__", "__weakref__"):
                continue

            # Override __init__ with our wrapper
            if key == "__init__":
                namespace["__init__"] = init_wrapper
                continue

            namespace[key] = val

    # Create a new class inheriting from SuperComponent with the original methods
    # We use (SuperComponent,) + cls.__bases__ to make the new class inherit from
    # SuperComponent and all the original class's bases
    new_cls = new_class(cls.__name__, (_SuperComponent,) + cls.__bases__, {}, copy_class_namespace)

    # Copy other class attributes
    new_cls.__module__ = cls.__module__
    new_cls.__qualname__ = cls.__qualname__
    new_cls.__doc__ = cls.__doc__

    # Apply the component decorator to the new class
    return component(new_cls)
