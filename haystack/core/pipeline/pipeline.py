# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import importlib
import itertools
from collections import defaultdict
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, TextIO, Tuple, Type, TypeVar, Union

import networkx  # type:ignore

from haystack import logging, tracing
from haystack.core.component import Component, InputSocket, OutputSocket, component
from haystack.core.errors import (
    PipelineConnectError,
    PipelineDrawingError,
    PipelineError,
    PipelineMaxLoops,
    PipelineRuntimeError,
    PipelineUnmarshalError,
    PipelineValidationError,
)
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.core.type_utils import _type_name, _types_are_compatible
from haystack.marshal import Marshaller, YamlMarshaller
from haystack.telemetry import pipeline_running
from haystack.utils import is_in_jupyter

from .descriptions import find_pipeline_inputs, find_pipeline_outputs
from .draw import _to_mermaid_image
from .template import PipelineTemplate, PredefinedPipeline

DEFAULT_MARSHALLER = YamlMarshaller()
logger = logging.getLogger(__name__)

# We use a generic type to annotate the return value of classmethods,
# so that static analyzers won't be confused when derived classes
# use those methods.
T = TypeVar("T", bound="Pipeline")


class Pipeline:
    """
    Components orchestration engine.

    Builds a graph of components and orchestrates their execution according to the execution graph.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        max_loops_allowed: int = 100,
        debug_path: Union[Path, str] = Path(".haystack_debug/"),
    ):
        """
        Creates the Pipeline.

        :param metadata:
            Arbitrary dictionary to store metadata about this pipeline. Make sure all the values contained in
            this dictionary can be serialized and deserialized if you wish to save this pipeline to file with
            `save_pipelines()/load_pipelines()`.
        :param max_loops_allowed:
            How many times the pipeline can run the same node before throwing an exception.
        :param debug_path:
            When debug is enabled in `run()`, where to save the debug data.
        """
        self._telemetry_runs = 0
        self._last_telemetry_sent: Optional[datetime] = None
        self.metadata = metadata or {}
        self.max_loops_allowed = max_loops_allowed
        self.graph = networkx.MultiDiGraph()
        self._debug: Dict[int, Dict[str, Any]] = {}
        self._debug_path = Path(debug_path)

    def __eq__(self, other) -> bool:
        """
        Equal pipelines share every metadata, node and edge, but they're not required to use
        the same node instances: this allows pipeline saved and then loaded back to be equal to themselves.
        """
        if not isinstance(other, Pipeline):
            return False
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """
        Returns a text representation of the Pipeline.
        If this runs in a Jupyter notebook, it will instead display the Pipeline image.
        """
        if is_in_jupyter():
            # If we're in a Jupyter notebook we want to display the image instead of the text repr.
            self.show()
            return ""

        res = f"{object.__repr__(self)}\n"
        if self.metadata:
            res += "🧱 Metadata\n"
            for k, v in self.metadata.items():
                res += f"  - {k}: {v}\n"

        res += "🚅 Components\n"
        for name, instance in self.graph.nodes(data="instance"):
            res += f"  - {name}: {instance.__class__.__name__}\n"

        res += "🛤️ Connections\n"
        for sender, receiver, edge_data in self.graph.edges(data=True):
            sender_socket = edge_data["from_socket"].name
            receiver_socket = edge_data["to_socket"].name
            res += f"  - {sender}.{sender_socket} -> {receiver}.{receiver_socket} ({edge_data['conn_type']})\n"

        return res

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the pipeline to a dictionary.
        This is meant to be an intermediate representation but it can be also used to save a pipeline to file.

        :returns:
            Dictionary with serialized data.
        """
        components = {}
        for name, instance in self.graph.nodes(data="instance"):  # type:ignore
            components[name] = component_to_dict(instance)

        connections = []
        for sender, receiver, edge_data in self.graph.edges.data():
            sender_socket = edge_data["from_socket"].name
            receiver_socket = edge_data["to_socket"].name
            connections.append({"sender": f"{sender}.{sender_socket}", "receiver": f"{receiver}.{receiver_socket}"})
        return {
            "metadata": self.metadata,
            "max_loops_allowed": self.max_loops_allowed,
            "components": components,
            "connections": connections,
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any], **kwargs) -> T:
        """
        Deserializes the pipeline from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :param kwargs:
            `components`: a dictionary of {name: instance} to reuse instances of components instead of creating new ones.
        :returns:
            Deserialized component.
        """
        metadata = data.get("metadata", {})
        max_loops_allowed = data.get("max_loops_allowed", 100)
        debug_path = Path(data.get("debug_path", ".haystack_debug/"))
        pipe = cls(metadata=metadata, max_loops_allowed=max_loops_allowed, debug_path=debug_path)
        components_to_reuse = kwargs.get("components", {})
        for name, component_data in data.get("components", {}).items():
            if name in components_to_reuse:
                # Reuse an instance
                instance = components_to_reuse[name]
            else:
                if "type" not in component_data:
                    raise PipelineError(f"Missing 'type' in component '{name}'")

                if component_data["type"] not in component.registry:
                    try:
                        # Import the module first...
                        module, _ = component_data["type"].rsplit(".", 1)
                        logger.debug("Trying to import {module}", module=module)
                        importlib.import_module(module)
                        # ...then try again
                        if component_data["type"] not in component.registry:
                            raise PipelineError(
                                f"Successfully imported module {module} but can't find it in the component registry."
                                "This is unexpected and most likely a bug."
                            )
                    except (ImportError, PipelineError) as e:
                        raise PipelineError(f"Component '{component_data['type']}' not imported.") from e

                # Create a new one
                component_class = component.registry[component_data["type"]]
                instance = component_from_dict(component_class, component_data)
            pipe.add_component(name=name, instance=instance)

        for connection in data.get("connections", []):
            if "sender" not in connection:
                raise PipelineError(f"Missing sender in connection: {connection}")
            if "receiver" not in connection:
                raise PipelineError(f"Missing receiver in connection: {connection}")
            pipe.connect(sender=connection["sender"], receiver=connection["receiver"])

        return pipe

    def dumps(self, marshaller: Marshaller = DEFAULT_MARSHALLER) -> str:
        """
        Returns the string representation of this pipeline according to the
        format dictated by the `Marshaller` in use.

        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :returns:
            A string representing the pipeline.
        """
        return marshaller.marshal(self.to_dict())

    def dump(self, fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER):
        """
        Writes the string representation of this pipeline to the file-like object
        passed in the `fp` argument.

        :param fp:
            A file-like object ready to be written to.
        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        """
        fp.write(marshaller.marshal(self.to_dict()))

    @classmethod
    def loads(cls, data: Union[str, bytes, bytearray], marshaller: Marshaller = DEFAULT_MARSHALLER) -> "Pipeline":
        """
        Creates a `Pipeline` object from the string representation passed in the `data` argument.

        :param data:
            The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :returns:
            A `Pipeline` object.
        """
        return cls.from_dict(marshaller.unmarshal(data))

    @classmethod
    def load(cls, fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> "Pipeline":
        """
        Creates a `Pipeline` object from the string representation read from the file-like
        object passed in the `fp` argument.

        :param data:
            The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
        :param fp:
            A file-like object ready to be read from.
        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :returns:
            A `Pipeline` object.
        """
        return cls.from_dict(marshaller.unmarshal(fp.read()))

    def add_component(self, name: str, instance: Component) -> None:
        """
        Add the given component to the pipeline.

        Components are not connected to anything by default: use `Pipeline.connect()` to connect components together.
        Component names must be unique, but component instances can be reused if needed.

        :param name:
            The name of the component to add.
        :param instance:
            The component instance to add.

        :raises ValueError:
            If a component with the same name already exists.
        :raises PipelineValidationError:
            If the given instance is not a Canals component.
        """
        # Component names are unique
        if name in self.graph.nodes:
            raise ValueError(f"A component named '{name}' already exists in this pipeline: choose another name.")

        # Components can't be named `_debug`
        if name == "_debug":
            raise ValueError("'_debug' is a reserved name for debug output. Choose another name.")

        # Component instances must be components
        if not isinstance(instance, Component):
            raise PipelineValidationError(
                f"'{type(instance)}' doesn't seem to be a component. Is this class decorated with @component?"
            )

        if getattr(instance, "__haystack_added_to_pipeline__", None):
            msg = (
                "Component has already been added in another Pipeline. "
                "Components can't be shared between Pipelines. Create a new instance instead."
            )
            raise PipelineError(msg)

        setattr(instance, "__haystack_added_to_pipeline__", self)

        # Add component to the graph, disconnected
        logger.debug("Adding component '{component_name}' ({component})", component_name=name, component=instance)
        # We're completely sure the fields exist so we ignore the type error
        self.graph.add_node(
            name,
            instance=instance,
            input_sockets=instance.__haystack_input__._sockets_dict,  # type: ignore[attr-defined]
            output_sockets=instance.__haystack_output__._sockets_dict,  # type: ignore[attr-defined]
            visits=0,
        )

    def connect(self, sender: str, receiver: str) -> "Pipeline":
        """
        Connects two components together.

        All components to connect must exist in the pipeline.
        If connecting to an component that has several output connections, specify the inputs and output names as
        'component_name.connections_name'.

        :param sender:
            The component that delivers the value. This can be either just a component name or can be
            in the format `component_name.connection_name` if the component has multiple outputs.
        :param receiver:
            The component that receives the value. This can be either just a component name or can be
            in the format `component_name.connection_name` if the component has multiple inputs.
        :returns:
            The Pipeline instance.

        :raises PipelineConnectError:
            If the two components cannot be connected (for example if one of the components is
            not present in the pipeline, or the connections don't match by type, and so on).
        """
        # Edges may be named explicitly by passing 'node_name.edge_name' to connect().
        sender_component_name, sender_socket_name = parse_connect_string(sender)
        receiver_component_name, receiver_socket_name = parse_connect_string(receiver)

        # Get the nodes data.
        try:
            from_sockets = self.graph.nodes[sender_component_name]["output_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {sender_component_name} not found in the pipeline.") from exc
        try:
            to_sockets = self.graph.nodes[receiver_component_name]["input_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {receiver_component_name} not found in the pipeline.") from exc

        # If the name of either socket is given, get the socket
        sender_socket: Optional[OutputSocket] = None
        if sender_socket_name:
            sender_socket = from_sockets.get(sender_socket_name)
            if not sender_socket:
                raise PipelineConnectError(
                    f"'{sender} does not exist. "
                    f"Output connections of {sender_component_name} are: "
                    + ", ".join([f"{name} (type {_type_name(socket.type)})" for name, socket in from_sockets.items()])
                )

        receiver_socket: Optional[InputSocket] = None
        if receiver_socket_name:
            receiver_socket = to_sockets.get(receiver_socket_name)
            if not receiver_socket:
                raise PipelineConnectError(
                    f"'{receiver} does not exist. "
                    f"Input connections of {receiver_component_name} are: "
                    + ", ".join([f"{name} (type {_type_name(socket.type)})" for name, socket in to_sockets.items()])
                )

        # Look for a matching connection among the possible ones.
        # Note that if there is more than one possible connection but two sockets match by name, they're paired.
        sender_socket_candidates: List[OutputSocket] = [sender_socket] if sender_socket else list(from_sockets.values())
        receiver_socket_candidates: List[InputSocket] = (
            [receiver_socket] if receiver_socket else list(to_sockets.values())
        )

        # Find all possible connections between these two components
        possible_connections = [
            (sender_sock, receiver_sock)
            for sender_sock, receiver_sock in itertools.product(sender_socket_candidates, receiver_socket_candidates)
            if _types_are_compatible(sender_sock.type, receiver_sock.type)
        ]

        # We need this status for error messages, since we might need it in multiple places we calculate it here
        status = _connections_status(
            sender_node=sender_component_name,
            sender_sockets=sender_socket_candidates,
            receiver_node=receiver_component_name,
            receiver_sockets=receiver_socket_candidates,
        )

        if not possible_connections:
            # There's no possible connection between these two components
            if len(sender_socket_candidates) == len(receiver_socket_candidates) == 1:
                msg = (
                    f"Cannot connect '{sender_component_name}.{sender_socket_candidates[0].name}' with '{receiver_component_name}.{receiver_socket_candidates[0].name}': "
                    f"their declared input and output types do not match.\n{status}"
                )
            else:
                msg = (
                    f"Cannot connect '{sender_component_name}' with '{receiver_component_name}': "
                    f"no matching connections available.\n{status}"
                )
            raise PipelineConnectError(msg)

        if len(possible_connections) == 1:
            # There's only one possible connection, use it
            sender_socket = possible_connections[0][0]
            receiver_socket = possible_connections[0][1]

        if len(possible_connections) > 1:
            # There are multiple possible connection, let's try to match them by name
            name_matches = [
                (out_sock, in_sock) for out_sock, in_sock in possible_connections if in_sock.name == out_sock.name
            ]
            if len(name_matches) != 1:
                # There's are either no matches or more than one, we can't pick one reliably
                msg = (
                    f"Cannot connect '{sender_component_name}' with '{receiver_component_name}': more than one connection is possible "
                    "between these components. Please specify the connection name, like: "
                    f"pipeline.connect('{sender_component_name}.{possible_connections[0][0].name}', "
                    f"'{receiver_component_name}.{possible_connections[0][1].name}').\n{status}"
                )
                raise PipelineConnectError(msg)

            # Get the only possible match
            sender_socket = name_matches[0][0]
            receiver_socket = name_matches[0][1]

        # Connection must be valid on both sender/receiver sides
        if not sender_socket or not receiver_socket or not sender_component_name or not receiver_component_name:
            if sender_component_name and sender_socket:
                sender_repr = f"{sender_component_name}.{sender_socket.name} ({_type_name(sender_socket.type)})"
            else:
                sender_repr = "input needed"

            if receiver_component_name and receiver_socket:
                receiver_repr = f"({_type_name(receiver_socket.type)}) {receiver_component_name}.{receiver_socket.name}"
            else:
                receiver_repr = "output"
            msg = f"Connection must have both sender and receiver: {sender_repr} -> {receiver_repr}"
            raise PipelineConnectError(msg)

        logger.debug(
            "Connecting '{sender_component}.{sender_socket_name}' to '{receiver_component}.{receiver_socket_name}'",
            sender_component=sender_component_name,
            sender_socket_name=sender_socket.name,
            receiver_component=receiver_component_name,
            receiver_socket_name=receiver_socket.name,
        )

        if receiver_component_name in sender_socket.receivers and sender_component_name in receiver_socket.senders:
            # This is already connected, nothing to do
            return self

        if receiver_socket.senders and not receiver_socket.is_variadic:
            # Only variadic input sockets can receive from multiple senders
            msg = (
                f"Cannot connect '{sender_component_name}.{sender_socket.name}' with '{receiver_component_name}.{receiver_socket.name}': "
                f"{receiver_component_name}.{receiver_socket.name} is already connected to {receiver_socket.senders}.\n"
            )
            raise PipelineConnectError(msg)

        # Update the sockets with the new connection
        sender_socket.receivers.append(receiver_component_name)
        receiver_socket.senders.append(sender_component_name)

        # Create the new connection
        self.graph.add_edge(
            sender_component_name,
            receiver_component_name,
            key=f"{sender_socket.name}/{receiver_socket.name}",
            conn_type=_type_name(sender_socket.type),
            from_socket=sender_socket,
            to_socket=receiver_socket,
            mandatory=receiver_socket.is_mandatory,
        )
        return self

    def get_component(self, name: str) -> Component:
        """
        Get the component with the specified name from the pipeline.

        :param name:
            The name of the component.
        :returns:
            The instance of that component.

        :raises ValueError:
            If a component with that name is not present in the pipeline.
        """
        try:
            return self.graph.nodes[name]["instance"]
        except KeyError as exc:
            raise ValueError(f"Component named {name} not found in the pipeline.") from exc

    def get_component_name(self, instance: Component) -> str:
        """
        Returns the name of the Component instance if it has been added to this Pipeline or an empty string otherwise.

        :param instance:
            The Component instance to look for.
        :returns:
            The name of the Component instance.
        """
        for name, inst in self.graph.nodes(data="instance"):
            if inst == instance:
                return name
        return ""

    def inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the inputs of a pipeline. Each key in the dictionary
        corresponds to a component name, and its value is another dictionary that describes the
        input sockets of that component, including their types and whether they are optional.

        :returns:
            A dictionary where each key is a pipeline component name and each value is a dictionary of
            inputs sockets of that component.
        """
        inputs: Dict[str, Dict[str, Any]] = {}
        for component_name, data in find_pipeline_inputs(self.graph).items():
            sockets_description = {}
            for socket in data:
                sockets_description[socket.name] = {"type": socket.type, "is_mandatory": socket.is_mandatory}
                if not socket.is_mandatory:
                    sockets_description[socket.name]["default_value"] = socket.default_value

            if sockets_description:
                inputs[component_name] = sockets_description
        return inputs

    def outputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the outputs of a pipeline. Each key in the dictionary
        corresponds to a component name, and its value is another dictionary that describes the
        output sockets of that component.

        :returns:
            A dictionary where each key is a pipeline component name and each value is a dictionary of
            output sockets of that component.
        """
        outputs = {
            comp: {socket.name: {"type": socket.type} for socket in data}
            for comp, data in find_pipeline_outputs(self.graph).items()
            if data
        }
        return outputs

    def show(self) -> None:
        """
        If running in a Jupyter notebook, display an image representing this `Pipeline`.

        """
        if is_in_jupyter():
            from IPython.display import Image, display

            image_data = _to_mermaid_image(self.graph)

            display(Image(image_data))
        else:
            msg = "This method is only supported in Jupyter notebooks. Use Pipeline.draw() to save an image locally."
            raise PipelineDrawingError(msg)

    def draw(self, path: Path) -> None:
        """
        Save an image representing this `Pipeline` to `path`.

        :param path:
            The path to save the image to.
        """
        # Before drawing we edit a bit the graph, to avoid modifying the original that is
        # used for running the pipeline we copy it.
        image_data = _to_mermaid_image(self.graph)
        Path(path).write_bytes(image_data)

    def walk(self) -> Iterator[Tuple[str, Component]]:
        """
        Visits each component in the pipeline exactly once and yields its name and instance.
        No guarantees are provided on the visiting order.

        :returns:
            An iterator of tuples of component name and component instance.
        """
        for component_name, instance in self.graph.nodes(data="instance"):
            yield component_name, instance

    def warm_up(self):
        """
        Make sure all nodes are warm.

        It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
        without re-initializing everything.
        """
        for node in self.graph.nodes:
            if hasattr(self.graph.nodes[node]["instance"], "warm_up"):
                logger.info("Warming up component {node}...", node=node)
                self.graph.nodes[node]["instance"].warm_up()

    def _validate_input(self, data: Dict[str, Any]):
        """
        Validates that data:
        * Each Component name actually exists in the Pipeline
        * Each Component is not missing any input
        * Each Component has only one input per input socket, if not variadic
        * Each Component doesn't receive inputs that are already sent by another Component

        :param data:
            A dictionary of inputs for the pipeline's components. Each key is a component name.

        :raises ValueError:
            If inputs are invalid according to the above.
        """
        for component_name, component_inputs in data.items():
            if component_name not in self.graph.nodes:
                raise ValueError(f"Component named {component_name} not found in the pipeline.")
            instance = self.graph.nodes[component_name]["instance"]
            for socket_name, socket in instance.__haystack_input__._sockets_dict.items():
                if socket.senders == [] and socket.is_mandatory and socket_name not in component_inputs:
                    raise ValueError(f"Missing input for component {component_name}: {socket_name}")
            for input_name in component_inputs.keys():
                if input_name not in instance.__haystack_input__._sockets_dict:
                    raise ValueError(f"Input {input_name} not found in component {component_name}.")

        for component_name in self.graph.nodes:
            instance = self.graph.nodes[component_name]["instance"]
            for socket_name, socket in instance.__haystack_input__._sockets_dict.items():
                component_inputs = data.get(component_name, {})
                if socket.senders == [] and socket.is_mandatory and socket_name not in component_inputs:
                    raise ValueError(f"Missing input for component {component_name}: {socket_name}")
                if socket.senders and socket_name in component_inputs and not socket.is_variadic:
                    raise ValueError(
                        f"Input {socket_name} for component {component_name} is already sent by {socket.senders}."
                    )

    # TODO: We're ignoring these linting rules for the time being, after we properly optimize this function we'll remove the noqa
    def run(  # noqa: C901, PLR0912, PLR0915 pylint: disable=too-many-branches
        self, data: Dict[str, Any], debug: bool = False
    ) -> Dict[str, Any]:
        """
        Runs the pipeline with given input data.

        :param data:
            A dictionary of inputs for the pipeline's components. Each key is a component name
            and its value is a dictionary of that component's input parameters.
        :param debug:
            Set to True to collect and return debug information.
        :returns:
            A dictionary containing the pipeline's output.

        :raises PipelineRuntimeError:
            If a component fails or returns unexpected output.

        Example a - Using named components:
        Consider a 'Hello' component that takes a 'word' input and outputs a greeting.

        ```python
        @component
        class Hello:
            @component.output_types(output=str)
            def run(self, word: str):
                return {"output": f"Hello, {word}!"}
        ```

        Create a pipeline with two 'Hello' components connected together:

        ```python
        pipeline = Pipeline()
        pipeline.add_component("hello", Hello())
        pipeline.add_component("hello2", Hello())
        pipeline.connect("hello.output", "hello2.word")
        result = pipeline.run(data={"hello": {"word": "world"}})
        ```

        This runs the pipeline with the specified input for 'hello', yielding
        {'hello2': {'output': 'Hello, Hello, world!!'}}.

        Example b - Using flat inputs:
        You can also pass inputs directly without specifying component names:

        ```python
        result = pipeline.run(data={"word": "world"})
        ```

        The pipeline resolves inputs to the correct components, returning
        {'hello2': {'output': 'Hello, Hello, world!!'}}.
        """
        pipeline_running(self)
        # NOTE: We're assuming data is formatted like so as of now
        # data = {
        #     "comp1": {"input1": 1, "input2": 2},
        # }
        #
        # TODO: Support also this format:
        # data = {
        #     "input1": 1, "input2": 2,
        # }

        # Reset the visits count for each component
        for node in self.graph.nodes:
            self.graph.nodes[node]["visits"] = 0

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        # check whether the data is a nested dictionary of component inputs where each key is a component name
        # and each value is a dictionary of input parameters for that component
        is_nested_component_input = all(isinstance(value, dict) for value in data.values())
        if not is_nested_component_input:
            # flat input, a dict where keys are input names and values are the corresponding values
            # we need to convert it to a nested dictionary of component inputs and then run the pipeline
            # just like in the previous case
            data, unresolved_inputs = self._prepare_component_input_data(data)
            if unresolved_inputs:
                logger.warning(
                    "Inputs {input_keys} were not matched to any component inputs, please check your run parameters.",
                    input_keys=list(unresolved_inputs.keys()),
                )

        # Raise if input is malformed in some way
        self._validate_input(data)
        # NOTE: The above NOTE and TODO are technically not true.
        # This implementation of run supports only the first format, but the second format is actually
        # never received by this method. It's handled by the `run()` method of the `Pipeline` class
        # defined in `haystack/pipeline.py`.
        # As of now we're ok with this, but we'll need to merge those two classes at some point.

        # deepcopying the inputs prevents the Pipeline run logic from being altered unexpectedly
        # when the same input reference is passed to multiple components.
        for component_name, component_inputs in data.items():
            data[component_name] = {k: deepcopy(v) for k, v in component_inputs.items()}

        for component_name, component_inputs in data.items():
            if component_name not in self.graph.nodes:
                # This is not a component name, it must be the name of one or more input sockets.
                # Those are handled in a different way, so we skip them here.
                continue
            instance = self.graph.nodes[component_name]["instance"]
            for component_input, input_value in component_inputs.items():
                # Handle mutable input data
                data[component_name][component_input] = copy(input_value)
                if instance.__haystack_input__._sockets_dict[component_input].is_variadic:
                    # Components that have variadic inputs need to receive lists as input.
                    # We don't want to force the user to always pass lists, so we convert single values to lists here.
                    # If it's already a list we assume the component takes a variadic input of lists, so we
                    # convert it in any case.
                    data[component_name][component_input] = [input_value]

        last_inputs: Dict[str, Dict[str, Any]] = {**data}

        # Take all components that have at least 1 input not connected or is variadic,
        # and all components that have no inputs at all
        to_run: List[Tuple[str, Component]] = []
        for node_name in self.graph.nodes:
            component = self.graph.nodes[node_name]["instance"]

            if len(component.__haystack_input__._sockets_dict) == 0:
                # Component has no input, can run right away
                to_run.append((node_name, component))
                continue

            for socket in component.__haystack_input__._sockets_dict.values():
                if not socket.senders or socket.is_variadic:
                    # Component has at least one input not connected or is variadic, can run right away.
                    to_run.append((node_name, component))
                    break

        # These variables are used to detect when we're stuck in a loop.
        # Stuck loops can happen when one or more components are waiting for input but
        # no other component is going to run.
        # This can happen when a whole branch of the graph is skipped for example.
        # When we find that two consecutive iterations of the loop where the waiting_for_input list is the same,
        # we know we're stuck in a loop and we can't make any progress.
        before_last_waiting_for_input: Optional[Set[str]] = None
        last_waiting_for_input: Optional[Set[str]] = None

        # The waiting_for_input list is used to keep track of components that are waiting for input.
        waiting_for_input: List[Tuple[str, Component]] = []

        with tracing.tracer.trace(
            "haystack.pipeline.run",
            tags={
                "haystack.pipeline.debug": debug,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_loops_allowed": self.max_loops_allowed,
            },
        ):
            # This is what we'll return at the end
            final_outputs = {}
            while len(to_run) > 0:
                name, comp = to_run.pop(0)

                if any(socket.is_variadic for socket in comp.__haystack_input__._sockets_dict.values()) and not getattr(  # type: ignore
                    comp, "is_greedy", False
                ):
                    there_are_non_variadics = False
                    for _, other_comp in to_run:
                        if not any(socket.is_variadic for socket in other_comp.__haystack_input__._sockets_dict.values()):  # type: ignore
                            there_are_non_variadics = True
                            break

                    if there_are_non_variadics:
                        if (name, comp) not in waiting_for_input:
                            waiting_for_input.append((name, comp))
                        continue

                if name in last_inputs and len(comp.__haystack_input__._sockets_dict) == len(last_inputs[name]):  # type: ignore
                    if self.graph.nodes[name]["visits"] > self.max_loops_allowed:
                        msg = f"Maximum loops count ({self.max_loops_allowed}) exceeded for component '{name}'"
                        raise PipelineMaxLoops(msg)
                    # This component has all the inputs it needs to run
                    with tracing.tracer.trace(
                        "haystack.component.run",
                        tags={
                            "haystack.component.name": name,
                            "haystack.component.type": comp.__class__.__name__,
                            "haystack.component.input_types": {
                                k: type(v).__name__ for k, v in last_inputs[name].items()
                            },
                            "haystack.component.input_spec": {
                                key: {
                                    "type": value.type.__name__ if isinstance(value.type, type) else str(value.type),
                                    "senders": value.senders,
                                }
                                for key, value in comp.__haystack_input__._sockets_dict.items()  # type: ignore
                            },
                            "haystack.component.output_spec": {
                                key: {
                                    "type": value.type.__name__ if isinstance(value.type, type) else str(value.type),
                                    "senders": value.receivers,
                                }
                                for key, value in comp.__haystack_output__._sockets_dict.items()  # type: ignore
                            },
                        },
                    ) as span:
                        span.set_content_tag("haystack.component.input", last_inputs[name])

                        res = comp.run(**last_inputs[name])
                        self.graph.nodes[name]["visits"] += 1

                        if not isinstance(res, Mapping):
                            raise PipelineRuntimeError(
                                f"Component '{name}' didn't return a dictionary. "
                                "Components must always return dictionaries: check the the documentation."
                            )

                        span.set_tags(tags={"haystack.component.visits": self.graph.nodes[name]["visits"]})
                        span.set_content_tag("haystack.component.output", res)

                    # Reset the waiting for input previous states, we managed to run a component
                    before_last_waiting_for_input = None
                    last_waiting_for_input = None

                    if (name, comp) in waiting_for_input:
                        # We manage to run this component that was in the waiting list, we can remove it.
                        # This happens when a component was put in the waiting list but we reached it from another edge.
                        waiting_for_input.remove((name, comp))

                    # We keep track of which keys to remove from res at the end of the loop.
                    # This is done after the output has been distributed to the next components, so that
                    # we're sure all components that need this output have received it.
                    to_remove_from_res = set()
                    for sender_component_name, receiver_component_name, edge_data in self.graph.edges(data=True):
                        if receiver_component_name == name and edge_data["to_socket"].is_variadic:
                            # Delete variadic inputs that were already consumed
                            last_inputs[name][edge_data["to_socket"].name] = []

                        if name != sender_component_name:
                            continue

                        if edge_data["from_socket"].name not in res:
                            # This output has not been produced by the component, skip it
                            continue

                        if receiver_component_name not in last_inputs:
                            last_inputs[receiver_component_name] = {}
                        to_remove_from_res.add(edge_data["from_socket"].name)
                        value = res[edge_data["from_socket"].name]

                        if edge_data["to_socket"].is_variadic:
                            if edge_data["to_socket"].name not in last_inputs[receiver_component_name]:
                                last_inputs[receiver_component_name][edge_data["to_socket"].name] = []
                            # Add to the list of variadic inputs
                            last_inputs[receiver_component_name][edge_data["to_socket"].name].append(value)
                        else:
                            last_inputs[receiver_component_name][edge_data["to_socket"].name] = value

                        pair = (receiver_component_name, self.graph.nodes[receiver_component_name]["instance"])
                        is_greedy = pair[1].__haystack_is_greedy__
                        is_variadic = edge_data["to_socket"].is_variadic
                        if is_variadic and is_greedy:
                            # If the receiver is greedy, we can run it right away.
                            # First we remove it from the lists it's in if it's there or we risk running it multiple times.
                            if pair in to_run:
                                to_run.remove(pair)
                            if pair in waiting_for_input:
                                waiting_for_input.remove(pair)
                            to_run.append(pair)

                        if pair not in waiting_for_input and pair not in to_run:
                            to_run.append(pair)

                    res = {k: v for k, v in res.items() if k not in to_remove_from_res}

                    if len(res) > 0:
                        final_outputs[name] = res
                else:
                    # This component doesn't have enough inputs so we can't run it yet
                    if (name, comp) not in waiting_for_input:
                        waiting_for_input.append((name, comp))

                if len(to_run) == 0 and len(waiting_for_input) > 0:
                    # Check if we're stuck in a loop.
                    # It's important to check whether previous waitings are None as it could be that no
                    # Component has actually been run yet.
                    if (
                        before_last_waiting_for_input is not None
                        and last_waiting_for_input is not None
                        and before_last_waiting_for_input == last_waiting_for_input
                    ):
                        # Are we actually stuck or there's a lazy variadic waiting for input?
                        # This is our last resort, if there's no lazy variadic waiting for input
                        # we're stuck for real and we can't make any progress.
                        for name, comp in waiting_for_input:
                            is_variadic = any(socket.is_variadic for socket in comp.__haystack_input__._sockets_dict.values())  # type: ignore
                            if is_variadic and not comp.__haystack_is_greedy__:  # type: ignore[attr-defined]
                                break
                        else:
                            # We're stuck in a loop for real, we can't make any progress.
                            # BAIL!
                            break

                        if len(waiting_for_input) == 1:
                            # We have a single component with variadic input waiting for input.
                            # If we're at this point it means it has been waiting for input for at least 2 iterations.
                            # This will never run.
                            # BAIL!
                            break

                        # There was a lazy variadic waiting for input, we can run it
                        waiting_for_input.remove((name, comp))
                        to_run.append((name, comp))
                        continue

                    before_last_waiting_for_input = (
                        last_waiting_for_input.copy() if last_waiting_for_input is not None else None
                    )
                    last_waiting_for_input = {item[0] for item in waiting_for_input}

                    # Remove from waiting only if there is actually enough input to run
                    for name, comp in waiting_for_input:
                        if name not in last_inputs:
                            last_inputs[name] = {}

                        # Lazy variadics must be removed only if there's nothing else to run at this stage
                        is_variadic = any(socket.is_variadic for socket in comp.__haystack_input__._sockets_dict.values())  # type: ignore
                        if is_variadic and not comp.__haystack_is_greedy__:  # type: ignore[attr-defined]
                            there_are_only_lazy_variadics = True
                            for other_name, other_comp in waiting_for_input:
                                if name == other_name:
                                    continue
                                there_are_only_lazy_variadics &= (
                                    any(
                                        socket.is_variadic for socket in other_comp.__haystack_input__._sockets_dict.values()  # type: ignore
                                    )
                                    and not other_comp.__haystack_is_greedy__  # type: ignore[attr-defined]
                                )

                            if not there_are_only_lazy_variadics:
                                continue

                        # Find the first component that has all the inputs it needs to run
                        has_enough_inputs = True
                        for input_socket in comp.__haystack_input__._sockets_dict.values():  # type: ignore
                            if input_socket.is_mandatory and input_socket.name not in last_inputs[name]:
                                has_enough_inputs = False
                                break
                            if input_socket.is_mandatory:
                                continue

                            if input_socket.name not in last_inputs[name]:
                                last_inputs[name][input_socket.name] = input_socket.default_value
                        if has_enough_inputs:
                            break

                    waiting_for_input.remove((name, comp))
                    to_run.append((name, comp))

            return final_outputs

    def _prepare_component_input_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Organizes input data for pipeline components and identifies any inputs that are not matched to any
        component's input slots.

        This method processes a flat dictionary of input data, where each key-value pair represents an input name
        and its corresponding value. It distributes these inputs to the appropriate pipeline components based on
        their input requirements. Inputs that don't match any component's input slots are classified as unresolved.

        :param data:
            A dictionary with input names as keys and input values as values.
        :returns:
            A tuple containing two elements:
             1. A dictionary mapping component names to their respective matched inputs.
             2. A dictionary of inputs that were not matched to any component, termed as unresolved keyword arguments.
        """
        pipeline_input_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        unresolved_kwargs = {}

        # Retrieve the input slots for each component in the pipeline
        available_inputs: Dict[str, Dict[str, Any]] = self.inputs()

        # Go through all provided to distribute them to the appropriate component inputs
        for input_name, input_value in data.items():
            resolved_at_least_once = False

            # Check each component to see if it has a slot for the current kwarg
            for component_name, component_inputs in available_inputs.items():
                if input_name in component_inputs:
                    # If a match is found, add the kwarg to the component's input data
                    pipeline_input_data[component_name][input_name] = input_value
                    resolved_at_least_once = True

            if not resolved_at_least_once:
                unresolved_kwargs[input_name] = input_value

        return pipeline_input_data, unresolved_kwargs

    @classmethod
    def from_template(
        cls, predefined_pipeline: PredefinedPipeline, template_params: Optional[Dict[str, Any]] = None
    ) -> "Pipeline":
        """
        Create a Pipeline from a predefined template. See `PredefinedPipeline` for available options.

        :param predefined_pipeline:
            The predefined pipeline to use.
        :param template_params:
            An optional dictionary of parameters to use when rendering the pipeline template.
        :returns:
            An instance of `Pipeline`.
        """
        tpl = PipelineTemplate.from_predefined(predefined_pipeline)
        # If tpl.render() fails, we let bubble up the original error
        rendered = tpl.render(template_params)

        # If there was a problem with the rendered version of the
        # template, we add it to the error stack for debugging
        try:
            return cls.loads(rendered)
        except Exception as e:
            msg = f"Error unmarshalling pipeline: {e}\n"
            msg += f"Source:\n{rendered}"
            raise PipelineUnmarshalError(msg)


def _connections_status(
    sender_node: str, receiver_node: str, sender_sockets: List[OutputSocket], receiver_sockets: List[InputSocket]
):
    """
    Lists the status of the sockets, for error messages.
    """
    sender_sockets_entries = []
    for sender_socket in sender_sockets:
        sender_sockets_entries.append(f" - {sender_socket.name}: {_type_name(sender_socket.type)}")
    sender_sockets_list = "\n".join(sender_sockets_entries)

    receiver_sockets_entries = []
    for receiver_socket in receiver_sockets:
        if receiver_socket.senders:
            sender_status = f"sent by {','.join(receiver_socket.senders)}"
        else:
            sender_status = "available"
        receiver_sockets_entries.append(
            f" - {receiver_socket.name}: {_type_name(receiver_socket.type)} ({sender_status})"
        )
    receiver_sockets_list = "\n".join(receiver_sockets_entries)

    return f"'{sender_node}':\n{sender_sockets_list}\n'{receiver_node}':\n{receiver_sockets_list}"


def parse_connect_string(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string.

    :param connection:
        The connection string.
    :returns:
        A tuple containing the component name and the connection name.
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None
