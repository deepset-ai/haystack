# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, TextIO, Tuple, Type, TypeVar, Union

import networkx  # type:ignore

from haystack import logging
from haystack.core.component import Component, InputSocket, OutputSocket, component
from haystack.core.errors import (
    DeserializationError,
    PipelineConnectError,
    PipelineDrawingError,
    PipelineError,
    PipelineRuntimeError,
    PipelineUnmarshalError,
    PipelineValidationError,
)
from haystack.core.serialization import DeserializationCallbacks, component_from_dict, component_to_dict
from haystack.core.type_utils import _type_name, _types_are_compatible
from haystack.marshal import Marshaller, YamlMarshaller
from haystack.utils import is_in_jupyter, type_serialization

from .descriptions import find_pipeline_inputs, find_pipeline_outputs
from .draw import _to_mermaid_image
from .template import PipelineTemplate, PredefinedPipeline
from .utils import parse_connect_string

DEFAULT_MARSHALLER = YamlMarshaller()

# We use a generic type to annotate the return value of classmethods,
# so that static analyzers won't be confused when derived classes
# use those methods.
T = TypeVar("T", bound="PipelineBase")

logger = logging.getLogger(__name__)


class PipelineBase:
    """
    Components orchestration engine.

    Builds a graph of components and orchestrates their execution according to the execution graph.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None, max_runs_per_component: int = 100):
        """
        Creates the Pipeline.

        :param metadata:
            Arbitrary dictionary to store metadata about this `Pipeline`. Make sure all the values contained in
            this dictionary can be serialized and deserialized if you wish to save this `Pipeline` to file.
        :param max_runs_per_component:
            How many times the `Pipeline` can run the same Component.
            If this limit is reached a `PipelineMaxComponentRuns` exception is raised.
            If not set defaults to 100 runs per Component.
        """
        self._telemetry_runs = 0
        self._last_telemetry_sent: Optional[datetime] = None
        self.metadata = metadata or {}
        self.graph = networkx.MultiDiGraph()
        self._max_runs_per_component = max_runs_per_component

    def __eq__(self, other) -> bool:
        """
        Pipeline equality is defined by their type and the equality of their serialized form.

        Pipelines of the same type share every metadata, node and edge, but they're not required to use
        the same node instances: this allows pipeline saved and then loaded back to be equal to themselves.
        """
        if not isinstance(self, type(other)):
            return False
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """
        Returns a text representation of the Pipeline.
        """
        res = f"{object.__repr__(self)}\n"
        if self.metadata:
            res += "🧱 Metadata\n"
            for k, v in self.metadata.items():
                res += f"  - {k}: {v}\n"

        res += "🚅 Components\n"
        for name, instance in self.graph.nodes(data="instance"):  # type: ignore # type wrongly defined in networkx
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
            components[name] = component_to_dict(instance, name)

        connections = []
        for sender, receiver, edge_data in self.graph.edges.data():
            sender_socket = edge_data["from_socket"].name
            receiver_socket = edge_data["to_socket"].name
            connections.append({"sender": f"{sender}.{sender_socket}", "receiver": f"{receiver}.{receiver_socket}"})
        return {
            "metadata": self.metadata,
            "max_runs_per_component": self._max_runs_per_component,
            "components": components,
            "connections": connections,
        }

    @classmethod
    def from_dict(
        cls: Type[T], data: Dict[str, Any], callbacks: Optional[DeserializationCallbacks] = None, **kwargs
    ) -> T:
        """
        Deserializes the pipeline from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :param callbacks:
            Callbacks to invoke during deserialization.
        :param kwargs:
            `components`: a dictionary of {name: instance} to reuse instances of components instead of creating new
            ones.
        :returns:
            Deserialized component.
        """
        data_copy = deepcopy(data)  # to prevent modification of original data
        metadata = data_copy.get("metadata", {})
        max_runs_per_component = data_copy.get("max_runs_per_component", 100)
        pipe = cls(metadata=metadata, max_runs_per_component=max_runs_per_component)
        components_to_reuse = kwargs.get("components", {})
        for name, component_data in data_copy.get("components", {}).items():
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
                        logger.debug("Trying to import module {module_name}", module_name=module)
                        type_serialization.thread_safe_import(module)
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

                try:
                    instance = component_from_dict(component_class, component_data, name, callbacks)
                except Exception as e:
                    msg = (
                        f"Couldn't deserialize component '{name}' of class '{component_class.__name__}' "
                        f"with the following data: {str(component_data)}. Possible reasons include "
                        "malformed serialized data, mismatch between the serialized component and the "
                        "loaded one (due to a breaking change, see "
                        "https://github.com/deepset-ai/haystack/releases), etc."
                    )
                    raise DeserializationError(msg) from e
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
        Returns the string representation of this pipeline according to the format dictated by the `Marshaller` in use.

        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :returns:
            A string representing the pipeline.
        """
        return marshaller.marshal(self.to_dict())

    def dump(self, fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER):
        """
        Writes the string representation of this pipeline to the file-like object passed in the `fp` argument.

        :param fp:
            A file-like object ready to be written to.
        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        """
        fp.write(marshaller.marshal(self.to_dict()))

    @classmethod
    def loads(
        cls: Type[T],
        data: Union[str, bytes, bytearray],
        marshaller: Marshaller = DEFAULT_MARSHALLER,
        callbacks: Optional[DeserializationCallbacks] = None,
    ) -> T:
        """
        Creates a `Pipeline` object from the string representation passed in the `data` argument.

        :param data:
            The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :param callbacks:
            Callbacks to invoke during deserialization.
        :raises DeserializationError:
            If an error occurs during deserialization.
        :returns:
            A `Pipeline` object.
        """
        try:
            deserialized_data = marshaller.unmarshal(data)
        except Exception as e:
            raise DeserializationError(
                "Error while unmarshalling serialized pipeline data. This is usually "
                "caused by malformed or invalid syntax in the serialized representation."
            ) from e

        return cls.from_dict(deserialized_data, callbacks)

    @classmethod
    def load(
        cls: Type[T],
        fp: TextIO,
        marshaller: Marshaller = DEFAULT_MARSHALLER,
        callbacks: Optional[DeserializationCallbacks] = None,
    ) -> T:
        """
        Creates a `Pipeline` object a string representation.

        The string representation is read from the file-like object passed in the `fp` argument.


        :param fp:
            A file-like object ready to be read from.
        :param marshaller:
            The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :param callbacks:
            Callbacks to invoke during deserialization.
        :raises DeserializationError:
            If an error occurs during deserialization.
        :returns:
            A `Pipeline` object.
        """
        return cls.loads(fp.read(), marshaller, callbacks)

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
                "Component has already been added in another Pipeline. Components can't be shared between Pipelines. "
                "Create a new instance instead."
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

    def remove_component(self, name: str) -> Component:
        """
        Remove and returns component from the pipeline.

        Remove an existing component from the pipeline by providing its name.
        All edges that connect to the component will also be deleted.

        :param name:
            The name of the component to remove.
        :returns:
            The removed Component instance.

        :raises ValueError:
            If there is no component with that name already in the Pipeline.
        """

        # Check that a component with that name is in the Pipeline
        try:
            instance = self.get_component(name)
        except ValueError as exc:
            raise ValueError(
                f"There is no component named '{name}' in the pipeline. The valid component names are: ",
                ", ".join(n for n in self.graph.nodes),
            ) from exc

        # Delete component from the graph, deleting all its connections
        self.graph.remove_node(name)

        # Reset the Component sockets' senders and receivers
        input_sockets = instance.__haystack_input__._sockets_dict  # type: ignore[attr-defined]
        for socket in input_sockets.values():
            socket.senders = []

        output_sockets = instance.__haystack_output__._sockets_dict  # type: ignore[attr-defined]
        for socket in output_sockets.values():
            socket.receivers = []

        # Reset the Component's pipeline reference
        setattr(instance, "__haystack_added_to_pipeline__", None)

        return instance

    def connect(self, sender: str, receiver: str) -> "PipelineBase":  # noqa: PLR0915
        """
        Connects two components together.

        All components to connect must exist in the pipeline.
        If connecting to a component that has several output connections, specify the inputs and output names as
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

        if sender_component_name == receiver_component_name:
            raise PipelineConnectError("Connecting a Component to itself is not supported.")

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
                    f"Cannot connect '{sender_component_name}.{sender_socket_candidates[0].name}' with "
                    f"'{receiver_component_name}.{receiver_socket_candidates[0].name}': "
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
                    f"Cannot connect '{sender_component_name}' with "
                    f"'{receiver_component_name}': more than one connection is possible "
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
                f"Cannot connect '{sender_component_name}.{sender_socket.name}' with "
                f"'{receiver_component_name}.{receiver_socket.name}': "
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
        for name, inst in self.graph.nodes(data="instance"):  # type: ignore # type wrongly defined in networkx
            if inst == instance:
                return name
        return ""

    def inputs(self, include_components_with_connected_inputs: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the inputs of a pipeline.

        Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
        the input sockets of that component, including their types and whether they are optional.

        :param include_components_with_connected_inputs:
            If `False`, only components that have disconnected input edges are
            included in the output.
        :returns:
            A dictionary where each key is a pipeline component name and each value is a dictionary of
            inputs sockets of that component.
        """
        inputs: Dict[str, Dict[str, Any]] = {}
        for component_name, data in find_pipeline_inputs(self.graph, include_components_with_connected_inputs).items():
            sockets_description = {}
            for socket in data:
                sockets_description[socket.name] = {"type": socket.type, "is_mandatory": socket.is_mandatory}
                if not socket.is_mandatory:
                    sockets_description[socket.name]["default_value"] = socket.default_value

            if sockets_description:
                inputs[component_name] = sockets_description
        return inputs

    def outputs(self, include_components_with_connected_outputs: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the outputs of a pipeline.

        Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
        the output sockets of that component.

        :param include_components_with_connected_outputs:
            If `False`, only components that have disconnected output edges are
            included in the output.
        :returns:
            A dictionary where each key is a pipeline component name and each value is a dictionary of
            output sockets of that component.
        """
        outputs = {
            comp: {socket.name: {"type": socket.type} for socket in data}
            for comp, data in find_pipeline_outputs(self.graph, include_components_with_connected_outputs).items()
            if data
        }
        return outputs

    def show(self) -> None:
        """
        If running in a Jupyter notebook, display an image representing this `Pipeline`.

        """
        if is_in_jupyter():
            from IPython.display import Image, display  # type: ignore

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
        for component_name, instance in self.graph.nodes(data="instance"):  # type: ignore # type is wrong in networkx
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
        Validates pipeline input data.

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

    def _prepare_component_input_data(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Prepares input data for pipeline components.

        Organizes input data for pipeline components and identifies any inputs that are not matched to any
        component's input slots. Deep-copies data items to avoid sharing mutables across multiple components.

        This method processes a flat dictionary of input data, where each key-value pair represents an input name
        and its corresponding value. It distributes these inputs to the appropriate pipeline components based on
        their input requirements. Inputs that don't match any component's input slots are classified as unresolved.

        :param data:
            A dictionary potentially having input names as keys and input values as values.

        :returns:
            A dictionary mapping component names to their respective matched inputs.
        """
        # check whether the data is a nested dictionary of component inputs where each key is a component name
        # and each value is a dictionary of input parameters for that component
        is_nested_component_input = all(isinstance(value, dict) for value in data.values())
        if not is_nested_component_input:
            # flat input, a dict where keys are input names and values are the corresponding values
            # we need to convert it to a nested dictionary of component inputs and then run the pipeline
            # just like in the previous case
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

            if unresolved_kwargs:
                logger.warning(
                    "Inputs {input_keys} were not matched to any component inputs, please check your run parameters.",
                    input_keys=list(unresolved_kwargs.keys()),
                )

            data = dict(pipeline_input_data)

        # deepcopying the inputs prevents the Pipeline run logic from being altered unexpectedly
        # when the same input reference is passed to multiple components.
        for component_name, component_inputs in data.items():
            data[component_name] = {k: deepcopy(v) for k, v in component_inputs.items()}

        return data

    def _normalize_varidiac_input_data(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Variadic inputs expect their value to be a list, this utility method creates that list from the user's input.
        """
        for component_name, component_inputs in data.items():
            if component_name not in self.graph.nodes:
                # This is not a component name, it must be the name of one or more input sockets.
                # Those are handled in a different way, so we skip them here.
                continue
            instance = self.graph.nodes[component_name]["instance"]
            for component_input, input_value in component_inputs.items():
                if instance.__haystack_input__._sockets_dict[component_input].is_variadic:
                    # Components that have variadic inputs need to receive lists as input.
                    # We don't want to force the user to always pass lists, so we convert single values to lists here.
                    # If it's already a list we assume the component takes a variadic input of lists, so we
                    # convert it in any case.
                    data[component_name][component_input] = [input_value]

        return {**data}

    @classmethod
    def from_template(
        cls, predefined_pipeline: PredefinedPipeline, template_params: Optional[Dict[str, Any]] = None
    ) -> "PipelineBase":
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

    def _init_graph(self):
        """Resets the visits count for each component"""
        for node in self.graph.nodes:
            self.graph.nodes[node]["visits"] = 0

    def _find_receivers_from(self, component_name: str) -> List[Tuple[str, OutputSocket, InputSocket]]:
        """
        Utility function to find all Components that receive input form `component_name`.

        :param component_name:
            Name of the sender Component

        :returns:
            List of tuples containing name of the receiver Component and sender OutputSocket
            and receiver InputSocket instances
        """
        res = []
        for _, receiver_name, connection in self.graph.edges(nbunch=component_name, data=True):
            sender_socket: OutputSocket = connection["from_socket"]
            receiver_socket: InputSocket = connection["to_socket"]
            res.append((receiver_name, sender_socket, receiver_socket))
        return res

    def _distribute_output(  # pylint: disable=too-many-positional-arguments
        self,
        receiver_components: List[Tuple[str, OutputSocket, InputSocket]],
        component_result: Dict[str, Any],
        components_inputs: Dict[str, Dict[str, Any]],
        run_queue: List[Tuple[str, Component]],
        waiting_queue: List[Tuple[str, Component]],
    ) -> Dict[str, Any]:
        """
        Distributes the output of a Component to the next Components that need it.

        This also updates the queues that keep track of which Components are ready to run and which are waiting for
        input.

        :param receiver_components:
            List of tuples containing name of receiver Components and relative sender OutputSocket
            and receiver InputSocket instances
        :param component_result:
            The output of the Component
        :param components_inputs:
            The current state of the inputs divided by Component name
        :param run_queue:
            Queue of Components to run
        :param waiting_queue:
            Queue of Components waiting for input

        :returns:
            The updated output of the Component without the keys that were distributed to other Components
        """
        # We keep track of which keys to remove from component_result at the end of the loop.
        # This is done after the output has been distributed to the next components, so that
        # we're sure all components that need this output have received it.
        to_remove_from_component_result = set()

        for receiver_name, sender_socket, receiver_socket in receiver_components:
            if sender_socket.name not in component_result:
                # This output wasn't created by the sender, nothing we can do.
                #
                # Some Components might have conditional outputs, so we need to check if they actually returned
                # some output while iterating over their output sockets.
                #
                # A perfect example of this would be the ConditionalRouter, which will have an output for each
                # condition it has been initialized with.
                # Though it will return only one output at a time.
                continue

            if receiver_name not in components_inputs:
                components_inputs[receiver_name] = {}

            # We keep track of the keys that were distributed to other Components.
            # This key will be removed from component_result at the end of the loop.
            to_remove_from_component_result.add(sender_socket.name)

            value = component_result[sender_socket.name]

            if receiver_socket.is_variadic:
                # Usually Component inputs can only be received from one sender, the Variadic type allows
                # instead to receive inputs from multiple senders.
                #
                # To keep track of all the inputs received internally we always store them in a list.
                if receiver_socket.name not in components_inputs[receiver_name]:
                    # Create the list if it doesn't exist
                    components_inputs[receiver_name][receiver_socket.name] = []
                else:
                    # Check if the value is actually a list
                    assert isinstance(components_inputs[receiver_name][receiver_socket.name], list)
                components_inputs[receiver_name][receiver_socket.name].append(value)
            else:
                components_inputs[receiver_name][receiver_socket.name] = value

            receiver = self.graph.nodes[receiver_name]["instance"]
            pair = (receiver_name, receiver)

            if receiver_socket.is_variadic:
                if receiver_socket.is_greedy:
                    # If the receiver is greedy, we can run it as soon as possible.
                    # First we remove it from the status lists it's in if it's there or
                    # we risk running it multiple times.
                    if pair in run_queue:
                        run_queue.remove(pair)
                    if pair in waiting_queue:
                        waiting_queue.remove(pair)
                    run_queue.insert(0, pair)
                else:
                    # If the receiver Component has a variadic input that is not greedy
                    # we put it in the waiting queue.
                    # This make sure that we don't run it earlier than necessary and we can collect
                    # as many inputs as we can before running it.
                    if pair not in waiting_queue:
                        waiting_queue.append(pair)

            if pair not in waiting_queue and pair not in run_queue:
                # Queue up the Component that received this input to run, only if it's not already waiting
                # for input or already ready to run.
                run_queue.append(pair)

        # Returns the output without the keys that were distributed to other Components
        return {k: v for k, v in component_result.items() if k not in to_remove_from_component_result}

    def _find_next_runnable_component(
        self, components_inputs: Dict[str, Dict[str, Any]], waiting_queue: List[Tuple[str, Component]]
    ) -> Tuple[str, Component]:
        """
        Finds the next Component that can be run and returns it.

        :param components_inputs: The current state of the inputs divided by Component name
        :param waiting_queue: Queue of Components waiting for input

        :returns: The name and the instance of the next Component that can be run
        """
        all_lazy_variadic = True
        all_with_default_inputs = True

        filtered_waiting_queue = []

        for name, comp in waiting_queue:
            if not _is_lazy_variadic(comp):
                # Components with variadic inputs that are not greedy must be removed only if there's nothing else to
                # run at this stage.
                # We need to wait as long as possible to run them, so we can collect as most inputs as we can.
                all_lazy_variadic = False

            if not _has_all_inputs_with_defaults(comp):
                # Components that have defaults for all their inputs must be treated the same identical way as we treat
                # lazy variadic components. If there are only components with defaults we can run them.
                # If we don't do this the order of execution of the Pipeline's Components will be affected cause we
                # enqueue the Components in `run_queue` at the start using the order they are added in the Pipeline.
                # If a Component A with defaults is added before a Component B that has no defaults, but in the Pipeline
                # logic A must be executed after B. However, B could run before A if we don't do this check.
                all_with_default_inputs = False

            if not _is_lazy_variadic(comp) and not _has_all_inputs_with_defaults(comp):
                # Keep track of the Components that are not lazy variadic and don't have all inputs with defaults.
                # We'll handle these later if necessary.
                filtered_waiting_queue.append((name, comp))

        # If all Components are lazy variadic or all Components have all inputs with defaults we can get one to run
        if all_lazy_variadic or all_with_default_inputs:
            return waiting_queue[0]

        for name, comp in filtered_waiting_queue:
            # Find the first component that has all the inputs it needs to run
            has_enough_inputs = True
            for input_socket in comp.__haystack_input__._sockets_dict.values():  # type: ignore
                if input_socket.name not in components_inputs.get(name, {}) and input_socket.is_mandatory:
                    has_enough_inputs = False
                    break

            if has_enough_inputs:
                return name, comp

        # If we reach this point it means that we found no Component that has enough inputs to run.
        # Ideally we should never reach this point, though we can't raise an exception either as
        # existing use cases rely on this behavior.
        # So we return the last Component, that could be the last from waiting_queue or filtered_waiting_queue.
        return name, comp

    def _find_next_runnable_lazy_variadic_or_default_component(
        self, waiting_queue: List[Tuple[str, Component]]
    ) -> Tuple[str, Component]:
        """
        Finds the next Component that can be run and has a lazy variadic input or all inputs with default values.

        :param waiting_queue: Queue of Components waiting for input

        :returns: The name and the instance of the next Component that can be run
        """
        for name, comp in waiting_queue:
            is_lazy_variadic = _is_lazy_variadic(comp)
            has_only_defaults = _has_all_inputs_with_defaults(comp)
            if is_lazy_variadic or has_only_defaults:
                return name, comp

        # If we reach this point it means that we found no Component that has a lazy variadic input or all inputs with
        # default values to run.
        # Similar to `_find_next_runnable_component` we might not find the Component we want, so we optimistically
        # return the last Component in the list.
        # We're probably stuck in a loop in this case, but we can't raise an exception as existing use cases might
        # rely on this behaviour.
        # The loop detection will be handled later on.
        return name, comp

    def _find_components_that_will_receive_no_input(
        self, component_name: str, component_result: Dict[str, Any], components_inputs: Dict[str, Dict[str, Any]]
    ) -> Set[Tuple[str, Component]]:
        """
        Find all the Components that are connected to component_name and didn't receive any input from it.

        Components that have a Variadic input and received already some input from other Components
        but not from component_name won't be returned as they have enough inputs to run.

        This includes the descendants of the Components that didn't receive any input from component_name.
        That is necessary to avoid getting stuck into infinite loops waiting for inputs that will never arrive.

        :param component_name: Name of the Component that created the output
        :param component_result: Output of the Component
        :param components_inputs: The current state of the inputs divided by Component name
        :return: A set of Components that didn't receive any input from component_name
        """

        # Simplifies the check if a Component is Variadic and received some input from other Components.
        def has_variadic_socket_with_existing_inputs(
            component: Component, component_name: str, sender_name: str, components_inputs: Dict[str, Dict[str, Any]]
        ) -> bool:
            for socket in component.__haystack_input__._sockets_dict.values():  # type: ignore
                if sender_name not in socket.senders:
                    continue
                if socket.is_variadic and len(components_inputs.get(component_name, {}).get(socket.name, [])) > 0:
                    return True
            return False

        # Makes it easier to verify if all connections between two Components are optional
        def all_connections_are_optional(sender_name: str, receiver: Component) -> bool:
            for socket in receiver.__haystack_input__._sockets_dict.values():  # type: ignore
                if sender_name not in socket.senders:
                    continue
                if socket.is_mandatory:
                    return False
            return True

        # Eases checking if other connections that are not between sender_name and receiver_name
        # already received inputs
        def other_connections_received_input(sender_name: str, receiver_name: str) -> bool:
            receiver: Component = self.graph.nodes[receiver_name]["instance"]
            for receiver_socket in receiver.__haystack_input__._sockets_dict.values():  # type: ignore
                if sender_name in receiver_socket.senders:
                    continue
                if components_inputs.get(receiver_name, {}).get(receiver_socket.name) is not None:
                    return True
            return False

        components = set()
        instance: Component = self.graph.nodes[component_name]["instance"]
        for socket_name, socket in instance.__haystack_output__._sockets_dict.items():  # type: ignore
            if socket_name in component_result:
                continue
            for receiver in socket.receivers:
                receiver_instance: Component = self.graph.nodes[receiver]["instance"]

                if has_variadic_socket_with_existing_inputs(
                    receiver_instance, receiver, component_name, components_inputs
                ):
                    # Components with Variadic input that already received some input
                    # can still run, even if branch is skipped.
                    # If we remove them they won't run.
                    continue

                if all_connections_are_optional(component_name, receiver_instance) and other_connections_received_input(
                    component_name, receiver
                ):
                    # If all the connections between component_name and receiver are optional
                    # and receiver received other inputs already it still has enough inputs to run.
                    # Even if it didn't receive input from component_name, so we can't remove it or its
                    # descendants.
                    continue

                components.add((receiver, receiver_instance))
                # Get the descendants too. When we remove a Component that received no input
                # it's extremely likely that its descendants will receive no input as well.
                # This is fine even if the Pipeline will merge back into a single Component
                # at a certain point. The merging Component will be put back into the run
                # queue at a later stage.
                for descendant_name in networkx.descendants(self.graph, receiver):
                    descendant = self.graph.nodes[descendant_name]["instance"]

                    # Components with Variadic input that already received some input
                    # can still run, even if branch is skipped.
                    # If we remove them they won't run.
                    if has_variadic_socket_with_existing_inputs(
                        descendant, descendant_name, receiver, components_inputs
                    ):
                        continue

                    components.add((descendant_name, descendant))

        return components

    def _is_stuck_in_a_loop(self, waiting_queue: List[Tuple[str, Component]]) -> bool:
        """
        Checks if the Pipeline is stuck in a loop.

        :param waiting_queue: Queue of Components waiting for input

        :returns: True if the Pipeline is stuck in a loop, False otherwise
        """
        # Are we actually stuck or there's a lazy variadic or a component with has only default inputs
        # waiting for input?
        # This is our last resort, if there's no lazy variadic or component with only default inputs
        # waiting for input we're stuck for real and we can't make any progress.
        component_found = False
        for _, comp in waiting_queue:
            if _is_lazy_variadic(comp) or _has_all_inputs_with_defaults(comp):
                component_found = True
                break

        if not component_found:
            # We're stuck in a loop for real, we can't make any progress.
            # BAIL!
            return True

        # If we have a single component with no variadic input or only default inputs waiting for input
        # it means it has been waiting for input for at least 2 iterations.
        # This will never run.
        # BAIL!
        return len(waiting_queue) == 1

    def _component_has_enough_inputs_to_run(self, name: str, inputs: Dict[str, Dict[str, Any]]) -> bool:
        """
        Returns True if the Component has all the inputs it needs to run.

        :param name: Name of the Component as defined in the Pipeline.
        :param inputs: The current state of the inputs divided by Component name.

        :return: Whether the Component can run or not.
        """
        instance: Component = self.graph.nodes[name]["instance"]
        if name not in inputs:
            return False
        expected_inputs = instance.__haystack_input__._sockets_dict.keys()  # type: ignore
        current_inputs = inputs[name].keys()
        return expected_inputs == current_inputs

    def _break_supported_cycles_in_graph(self) -> Tuple[networkx.MultiDiGraph, Dict[str, List[List[str]]]]:
        """
        Utility function to remove supported cycles in the Pipeline's graph.

        Given that the Pipeline execution would wait to run a Component until it has received
        all its mandatory inputs, it doesn't make sense for us to try and break cycles by
        removing a connection to a mandatory input. The Pipeline would just get stuck at a later time.

        So we can only break connections in cycles that have a Variadic or GreedyVariadic type or a default value.

        This will raise a PipelineRuntimeError if we there are cycles that can't be broken.
        That is bound to happen when at least one of the inputs in a cycle is mandatory.

        If the Pipeline's graph doesn't have any cycle it will just return that graph and an empty dictionary.

        :returns:
            A tuple containing:
                * A copy of the Pipeline's graph without cycles
                * A dictionary of Component's names and a list of all the cycles they were part of.
                  The cycles are a list of Component's names that create that cycle.
        """
        if networkx.is_directed_acyclic_graph(self.graph):
            return self.graph, {}

        temp_graph: networkx.MultiDiGraph = self.graph.copy()
        # A list of all the cycles that are found in the graph, each inner list contains
        # the Component names that create that cycle.
        cycles: List[List[str]] = list(networkx.simple_cycles(self.graph))
        # Maps a Component name to a list of its output socket names that have been broken
        edges_removed: Dict[str, List[str]] = defaultdict(list)
        # This keeps track of all the cycles that a component is part of.
        # Maps a Component name to a list of cycles, each inner list contains
        # the Component names that create that cycle (the key will also be
        # an element in each list). The last Component in each list is implicitly
        # connected to the first.
        components_in_cycles: Dict[str, List[List[str]]] = defaultdict(list)

        # Used to minimize the number of time we check whether the graph has any more
        # cycles left to break or not.
        graph_has_cycles = True

        # Iterate all the cycles to find the least amount of connections that we can remove
        # to make the Pipeline graph acyclic.
        # As soon as the graph is acyclic we stop breaking connections and return.
        for cycle in cycles:
            for comp in cycle:
                components_in_cycles[comp].append(cycle)

            # Iterate this cycle, we zip the cycle with itself so that at the last iteration
            # sender_comp will be the last element of cycle and receiver_comp will be the first.
            # So if cycle is [1, 2, 3, 4] we would call zip([1, 2, 3, 4], [2, 3, 4, 1]).
            for sender_comp, receiver_comp in zip(cycle, cycle[1:] + cycle[:1]):
                # We get the key and iterate those as we want to edit the graph data while
                # iterating the edges and that would raise.
                # Even though the connection key set in Pipeline.connect() uses only the
                # sockets name we don't have clashes since it's only used to differentiate
                # multiple edges between two nodes.
                edge_keys = list(temp_graph.get_edge_data(sender_comp, receiver_comp).keys())
                for edge_key in edge_keys:
                    edge_data = temp_graph.get_edge_data(sender_comp, receiver_comp)[edge_key]
                    receiver_socket = edge_data["to_socket"]
                    if not receiver_socket.is_variadic and receiver_socket.is_mandatory:
                        continue

                    # We found a breakable edge
                    sender_socket = edge_data["from_socket"]
                    edges_removed[sender_comp].append(sender_socket.name)
                    temp_graph.remove_edge(sender_comp, receiver_comp, edge_key)

                    graph_has_cycles = not networkx.is_directed_acyclic_graph(temp_graph)
                    if not graph_has_cycles:
                        # We removed all the cycles, we can stop
                        break

            if not graph_has_cycles:
                # We removed all the cycles, nice
                break

        if graph_has_cycles:
            msg = "Pipeline contains a cycle that we can't execute"
            raise PipelineRuntimeError(msg)

        return temp_graph, components_in_cycles


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


def _is_lazy_variadic(c: Component) -> bool:
    """
    Small utility function to check if a Component has at least a Variadic input and no GreedyVariadic input.
    """
    is_variadic = any(
        socket.is_variadic
        for socket in c.__haystack_input__._sockets_dict.values()  # type: ignore
    )
    if not is_variadic:
        return False
    return not any(
        socket.is_greedy
        for socket in c.__haystack_input__._sockets_dict.values()  # type: ignore
    )


def _has_all_inputs_with_defaults(c: Component) -> bool:
    """
    Small utility function to check if a Component has all inputs with defaults.
    """
    return all(
        not socket.is_mandatory
        for socket in c.__haystack_input__._sockets_dict.values()  # type: ignore
    )


def _add_missing_input_defaults(name: str, comp: Component, components_inputs: Dict[str, Dict[str, Any]]):
    """
    Updates the inputs with the default values for the inputs that are missing

    :param name: Name of the Component
    :param comp: Instance of the Component
    :param components_inputs: The current state of the inputs divided by Component name
    """
    if name not in components_inputs:
        components_inputs[name] = {}

    for input_socket in comp.__haystack_input__._sockets_dict.values():  # type: ignore
        if input_socket.is_mandatory:
            continue

        if input_socket.name not in components_inputs[name]:
            components_inputs[name][input_socket.name] = input_socket.default_value


def _enqueue_component(
    component_pair: Tuple[str, Component],
    run_queue: List[Tuple[str, Component]],
    waiting_queue: List[Tuple[str, Component]],
):
    """
    Append a Component in the queue of Components to run if not already in it.

    Remove it from the waiting list if it's there.

    :param component_pair: Tuple of Component name and instance
    :param run_queue: Queue of Components to run
    :param waiting_queue: Queue of Components waiting for input
    """
    if component_pair in waiting_queue:
        waiting_queue.remove(component_pair)

    if component_pair not in run_queue:
        run_queue.append(component_pair)


def _dequeue_component(
    component_pair: Tuple[str, Component],
    run_queue: List[Tuple[str, Component]],
    waiting_queue: List[Tuple[str, Component]],
):
    """
    Removes a Component both from the queue of Components to run and the waiting list.

    :param component_pair: Tuple of Component name and instance
    :param run_queue: Queue of Components to run
    :param waiting_queue: Queue of Components waiting for input
    """
    if component_pair in waiting_queue:
        waiting_queue.remove(component_pair)

    if component_pair in run_queue:
        run_queue.remove(component_pair)


def _enqueue_waiting_component(component_pair: Tuple[str, Component], waiting_queue: List[Tuple[str, Component]]):
    """
    Append a Component in the queue of Components that are waiting for inputs if not already in it.

    :param component_pair: Tuple of Component name and instance
    :param waiting_queue: Queue of Components waiting for input
    """
    if component_pair not in waiting_queue:
        waiting_queue.append(component_pair)


def _dequeue_waiting_component(component_pair: Tuple[str, Component], waiting_queue: List[Tuple[str, Component]]):
    """
    Removes a Component from the queue of Components that are waiting for inputs.

    :param component_pair: Tuple of Component name and instance
    :param waiting_queue: Queue of Components waiting for input
    """
    if component_pair in waiting_queue:
        waiting_queue.remove(component_pair)
