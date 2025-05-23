# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import defaultdict
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, TextIO, Tuple, Type, TypeVar, Union

import networkx  # type:ignore

from haystack import logging, tracing
from haystack.core.component import Component, InputSocket, OutputSocket, component
from haystack.core.errors import (
    DeserializationError,
    PipelineComponentsBlockedError,
    PipelineConnectError,
    PipelineDrawingError,
    PipelineError,
    PipelineMaxComponentRuns,
    PipelineUnmarshalError,
    PipelineValidationError,
)
from haystack.core.pipeline.component_checks import (
    _NO_OUTPUT_PRODUCED,
    all_predecessors_executed,
    are_all_lazy_variadic_sockets_resolved,
    are_all_sockets_ready,
    can_component_run,
    is_any_greedy_socket_ready,
    is_socket_lazy_variadic,
)
from haystack.core.pipeline.utils import (
    FIFOPriorityQueue,
    _deepcopy_with_exceptions,
    args_deprecated,
    parse_connect_string,
)
from haystack.core.serialization import DeserializationCallbacks, component_from_dict, component_to_dict
from haystack.core.type_utils import _type_name, _types_are_compatible
from haystack.marshal import Marshaller, YamlMarshaller
from haystack.utils import is_in_jupyter, type_serialization

from .descriptions import find_pipeline_inputs, find_pipeline_outputs
from .draw import _to_mermaid_image
from .template import PipelineTemplate, PredefinedPipeline

DEFAULT_MARSHALLER = YamlMarshaller()

# We use a generic type to annotate the return value of class methods,
# so that static analyzers won't be confused when derived classes
# use those methods.
T = TypeVar("T", bound="PipelineBase")

logger = logging.getLogger(__name__)


# Constants for tracing tags
_COMPONENT_INPUT = "haystack.component.input"
_COMPONENT_OUTPUT = "haystack.component.output"
_COMPONENT_VISITS = "haystack.component.visits"


class ComponentPriority(IntEnum):
    HIGHEST = 1
    READY = 2
    DEFER = 3
    DEFER_LAST = 4
    BLOCKED = 5


class PipelineBase:
    """
    Components orchestration engine.

    Builds a graph of components and orchestrates their execution according to the execution graph.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        max_runs_per_component: int = 100,
        connection_type_validation: bool = True,
    ):
        """
        Creates the Pipeline.

        :param metadata:
            Arbitrary dictionary to store metadata about this `Pipeline`. Make sure all the values contained in
            this dictionary can be serialized and deserialized if you wish to save this `Pipeline` to file.
        :param max_runs_per_component:
            How many times the `Pipeline` can run the same Component.
            If this limit is reached a `PipelineMaxComponentRuns` exception is raised.
            If not set defaults to 100 runs per Component.
        :param connection_type_validation: Whether the pipeline will validate the types of the connections.
            Defaults to True.
        """
        self._telemetry_runs = 0
        self._last_telemetry_sent: Optional[datetime] = None
        self.metadata = metadata or {}
        self.graph = networkx.MultiDiGraph()
        self._max_runs_per_component = max_runs_per_component
        self._connection_type_validation = connection_type_validation

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
            "connection_type_validation": self._connection_type_validation,
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
        data_copy = _deepcopy_with_exceptions(data)  # to prevent modification of original data
        metadata = data_copy.get("metadata", {})
        max_runs_per_component = data_copy.get("max_runs_per_component", 100)
        connection_type_validation = data_copy.get("connection_type_validation", True)
        pipe = cls(
            metadata=metadata,
            max_runs_per_component=max_runs_per_component,
            connection_type_validation=connection_type_validation,
        )
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
                                f"Successfully imported module '{module}' but couldn't find "
                                f"'{component_data['type']}' in the component registry.\n"
                                f"The component might be registered under a different path. "
                                f"Here are the registered components:\n {list(component.registry.keys())}\n"
                            )
                    except (ImportError, PipelineError, ValueError) as e:
                        raise PipelineError(
                            f"Component '{component_data['type']}' (name: '{name}') not imported. Please "
                            f"check that the package is installed and the component path is correct."
                        ) from e

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
            If the given instance is not a component.
        """
        # Component names are unique
        if name in self.graph.nodes:
            raise ValueError(f"A component named '{name}' already exists in this pipeline: choose another name.")

        # Components can't be named `_debug`
        if name == "_debug":
            raise ValueError("'_debug' is a reserved name for debug output. Choose another name.")

        # Component names can't have "."
        if "." in name:
            raise ValueError(f"{name} is an invalid component name, cannot contain '.' (dot) characters.")

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

    def connect(self, sender: str, receiver: str) -> "PipelineBase":  # noqa: PLR0915 PLR0912
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
            sender_sockets = self.graph.nodes[sender_component_name]["output_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {sender_component_name} not found in the pipeline.") from exc
        try:
            receiver_sockets = self.graph.nodes[receiver_component_name]["input_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {receiver_component_name} not found in the pipeline.") from exc

        # If the name of either socket is given, get the socket
        sender_socket: Optional[OutputSocket] = None
        if sender_socket_name:
            sender_socket = sender_sockets.get(sender_socket_name)
            if not sender_socket:
                raise PipelineConnectError(
                    f"'{sender} does not exist. "
                    f"Output connections of {sender_component_name} are: "
                    + ", ".join([f"{name} (type {_type_name(socket.type)})" for name, socket in sender_sockets.items()])
                )

        receiver_socket: Optional[InputSocket] = None
        if receiver_socket_name:
            receiver_socket = receiver_sockets.get(receiver_socket_name)
            if not receiver_socket:
                raise PipelineConnectError(
                    f"'{receiver} does not exist. "
                    f"Input connections of {receiver_component_name} are: "
                    + ", ".join(
                        [f"{name} (type {_type_name(socket.type)})" for name, socket in receiver_sockets.items()]
                    )
                )

        # Look for a matching connection among the possible ones.
        # Note that if there is more than one possible connection but two sockets match by name, they're paired.
        sender_socket_candidates: List[OutputSocket] = (
            [sender_socket] if sender_socket else list(sender_sockets.values())
        )
        receiver_socket_candidates: List[InputSocket] = (
            [receiver_socket] if receiver_socket else list(receiver_sockets.values())
        )

        # Find all possible connections between these two components
        possible_connections = []
        for sender_sock, receiver_sock in itertools.product(sender_socket_candidates, receiver_socket_candidates):
            if _types_are_compatible(sender_sock.type, receiver_sock.type, self._connection_type_validation):
                possible_connections.append((sender_sock, receiver_sock))

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

    @args_deprecated
    def show(
        self,
        server_url: str = "https://mermaid.ink",
        params: Optional[dict] = None,
        timeout: int = 30,
        super_component_expansion: bool = False,
    ) -> None:
        """
        Display an image representing this `Pipeline` in a Jupyter notebook.

        This function generates a diagram of the `Pipeline` using a Mermaid server and displays it directly in
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

        :param super_component_expansion:
            If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
            super-components as if they were components part of the pipeline instead of a "black-box".
            Otherwise, only the super-component itself will be displayed.

        :raises PipelineDrawingError:
            If the function is called outside of a Jupyter notebook or if there is an issue with rendering.
        """

        # Call the internal implementation with keyword arguments
        self._show_internal(
            server_url=server_url, params=params, timeout=timeout, super_component_expansion=super_component_expansion
        )

    def _show_internal(
        self,
        *,
        server_url: str = "https://mermaid.ink",
        params: Optional[dict] = None,
        timeout: int = 30,
        super_component_expansion: bool = False,
    ) -> None:
        """
        Internal implementation of show() that uses keyword-only arguments.

        ToDo: after 2.14.0 release make this the main function and remove the old one.
        """
        if is_in_jupyter():
            from IPython.display import Image, display  # type: ignore

            if super_component_expansion:
                graph, super_component_mapping = self._merge_super_component_pipelines()
            else:
                graph = self.graph
                super_component_mapping = None

            image_data = _to_mermaid_image(
                graph,
                server_url=server_url,
                params=params,
                timeout=timeout,
                super_component_mapping=super_component_mapping,
            )
            display(Image(image_data))
        else:
            msg = "This method is only supported in Jupyter notebooks. Use Pipeline.draw() to save an image locally."
            raise PipelineDrawingError(msg)

    @args_deprecated
    def draw(  # pylint: disable=too-many-positional-arguments
        self,
        path: Path,
        server_url: str = "https://mermaid.ink",
        params: Optional[dict] = None,
        timeout: int = 30,
        super_component_expansion: bool = False,
    ) -> None:
        """
        Save an image representing this `Pipeline` to the specified file path.

        This function generates a diagram of the `Pipeline` using the Mermaid server and saves it to the provided path.

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

        :param super_component_expansion:
            If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
            super-components as if they were components part of the pipeline instead of a "black-box".
            Otherwise, only the super-component itself will be displayed.

        :raises PipelineDrawingError:
            If there is an issue with rendering or saving the image.
        """

        # Call the internal implementation with keyword arguments
        self._draw_internal(
            path=path,
            server_url=server_url,
            params=params,
            timeout=timeout,
            super_component_expansion=super_component_expansion,
        )

    def _draw_internal(
        self,
        *,
        path: Path,
        server_url: str = "https://mermaid.ink",
        params: Optional[dict] = None,
        timeout: int = 30,
        super_component_expansion: bool = False,
    ) -> None:
        """
        Internal implementation of draw() that uses keyword-only arguments.

        ToDo: after 2.14.0 release make this the main function and remove the old one.
        """
        # Before drawing we edit a bit the graph, to avoid modifying the original that is
        # used for running the pipeline we copy it.
        if super_component_expansion:
            graph, super_component_mapping = self._merge_super_component_pipelines()
        else:
            graph = self.graph
            super_component_mapping = None

        image_data = _to_mermaid_image(
            graph,
            server_url=server_url,
            params=params,
            timeout=timeout,
            super_component_mapping=super_component_mapping,
        )
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

    @staticmethod
    def _create_component_span(
        component_name: str, instance: Component, inputs: Dict[str, Any], parent_span: Optional[tracing.Span] = None
    ):
        return tracing.tracer.trace(
            "haystack.component.run",
            tags={
                "haystack.component.name": component_name,
                "haystack.component.type": instance.__class__.__name__,
                "haystack.component.input_types": {k: type(v).__name__ for k, v in inputs.items()},
                "haystack.component.input_spec": {
                    key: {
                        "type": (value.type.__name__ if isinstance(value.type, type) else str(value.type)),
                        "senders": value.senders,
                    }
                    for key, value in instance.__haystack_input__._sockets_dict.items()  # type: ignore
                },
                "haystack.component.output_spec": {
                    key: {
                        "type": (value.type.__name__ if isinstance(value.type, type) else str(value.type)),
                        "receivers": value.receivers,
                    }
                    for key, value in instance.__haystack_output__._sockets_dict.items()  # type: ignore
                },
            },
            parent_span=parent_span,
        )

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
            data[component_name] = {k: _deepcopy_with_exceptions(v) for k, v in component_inputs.items()}

        return data

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

    def _find_receivers_from(self, component_name: str) -> List[Tuple[str, OutputSocket, InputSocket]]:
        """
        Utility function to find all Components that receive input from `component_name`.

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

    @staticmethod
    def _convert_to_internal_format(pipeline_inputs: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
        """
        Converts the inputs to the pipeline to the format that is needed for the internal `Pipeline.run` logic.

        Example Input:
        {'prompt_builder': {'question': 'Who lives in Paris?'}, 'retriever': {'query': 'Who lives in Paris?'}}
        Example Output:
        {'prompt_builder': {'question': [{'sender': None, 'value': 'Who lives in Paris?'}]},
         'retriever': {'query': [{'sender': None, 'value': 'Who lives in Paris?'}]}}

        :param pipeline_inputs: Inputs to the pipeline.
        :returns: Converted inputs that can be used by the internal `Pipeline.run` logic.
        """
        inputs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for component_name, socket_dict in pipeline_inputs.items():
            inputs[component_name] = {}
            for socket_name, value in socket_dict.items():
                inputs[component_name][socket_name] = [{"sender": None, "value": value}]

        return inputs

    @staticmethod
    def _consume_component_inputs(component_name: str, component: Dict, inputs: Dict) -> Dict[str, Any]:
        """
        Extracts the inputs needed to run for the component and removes them from the global inputs state.

        :param component_name: The name of a component.
        :param component: Component with component metadata.
        :param inputs: Global inputs state.
        :returns: The inputs for the component.
        """
        component_inputs = inputs.get(component_name, {})
        consumed_inputs = {}
        greedy_inputs_to_remove = set()
        for socket_name, socket in component["input_sockets"].items():
            socket_inputs = component_inputs.get(socket_name, [])
            socket_inputs = [sock["value"] for sock in socket_inputs if sock["value"] is not _NO_OUTPUT_PRODUCED]
            if socket_inputs:
                if not socket.is_variadic:
                    # We only care about the first input provided to the socket.
                    consumed_inputs[socket_name] = socket_inputs[0]
                elif socket.is_greedy:
                    # We need to keep track of greedy inputs because we always remove them, even if they come from
                    # outside the pipeline. Otherwise, a greedy input from the user would trigger a pipeline to run
                    # indefinitely.
                    greedy_inputs_to_remove.add(socket_name)
                    consumed_inputs[socket_name] = [socket_inputs[0]]
                elif is_socket_lazy_variadic(socket):
                    # We use all inputs provided to the socket on a lazy variadic socket.
                    consumed_inputs[socket_name] = socket_inputs

        # We prune all inputs except for those that were provided from outside the pipeline (e.g. user inputs).
        pruned_inputs = {
            socket_name: [
                sock for sock in socket if sock["sender"] is None and not socket_name in greedy_inputs_to_remove
            ]
            for socket_name, socket in component_inputs.items()
        }
        pruned_inputs = {socket_name: socket for socket_name, socket in pruned_inputs.items() if len(socket) > 0}

        inputs[component_name] = pruned_inputs

        return consumed_inputs

    def _fill_queue(
        self, component_names: List[str], inputs: Dict[str, Any], component_visits: Dict[str, int]
    ) -> FIFOPriorityQueue:
        """
        Calculates the execution priority for each component and inserts it into the priority queue.

        :param component_names: Names of the components to put into the queue.
        :param inputs: Inputs to the components.
        :param component_visits: Current state of component visits.
        :returns: A prioritized queue of component names.
        """
        priority_queue = FIFOPriorityQueue()
        for component_name in component_names:
            component = self._get_component_with_graph_metadata_and_visits(
                component_name, component_visits[component_name]
            )
            priority = self._calculate_priority(component, inputs.get(component_name, {}))
            priority_queue.push(component_name, priority)

        return priority_queue

    @staticmethod
    def _calculate_priority(component: Dict, inputs: Dict) -> ComponentPriority:
        """
        Calculates the execution priority for a component depending on the component's inputs.

        :param component: Component metadata and component instance.
        :param inputs: Inputs to the component.
        :returns: Priority value for the component.
        """
        if not can_component_run(component, inputs):
            return ComponentPriority.BLOCKED
        elif is_any_greedy_socket_ready(component, inputs) and are_all_sockets_ready(component, inputs):
            return ComponentPriority.HIGHEST
        elif all_predecessors_executed(component, inputs):
            return ComponentPriority.READY
        elif are_all_lazy_variadic_sockets_resolved(component, inputs):
            return ComponentPriority.DEFER
        else:
            return ComponentPriority.DEFER_LAST

    def _get_component_with_graph_metadata_and_visits(self, component_name: str, visits: int) -> Dict[str, Any]:
        """
        Returns the component instance alongside input/output-socket metadata from the graph and adds current visits.

        We can't store visits in the pipeline graph because this would prevent reentrance / thread-safe execution.

        :param component_name: The name of the component.
        :param visits: Number of visits for the component.
        :returns: Dict including component instance, input/output-sockets and visits.
        """
        comp_dict = self.graph.nodes[component_name]
        comp_dict = {**comp_dict, "visits": visits}
        return comp_dict

    def _get_next_runnable_component(
        self, priority_queue: FIFOPriorityQueue, component_visits: Dict[str, int]
    ) -> Union[Tuple[ComponentPriority, str, Dict[str, Any]], None]:
        """
        Returns the next runnable component alongside its metadata from the priority queue.

        :param priority_queue: Priority queue of component names.
        :param component_visits: Current state of component visits.
        :returns: The next runnable component, the component name, and its priority
            or None if no component in the queue can run.
        :raises: PipelineMaxComponentRuns if the next runnable component has exceeded the maximum number of runs.
        """
        priority_and_component_name: Union[Tuple[ComponentPriority, str], None] = (
            None if (item := priority_queue.get()) is None else (ComponentPriority(item[0]), str(item[1]))
        )

        if priority_and_component_name is not None and priority_and_component_name[0] != ComponentPriority.BLOCKED:
            priority, component_name = priority_and_component_name
            component = self._get_component_with_graph_metadata_and_visits(
                component_name, component_visits[component_name]
            )
            if component["visits"] > self._max_runs_per_component:
                msg = f"Maximum run count {self._max_runs_per_component} reached for component '{component_name}'"
                raise PipelineMaxComponentRuns(msg)

            return priority, component_name, component

        return None

    @staticmethod
    def _add_missing_input_defaults(component_inputs: Dict[str, Any], component_input_sockets: Dict[str, InputSocket]):
        """
        Updates the inputs with the default values for the inputs that are missing

        :param component_inputs: Inputs for the component.
        :param component_input_sockets: Input sockets of the component.
        """
        for name, socket in component_input_sockets.items():
            if not socket.is_mandatory and name not in component_inputs:
                if socket.is_variadic:
                    component_inputs[name] = [socket.default_value]
                else:
                    component_inputs[name] = socket.default_value

        return component_inputs

    def _tiebreak_waiting_components(
        self,
        component_name: str,
        priority: ComponentPriority,
        priority_queue: FIFOPriorityQueue,
        topological_sort: Union[Dict[str, int], None],
    ):
        """
        Decides which component to run when multiple components are waiting for inputs with the same priority.

        :param component_name: The name of the component.
        :param priority: Priority of the component.
        :param priority_queue: Priority queue of component names.
        :param topological_sort: Cached topological sort of all components in the pipeline.
        """
        components_with_same_priority = [component_name]

        while len(priority_queue) > 0:
            next_priority, next_component_name = priority_queue.peek()
            if next_priority == priority:
                priority_queue.pop()  # actually remove the component
                components_with_same_priority.append(next_component_name)
            else:
                break

        if len(components_with_same_priority) > 1:
            if topological_sort is None:
                if networkx.is_directed_acyclic_graph(self.graph):
                    topological_sort = networkx.lexicographical_topological_sort(self.graph)
                    topological_sort = {node: idx for idx, node in enumerate(topological_sort)}
                else:
                    condensed = networkx.condensation(self.graph)
                    condensed_sorted = {node: idx for idx, node in enumerate(networkx.topological_sort(condensed))}
                    topological_sort = {
                        component_name: condensed_sorted[node]
                        for component_name, node in condensed.graph["mapping"].items()
                    }

            components_with_same_priority = sorted(
                components_with_same_priority, key=lambda comp_name: (topological_sort[comp_name], comp_name.lower())
            )

            component_name = components_with_same_priority[0]

        return component_name, topological_sort

    @staticmethod
    def _write_component_outputs(
        component_name: str,
        component_outputs: Dict[str, Any],
        inputs: Dict[str, Any],
        receivers: List[Tuple],
        include_outputs_from: Set[str],
    ) -> Dict[str, Any]:
        """
        Distributes the outputs of a component to the input sockets that it is connected to.

        :param component_name: The name of the component.
        :param component_outputs: The outputs of the component.
        :param inputs: The current global input state.
        :param receivers: List of components that receive inputs from the component.
        :param include_outputs_from: List of component names that should always return an output from the pipeline.
        """
        for receiver_name, sender_socket, receiver_socket in receivers:
            # We either get the value that was produced by the actor or we use the _NO_OUTPUT_PRODUCED class to indicate
            # that the sender did not produce an output for this socket.
            # This allows us to track if a predecessor already ran but did not produce an output.
            value = component_outputs.get(sender_socket.name, _NO_OUTPUT_PRODUCED)

            if receiver_name not in inputs:
                inputs[receiver_name] = {}

            if is_socket_lazy_variadic(receiver_socket):
                # If the receiver socket is lazy variadic, we append the new input.
                # Lazy variadic sockets can collect multiple inputs.
                _write_to_lazy_variadic_socket(
                    inputs=inputs,
                    receiver_name=receiver_name,
                    receiver_socket_name=receiver_socket.name,
                    component_name=component_name,
                    value=value,
                )
            else:
                # If the receiver socket is not lazy variadic, it is greedy variadic or non-variadic.
                # We overwrite with the new input if it's not _NO_OUTPUT_PRODUCED or if the current value is None.
                _write_to_standard_socket(
                    inputs=inputs,
                    receiver_name=receiver_name,
                    receiver_socket_name=receiver_socket.name,
                    component_name=component_name,
                    value=value,
                )

        # If we want to include all outputs from this actor in the final outputs, we don't need to prune any consumed
        # outputs
        if component_name in include_outputs_from:
            return component_outputs

        # We prune outputs that were consumed by any receiving sockets.
        # All remaining outputs will be added to the final outputs of the pipeline.
        consumed_outputs = {sender_socket.name for _, sender_socket, __ in receivers}
        pruned_outputs = {key: value for key, value in component_outputs.items() if key not in consumed_outputs}

        return pruned_outputs

    @staticmethod
    def _is_queue_stale(priority_queue: FIFOPriorityQueue) -> bool:
        """
        Checks if the priority queue needs to be recomputed because the priorities might have changed.

        :param priority_queue: Priority queue of component names.
        """
        return len(priority_queue) == 0 or priority_queue.peek()[0] > ComponentPriority.READY

    @staticmethod
    def validate_pipeline(priority_queue: FIFOPriorityQueue) -> None:
        """
        Validate the pipeline to check if it is blocked or has no valid entry point.

        :param priority_queue: Priority queue of component names.
        :raises PipelineRuntimeError:
            If the pipeline is blocked or has no valid entry point.
        """
        if len(priority_queue) == 0:
            return

        candidate = priority_queue.peek()
        if candidate is not None and candidate[0] == ComponentPriority.BLOCKED:
            raise PipelineComponentsBlockedError()

    def _find_super_components(self) -> list[tuple[str, Component]]:
        """
        Find all SuperComponents in the pipeline.

        :returns:
            List of tuples containing (component_name, component_instance) representing a SuperComponent.
        """

        super_components = []
        for comp_name, comp in self.walk():
            # a SuperComponent has a "pipeline" attribute which itself a Pipeline instance
            # we don't test against SuperComponent because doing so always lead to circular imports
            if hasattr(comp, "pipeline") and isinstance(comp.pipeline, self.__class__):
                super_components.append((comp_name, comp))
        return super_components

    def _merge_super_component_pipelines(self) -> Tuple["networkx.MultiDiGraph", Dict[str, str]]:
        """
        Merge the internal pipelines of SuperComponents into the main pipeline graph structure.

        This creates a new networkx.MultiDiGraph containing all the components from both the main pipeline
        and all the internal SuperComponents' pipelines. The SuperComponents are removed and their internal
        components are connected to corresponding input and output sockets of the main pipeline.

        :returns:
            A tuple containing:
            - A networkx.MultiDiGraph with the expanded structure of the main pipeline and all it's SuperComponents
            - A dictionary mapping component names to boolean indicating that this component was part of a
              SuperComponent
            - A dictionary mapping component names to their SuperComponent name
        """
        merged_graph = self.graph.copy()
        super_component_mapping: Dict[str, str] = {}

        for super_name, super_component in self._find_super_components():
            internal_pipeline = super_component.pipeline  # type: ignore
            internal_graph = internal_pipeline.graph.copy()

            # Mark all components in the internal pipeline as being part of a SuperComponent
            for node in internal_graph.nodes():
                super_component_mapping[node] = super_name

            # edges connected to the super component
            incoming_edges = list(merged_graph.in_edges(super_name, data=True))
            outgoing_edges = list(merged_graph.out_edges(super_name, data=True))

            # merge the SuperComponent graph into the main graph and remove the super component node
            # since its components are now part of the main graph
            merged_graph = networkx.compose(merged_graph, internal_graph)
            merged_graph.remove_node(super_name)

            # get the entry and exit points of the SuperComponent internal pipeline
            entry_points = [n for n in internal_graph.nodes() if internal_graph.in_degree(n) == 0]
            exit_points = [n for n in internal_graph.nodes() if internal_graph.out_degree(n) == 0]

            # connect the incoming edges to entry points
            for sender, _, edge_data in incoming_edges:
                sender_socket = edge_data["from_socket"]
                for entry_point in entry_points:
                    # find a matching input socket in the entry point
                    entry_point_sockets = internal_graph.nodes[entry_point]["input_sockets"]
                    for socket_name, socket in entry_point_sockets.items():
                        if _types_are_compatible(sender_socket.type, socket.type, self._connection_type_validation):
                            merged_graph.add_edge(
                                sender,
                                entry_point,
                                key=f"{sender_socket.name}/{socket_name}",
                                conn_type=_type_name(sender_socket.type),
                                from_socket=sender_socket,
                                to_socket=socket,
                                mandatory=socket.is_mandatory,
                            )

            # connect outgoing edges from exit points
            for _, receiver, edge_data in outgoing_edges:
                receiver_socket = edge_data["to_socket"]
                for exit_point in exit_points:
                    # find a matching output socket in the exit point
                    exit_point_sockets = internal_graph.nodes[exit_point]["output_sockets"]
                    for socket_name, socket in exit_point_sockets.items():
                        if _types_are_compatible(socket.type, receiver_socket.type, self._connection_type_validation):
                            merged_graph.add_edge(
                                exit_point,
                                receiver,
                                key=f"{socket_name}/{receiver_socket.name}",
                                conn_type=_type_name(socket.type),
                                from_socket=socket,
                                to_socket=receiver_socket,
                                mandatory=receiver_socket.is_mandatory,
                            )

        return merged_graph, super_component_mapping


def _connections_status(
    sender_node: str, receiver_node: str, sender_sockets: List[OutputSocket], receiver_sockets: List[InputSocket]
) -> str:
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


# Utility functions for writing to sockets


def _write_to_lazy_variadic_socket(
    inputs: Dict[str, Any], receiver_name: str, receiver_socket_name: str, component_name: str, value: Any
) -> None:
    """
    Write to a lazy variadic socket.

    Mutates inputs in place.
    """
    if not inputs[receiver_name].get(receiver_socket_name):
        inputs[receiver_name][receiver_socket_name] = []

    inputs[receiver_name][receiver_socket_name].append({"sender": component_name, "value": value})


def _write_to_standard_socket(
    inputs: Dict[str, Any], receiver_name: str, receiver_socket_name: str, component_name: str, value: Any
) -> None:
    """
    Write to a greedy variadic or non-variadic socket.

    Mutates inputs in place.
    """
    current_value = inputs[receiver_name].get(receiver_socket_name)

    # Only overwrite if there's no existing value, or we have a new value to provide
    if current_value is None or value is not _NO_OUTPUT_PRODUCED:
        inputs[receiver_name][receiver_socket_name] = [{"sender": component_name, "value": value}]
