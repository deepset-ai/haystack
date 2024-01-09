# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any, Dict, List, Union, TypeVar, Type, Set

import os
import json
import datetime
import logging
import importlib
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import networkx  # type:ignore

from haystack.core.component import component, Component, InputSocket, OutputSocket
from haystack.core.errors import (
    PipelineError,
    PipelineConnectError,
    PipelineMaxLoops,
    PipelineRuntimeError,
    PipelineValidationError,
)
from haystack.core.pipeline.descriptions import find_pipeline_outputs
from haystack.core.pipeline.draw.draw import _draw, RenderingEngines
from haystack.core.pipeline.validation import validate_pipeline_input, find_pipeline_inputs
from haystack.core.component.connection import Connection, parse_connect_string
from haystack.core.type_utils import _type_name
from haystack.core.serialization import component_to_dict, component_from_dict

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
        debug_path: Union[Path, str] = Path(".canals_debug/"),
    ):
        """
        Creates the Pipeline.

        Args:
            metadata: arbitrary dictionary to store metadata about this pipeline. Make sure all the values contained in
                this dictionary can be serialized and deserialized if you wish to save this pipeline to file with
                `save_pipelines()/load_pipelines()`.
            max_loops_allowed: how many times the pipeline can run the same node before throwing an exception.
            debug_path: when debug is enabled in `run()`, where to save the debug data.
        """
        self.metadata = metadata or {}
        self.max_loops_allowed = max_loops_allowed
        self.graph = networkx.MultiDiGraph()
        self._connections: List[Connection] = []
        self._mandatory_connections: Dict[str, List[Connection]] = defaultdict(list)
        self._debug: Dict[int, Dict[str, Any]] = {}
        self._debug_path = Path(debug_path)

    def __eq__(self, other) -> bool:
        """
        Equal pipelines share every metadata, node and edge, but they're not required to use
        the same node instances: this allows pipeline saved and then loaded back to be equal to themselves.
        """
        if (
            not isinstance(other, type(self))
            or not getattr(self, "metadata") == getattr(other, "metadata")
            or not getattr(self, "max_loops_allowed") == getattr(other, "max_loops_allowed")
            or not hasattr(self, "graph")
            or not hasattr(other, "graph")
        ):
            return False

        return (
            self.graph.adj == other.graph.adj
            and self._comparable_nodes_list(self.graph) == self._comparable_nodes_list(other.graph)
            and self.graph.graph == other.graph.graph
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns this Pipeline instance as a dictionary.
        This is meant to be an intermediate representation but it can be also used to save a pipeline to file.
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
        Creates a Pipeline instance from a dictionary.
        A sample `data` dictionary could be formatted like so:
        ```
        {
            "metadata": {"test": "test"},
            "max_loops_allowed": 100,
            "components": {
                "add_two": {
                    "type": "AddFixedValue",
                    "init_parameters": {"add": 2},
                },
                "add_default": {
                    "type": "AddFixedValue",
                    "init_parameters": {"add": 1},
                },
                "double": {
                    "type": "Double",
                },
            },
            "connections": [
                {"sender": "add_two.result", "receiver": "double.value"},
                {"sender": "double.value", "receiver": "add_default.value"},
            ],
        }
        ```

        Supported kwargs:
        `components`: a dictionary of {name: instance} to reuse instances of components instead of creating new ones.
        """
        metadata = data.get("metadata", {})
        max_loops_allowed = data.get("max_loops_allowed", 100)
        debug_path = Path(data.get("debug_path", ".canals_debug/"))
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
                        logger.debug("Trying to import %s", module)
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
            pipe.connect(connect_from=connection["sender"], connect_to=connection["receiver"])

        return pipe

    def _comparable_nodes_list(self, graph: networkx.MultiDiGraph) -> List[Dict[str, Any]]:
        """
        Replaces instances of nodes with their class name in order to make sure they're comparable.
        """
        nodes = []
        for node in graph.nodes:
            comparable_node = graph.nodes[node]
            comparable_node["instance"] = comparable_node["instance"].__class__
            nodes.append(comparable_node)
        nodes.sort()
        return nodes

    def add_component(self, name: str, instance: Component) -> None:
        """
        Create a component for the given component. Components are not connected to anything by default:
        use `Pipeline.connect()` to connect components together.

        Component names must be unique, but component instances can be reused if needed.

        Args:
            name: the name of the component.
            instance: the component instance.

        Returns:
            None

        Raises:
            ValueError: if a component with the same name already exists
            PipelineValidationError: if the given instance is not a Canals component
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

        # Create the component's input and output sockets
        input_sockets = getattr(instance, "__canals_input__", {})
        output_sockets = getattr(instance, "__canals_output__", {})

        # Add component to the graph, disconnected
        logger.debug("Adding component '%s' (%s)", name, instance)
        self.graph.add_node(
            name, instance=instance, input_sockets=input_sockets, output_sockets=output_sockets, visits=0
        )

    def connect(self, connect_from: str, connect_to: str) -> None:
        """
        Connects two components together. All components to connect must exist in the pipeline.
        If connecting to an component that has several output connections, specify the inputs and output names as
        'component_name.connections_name'.

        Args:
            connect_from: the component that delivers the value. This can be either just a component name or can be
                in the format `component_name.connection_name` if the component has multiple outputs.
            connect_to: the component that receives the value. This can be either just a component name or can be
                in the format `component_name.connection_name` if the component has multiple inputs.

        Returns:
            None

        Raises:
            PipelineConnectError: if the two components cannot be connected (for example if one of the components is
                not present in the pipeline, or the connections don't match by type, and so on).
        """
        # Edges may be named explicitly by passing 'node_name.edge_name' to connect().
        sender, sender_socket_name = parse_connect_string(connect_from)
        receiver, receiver_socket_name = parse_connect_string(connect_to)

        # Get the nodes data.
        try:
            from_sockets = self.graph.nodes[sender]["output_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {sender} not found in the pipeline.") from exc
        try:
            to_sockets = self.graph.nodes[receiver]["input_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {receiver} not found in the pipeline.") from exc

        # If the name of either socket is given, get the socket
        sender_socket: Optional[OutputSocket] = None
        if sender_socket_name:
            sender_socket = from_sockets.get(sender_socket_name)
            if not sender_socket:
                raise PipelineConnectError(
                    f"'{connect_from} does not exist. "
                    f"Output connections of {sender} are: "
                    + ", ".join([f"{name} (type {_type_name(socket.type)})" for name, socket in from_sockets.items()])
                )

        receiver_socket: Optional[InputSocket] = None
        if receiver_socket_name:
            receiver_socket = to_sockets.get(receiver_socket_name)
            if not receiver_socket:
                raise PipelineConnectError(
                    f"'{connect_to} does not exist. "
                    f"Input connections of {receiver} are: "
                    + ", ".join([f"{name} (type {_type_name(socket.type)})" for name, socket in to_sockets.items()])
                )

        # Look for a matching connection among the possible ones.
        # Note that if there is more than one possible connection but two sockets match by name, they're paired.
        sender_socket_candidates: List[OutputSocket] = [sender_socket] if sender_socket else list(from_sockets.values())
        receiver_socket_candidates: List[InputSocket] = (
            [receiver_socket] if receiver_socket else list(to_sockets.values())
        )

        connection = Connection.from_list_of_sockets(
            sender, sender_socket_candidates, receiver, receiver_socket_candidates
        )

        # Connection must be valid on both sender/receiver sides
        if (
            not connection.sender_socket
            or not connection.receiver_socket
            or not connection.sender
            or not connection.receiver
        ):
            raise PipelineConnectError("Connection must have both sender and receiver: {connection}")

        # Create the connection
        logger.debug(
            "Connecting '%s.%s' to '%s.%s'",
            connection.sender,
            connection.sender_socket.name,
            connection.receiver,
            connection.receiver_socket.name,
        )

        self.graph.add_edge(
            connection.sender,
            connection.receiver,
            key=f"{connection.sender_socket.name}/{connection.receiver_socket.name}",
            conn_type=_type_name(connection.sender_socket.type),
            from_socket=connection.sender_socket,
            to_socket=connection.receiver_socket,
        )

        self._connections.append(connection)
        if connection.is_mandatory:
            self._mandatory_connections[connection.receiver].append(connection)

    def get_component(self, name: str) -> Component:
        """
        Returns an instance of a component.

        Args:
            name: the name of the component

        Returns:
            The instance of that component.

        Raises:
            ValueError: if a component with that name is not present in the pipeline.
        """
        try:
            return self.graph.nodes[name]["instance"]
        except KeyError as exc:
            raise ValueError(f"Component named {name} not found in the pipeline.") from exc

    def inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the inputs of a pipeline. Each key in the dictionary
        corresponds to a component name, and its value is another dictionary that describes the
        input sockets of that component, including their types and whether they are optional.

        Returns:
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

        Returns:
            A dictionary where each key is a pipeline component name and each value is a dictionary of
            output sockets of that component.
        """
        outputs = {
            comp: {socket.name: {"type": socket.type} for socket in data}
            for comp, data in find_pipeline_outputs(self.graph).items()
            if data
        }
        return outputs

    def draw(self, path: Path, engine: RenderingEngines = "mermaid-image") -> None:
        """
        Draws the pipeline. Requires either `graphviz` as a system dependency, or an internet connection for Mermaid.
        Run `pip install graphviz` or `pip install mermaid` to install missing dependencies.

        Args:
            path: where to save the diagram.
            engine: which format to save the graph as. Accepts 'graphviz', 'mermaid-text', 'mermaid-image'.
                Default is 'mermaid-image'.

        Returns:
            None

        Raises:
            ImportError: if `engine='graphviz'` and `pygraphviz` is not installed.
            HTTPConnectionError: (and similar) if the internet connection is down or other connection issues.
        """
        _draw(graph=networkx.MultiDiGraph(self.graph), path=path, engine=engine)

    def warm_up(self):
        """
        Make sure all nodes are warm.

        It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
        without re-initializing everything.
        """
        for node in self.graph.nodes:
            if hasattr(self.graph.nodes[node]["instance"], "warm_up"):
                logger.info("Warming up component %s...", node)
                self.graph.nodes[node]["instance"].warm_up()

    def run(self, data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:  # pylint: disable=too-many-locals
        """
        Runs the pipeline.

        Args:
            data: the inputs to give to the input components of the Pipeline.
            debug: whether to collect and return debug information.

        Returns:
            A dictionary with the outputs of the output components of the Pipeline.

        Raises:
            PipelineRuntimeError: if the any of the components fail or return unexpected output.
        """
        self._clear_visits_count()
        data = validate_pipeline_input(self.graph, input_values=data)
        logger.info("Pipeline execution started.")

        self._debug = {}
        if debug:
            logger.info("Debug mode ON.")
            os.makedirs("debug", exist_ok=True)

        logger.debug(
            "Mandatory connections:\n%s",
            "\n".join(
                f" - {component}: {', '.join([str(s) for s in sockets])}"
                for component, sockets in self._mandatory_connections.items()
            ),
        )

        self.warm_up()

        # Prepare the inputs buffers and components queue
        components_queue: List[str] = []
        mandatory_values_buffer: Dict[Connection, Any] = {}
        optional_values_buffer: Dict[Connection, Any] = {}
        pipeline_output: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for node_name, input_data in data.items():
            for socket_name, value in input_data.items():
                # Make a copy of the input value so components don't need to
                # take care of mutability.
                value = deepcopy(value)
                connection = Connection(
                    None, None, node_name, self.graph.nodes[node_name]["input_sockets"][socket_name]
                )
                self._add_value_to_buffers(
                    value, connection, components_queue, mandatory_values_buffer, optional_values_buffer
                )

        # *** PIPELINE EXECUTION LOOP ***
        step = 0
        while components_queue:  # pylint: disable=too-many-nested-blocks
            step += 1
            if debug:
                self._record_pipeline_step(
                    step, components_queue, mandatory_values_buffer, optional_values_buffer, pipeline_output
                )

            component_name = components_queue.pop(0)
            logger.debug("> Queue at step %s: %s %s", step, component_name, components_queue)
            self._check_max_loops(component_name)

            # **** RUN THE NODE ****
            if not self._ready_to_run(component_name, mandatory_values_buffer, components_queue):
                continue

            inputs = {
                **self._extract_inputs_from_buffer(component_name, mandatory_values_buffer),
                **self._extract_inputs_from_buffer(component_name, optional_values_buffer),
            }
            outputs = self._run_component(name=component_name, inputs=dict(inputs))

            # **** PROCESS THE OUTPUT ****
            for socket_name, value in outputs.items():
                targets = self._collect_targets(component_name, socket_name)
                if not targets:
                    pipeline_output[component_name][socket_name] = value
                else:
                    for target in targets:
                        self._add_value_to_buffers(
                            value, target, components_queue, mandatory_values_buffer, optional_values_buffer
                        )

        if debug:
            self._record_pipeline_step(
                step + 1, components_queue, mandatory_values_buffer, optional_values_buffer, pipeline_output
            )
            os.makedirs(self._debug_path, exist_ok=True)
            with open(self._debug_path / "data.json", "w", encoding="utf-8") as datafile:
                json.dump(self._debug, datafile, indent=4, default=str)
            pipeline_output["_debug"] = self._debug  # type: ignore

        logger.info("Pipeline executed successfully.")
        return dict(pipeline_output)

    def _record_pipeline_step(
        self, step, components_queue, mandatory_values_buffer, optional_values_buffer, pipeline_output
    ):
        """
        Stores a snapshot of this step into the self.debug dictionary of the pipeline.
        """
        self._debug[step] = {
            "time": datetime.datetime.now(),
            "components_queue": components_queue,
            "mandatory_values_buffer": mandatory_values_buffer,
            "optional_values_buffer": optional_values_buffer,
            "pipeline_output": pipeline_output,
        }

    def _clear_visits_count(self):
        """
        Make sure all nodes's visits count is zero.
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]["visits"] = 0

    def _check_max_loops(self, component_name: str):
        """
        Verify whether this component run too many times.
        """
        if self.graph.nodes[component_name]["visits"] > self.max_loops_allowed:
            raise PipelineMaxLoops(
                f"Maximum loops count ({self.max_loops_allowed}) exceeded for component '{component_name}'."
            )

    def _add_value_to_buffers(
        self,
        value: Any,
        connection: Connection,
        components_queue: List[str],
        mandatory_values_buffer: Dict[Connection, Any],
        optional_values_buffer: Dict[Connection, Any],
    ):
        """
        Given a value and the connection it is being sent on, it updates the buffers and the components queue.
        """
        if connection.is_mandatory:
            mandatory_values_buffer[connection] = value
            if connection.receiver and connection.receiver not in components_queue:
                components_queue.append(connection.receiver)
        else:
            optional_values_buffer[connection] = value

    def _ready_to_run(
        self, component_name: str, mandatory_values_buffer: Dict[Connection, Any], components_queue: List[str]
    ) -> bool:
        """
        Returns True if a component is ready to run, False otherwise.
        """
        connections_with_value = {conn for conn in mandatory_values_buffer.keys() if conn.receiver == component_name}
        expected_connections = set(self._mandatory_connections[component_name])
        if expected_connections.issubset(connections_with_value):
            logger.debug("Component '%s' is ready to run. All mandatory values were received.", component_name)
            return True

        # Collect the missing values still being computed we need to wait for
        missing_connections: Set[Connection] = expected_connections - connections_with_value
        connections_to_wait = []
        for missing_conn in missing_connections:
            if any(
                networkx.has_path(self.graph, component_to_run, missing_conn.sender)
                for component_to_run in components_queue
            ):
                connections_to_wait.append(missing_conn)

        if not connections_to_wait:
            # None of the missing values are needed to visit this part of the graph: we can run the component
            logger.debug(
                "Component '%s' is ready to run. A variadic input parameter received all the expected values.",
                component_name,
            )
            return True

        # Component can't run, waiting for the values needed by `connections_to_wait`
        logger.debug(
            "Component '%s' is not ready to run, some values are still missing: %s", component_name, connections_to_wait
        )
        # Put the component back in the queue
        components_queue.append(component_name)
        return False

    def _extract_inputs_from_buffer(self, component_name: str, buffer: Dict[Connection, Any]) -> Dict[str, Any]:
        """
        Extract a component's input values from one of the value buffers.
        """
        inputs = defaultdict(list)
        connections: List[Connection] = []

        for connection in buffer.keys():
            if connection.receiver == component_name:
                connections.append(connection)

        for key in connections:
            value = buffer.pop(key)
            if key.receiver_socket:
                if key.receiver_socket.is_variadic:
                    inputs[key.receiver_socket.name].append(value)
                else:
                    inputs[key.receiver_socket.name] = value
        return inputs

    def _run_component(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Once we're confident this component is ready to run, run it and collect the output.
        """
        self.graph.nodes[name]["visits"] += 1
        instance = self.graph.nodes[name]["instance"]
        try:
            logger.info("* Running %s", name)
            logger.debug("   '%s' inputs: %s", name, inputs)

            outputs = instance.run(**inputs)

            # Unwrap the output
            logger.debug("   '%s' outputs: %s\n", name, outputs)

            # Make sure the component returned a dictionary
            if not isinstance(outputs, dict):
                raise PipelineRuntimeError(
                    f"Component '{name}' returned a value of type '{_type_name(type(outputs))}' instead of a dict. "
                    "Components must always return dictionaries: check the the documentation."
                )

        except Exception as e:
            raise PipelineRuntimeError(
                f"{name} raised '{e.__class__.__name__}: {e}' \nInputs: {inputs}\n\n"
                "See the stacktrace above for more information."
            ) from e

        return outputs

    def _collect_targets(self, component_name: str, socket_name: str) -> List[Connection]:
        """
        Given a component and an output socket name, return a list of Connections
        for which they represent the sender. Used to route output.
        """
        return [
            connection
            for connection in self._connections
            if connection.sender == component_name
            and connection.sender_socket
            and connection.sender_socket.name == socket_name
        ]
