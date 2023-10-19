# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any, Dict, List, Literal, Union

import os
import json
import datetime
import logging
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

import networkx

from canals.component import component, Component, InputSocket, OutputSocket
from canals.errors import (
    PipelineError,
    PipelineConnectError,
    PipelineMaxLoops,
    PipelineRuntimeError,
    PipelineValidationError,
)
from canals.pipeline.draw import _draw, _convert_for_debug, RenderingEngines
from canals.pipeline.validation import validate_pipeline_input
from canals.pipeline.connections import parse_connection, _find_unambiguous_connection
from canals.type_utils import _type_name
from canals.serialization import component_to_dict, component_from_dict

logger = logging.getLogger(__name__)


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
        self.debug: Dict[int, Dict[str, Any]] = {}
        self.debug_path = Path(debug_path)

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
        for name, instance in self.graph.nodes(data="instance"):
            components[name] = component_to_dict(instance)

        connections = []
        for sender, receiver, edge_data in self.graph.edges.data():
            sender_socket = edge_data["from_socket"].name
            receiver_socket = edge_data["to_socket"].name
            connections.append(
                {
                    "sender": f"{sender}.{sender_socket}",
                    "receiver": f"{receiver}.{receiver_socket}",
                }
            )
        return {
            "metadata": self.metadata,
            "max_loops_allowed": self.max_loops_allowed,
            "components": components,
            "connections": connections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "Pipeline":
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
        pipe = cls(
            metadata=metadata,
            max_loops_allowed=max_loops_allowed,
            debug_path=debug_path,
        )
        components_to_reuse = kwargs.get("components", {})
        for name, component_data in data.get("components", {}).items():
            if name in components_to_reuse:
                # Reuse an instance
                instance = components_to_reuse[name]
            else:
                if "type" not in component_data:
                    raise PipelineError(f"Missing 'type' in component '{name}'")
                if component_data["type"] not in component.registry:
                    raise PipelineError(f"Component '{component_data['type']}' not imported.")
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
            name,
            instance=instance,
            input_sockets=input_sockets,
            output_sockets=output_sockets,
            visits=0,
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
        from_node, from_socket_name = parse_connection(connect_from)
        to_node, to_socket_name = parse_connection(connect_to)

        # Get the nodes data.
        try:
            from_sockets = self.graph.nodes[from_node]["output_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {from_node} not found in the pipeline.") from exc

        try:
            to_sockets = self.graph.nodes[to_node]["input_sockets"]
        except KeyError as exc:
            raise ValueError(f"Component named {to_node} not found in the pipeline.") from exc

        # If the name of either socket is given, get the socket
        if from_socket_name:
            from_socket = from_sockets.get(from_socket_name, None)
            if not from_socket:
                raise PipelineConnectError(
                    f"'{from_node}.{from_socket_name} does not exist. "
                    f"Output connections of {from_node} are: "
                    + ", ".join([f"{name} (type {_type_name(socket.type)})" for name, socket in from_sockets.items()])
                )
        if to_socket_name:
            to_socket = to_sockets.get(to_socket_name, None)
            if not to_socket:
                raise PipelineConnectError(
                    f"'{to_node}.{to_socket_name} does not exist. "
                    f"Input connections of {to_node} are: "
                    + ", ".join([f"{name} (type {_type_name(socket.type)})" for name, socket in to_sockets.items()])
                )

        # Look for an unambiguous connection among the possible ones.
        # Note that if there is more than one possible connection but two sockets match by name, they're paired.
        from_sockets = [from_socket] if from_socket_name else list(from_sockets.values())
        to_sockets = [to_socket] if to_socket_name else list(to_sockets.values())
        from_socket, to_socket = _find_unambiguous_connection(
            sender_node=from_node, sender_sockets=from_sockets, receiver_node=to_node, receiver_sockets=to_sockets
        )

        # Connect the components on these sockets
        self._direct_connect(from_node=from_node, from_socket=from_socket, to_node=to_node, to_socket=to_socket)

    def _direct_connect(self, from_node: str, from_socket: OutputSocket, to_node: str, to_socket: InputSocket) -> None:
        """
        Directly connect socket to socket. This method does not type-check the connections: use 'Pipeline.connect()'
        instead (which uses 'find_unambiguous_connection()' to validate types).
        """
        # Make sure the receiving socket isn't already connected, unless it's variadic. Sending sockets can be
        # connected as many times as needed, so they don't need this check
        if to_socket.sender and not to_socket.is_variadic:
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_socket.name}' with '{to_node}.{to_socket.name}': "
                f"{to_node}.{to_socket.name} is already connected to {to_socket.sender}.\n"
            )

        # Create the connection
        logger.debug("Connecting '%s.%s' to '%s.%s'", from_node, from_socket.name, to_node, to_socket.name)
        edge_key = f"{from_socket.name}/{to_socket.name}"
        self.graph.add_edge(
            from_node,
            to_node,
            key=edge_key,
            conn_type=_type_name(from_socket.type),
            from_socket=from_socket,
            to_socket=to_socket,
        )

        # Stores the name of the nodes that will send its output to this socket
        to_socket.sender.append(from_node)

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

    def draw(self, path: Path, engine: RenderingEngines = "mermaid-image") -> None:
        """
        Draws the pipeline. Requires either `graphviz` as a system dependency, or an internet connection for Mermaid.
        Run `pip install canals[graphviz]` or `pip install canals[mermaid]` to install missing dependencies.

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

    def run(self, data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """
        Runs the pipeline.

        Args:
            data: the inputs to give to the input components of the Pipeline.
            parameters: a dictionary with all the parameters of all the components, namespaced by component.
            debug: whether to collect and return debug information.

        Returns:
            A dictionary with the outputs of the output components of the Pipeline.

        Raises:
            PipelineRuntimeError: if the any of the components fail or return unexpected output.
        """
        # **** The Pipeline.run() algorithm ****
        #
        # Nodes are run as soon as an input for them appears in the inputs buffer.
        # When there's more than a node at once in the buffer (which means some
        # branches are running in parallel or that there are loops) they are selected to
        # run in FIFO order by the `inputs_buffer` OrderedDict.
        #
        # Inputs are labeled with the name of the node they're aimed for:
        #
        #   ````
        #   inputs_buffer[target_node] = {"input_name": input_value, ...}
        #   ```
        #
        # Nodes should wait until all the necessary input data has arrived before running.
        # If they're popped from the input_buffer before they're ready, they're put back in.
        # If the pipeline has branches of different lengths, it's possible that a node has to
        # wait a bit and let other nodes pass before receiving all the input data it needs.
        #
        # Chetsheet for networkx data access:
        # - Name of the node       # self.graph.nodes  (List[str])
        # - Node instance          # self.graph.nodes[node]["instance"]
        # - Input nodes            # [e[0] for e in self.graph.in_edges(node)]
        # - Output nodes           # [e[1] for e in self.graph.out_edges(node)]
        # - Output edges           # [e[2]["label"] for e in self.graph.out_edges(node, data=True)]
        #
        # if debug:
        #     os.makedirs("debug", exist_ok=True)

        data = validate_pipeline_input(self.graph, input_values=data)

        logger.info("Pipeline execution started.")
        inputs_buffer = self._prepare_inputs_buffer(data)
        pipeline_output: Dict[str, Dict[str, Any]] = {}
        self._clear_visits_count()
        self.warm_up()

        if debug:
            logger.info("Debug mode ON.")
        self.debug = {}

        # *** PIPELINE EXECUTION LOOP ***
        # We select the nodes to run by popping them in FIFO order from the inputs buffer.
        step = 0
        while inputs_buffer:
            step += 1
            if debug:
                self._record_pipeline_step(step, inputs_buffer, pipeline_output)
            logger.debug("> Queue at step %s: %s", step, {k: list(v.keys()) for k, v in inputs_buffer.items()})

            component_name, inputs = inputs_buffer.popitem(last=False)  # FIFO

            # Make sure it didn't run too many times already
            self._check_max_loops(component_name)

            # **** IS IT MY TURN YET? ****
            # Check if the component should be run or not
            action = self._calculate_action(name=component_name, inputs=inputs, inputs_buffer=inputs_buffer)

            # This component is missing data: let's put it back in the queue and wait.
            if action == "wait":
                if not inputs_buffer:
                    # What if there are no components to wait for?
                    raise PipelineRuntimeError(
                        f"'{component_name}' is stuck waiting for input, but there are no other components to run. "
                        "This is likely a Canals bug. Open an issue at https://github.com/deepset-ai/canals."
                    )

                inputs_buffer[component_name] = inputs
                continue

            # This component did not receive the input it needs: it must be on a skipped branch. Let's not run it.
            if action == "skip":
                self.graph.nodes[component_name]["visits"] += 1
                inputs_buffer = self._skip_downstream_unvisited_nodes(
                    component_name=component_name, inputs_buffer=inputs_buffer
                )
                continue

            if action == "remove":
                # This component has no reason of being in the run queue and we need to remove it. For example, this can happen to components that are connected to skipped branches of the pipeline.
                continue

            # **** RUN THE NODE ****
            # It is our turn! The node is ready to run and all necessary inputs are present
            output = self._run_component(name=component_name, inputs=inputs)

            # **** PROCESS THE OUTPUT ****
            # The node run successfully. Let's store or distribute the output it produced, if it's valid.
            if not self.graph.out_edges(component_name):
                # Note: if a node outputs many times (like in loops), the output will be overwritten
                pipeline_output[component_name] = output
            else:
                inputs_buffer = self._route_output(
                    node_results=output, node_name=component_name, inputs_buffer=inputs_buffer
                )

        if debug:
            self._record_pipeline_step(step + 1, inputs_buffer, pipeline_output)

            # Save to json
            os.makedirs(self.debug_path, exist_ok=True)
            with open(self.debug_path / "data.json", "w", encoding="utf-8") as datafile:
                json.dump(self.debug, datafile, indent=4, default=str)

            # Store in the output
            pipeline_output["_debug"] = self.debug  # type: ignore

        logger.info("Pipeline executed successfully.")
        return pipeline_output

    def _record_pipeline_step(self, step, inputs_buffer, pipeline_output):
        """
        Stores a snapshot of this step into the self.debug dictionary of the pipeline.
        """
        mermaid_graph = _convert_for_debug(deepcopy(self.graph))
        self.debug[step] = {
            "time": datetime.datetime.now(),
            "inputs_buffer": list(inputs_buffer.items()),
            "pipeline_output": pipeline_output,
            "diagram": mermaid_graph,
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

    # This function is complex so it contains quite some logic, it needs tons of information
    # regarding a component to understand what action it should take so we have many local
    # variables and to keep things simple we also have multiple returns.
    # In the end this amount of information makes it easier to understand the internal logic so
    # we chose to ignore these pylint warnings.
    def _calculate_action(  # pylint: disable=too-many-locals, too-many-return-statements
        self, name: str, inputs: Dict[str, Any], inputs_buffer: Dict[str, Any]
    ) -> Literal["run", "wait", "skip", "remove"]:
        """
        Calculates the action to take for the component specified by `name`.
        There are four possible actions:
            * run
            * wait
            * skip
            * remove

        The below conditions are evaluated in this order.

        Component will run if at least one of the following statements is true:
            * It received all mandatory inputs
            * It received all mandatory inputs and it has no optional inputs
            * It received all mandatory inputs and all optional inputs are skipped
            * It received all mandatory inputs and some optional inputs and the rest are skipped
            * It received some of its inputs and the others are defaulted
            * It's the first component of the pipeline

        Component will wait if:
            * It received some of its inputs and the other are not skipped
            * It received all mandatory inputs and some optional inputs have not been skipped

        Component will be skipped if:
            * It never ran nor waited

        Component will be removed if:
            * It ran or waited at least once but can't do it again

        If none of the above condition is met a PipelineRuntimeError is raised.

        For simplicity sake input components that create a cycle, or components that already ran
        and don't create a cycle are considered as skipped.

        Args:
            name: Name of the component
            inputs: Values that the component will take as input
            inputs_buffer: Other components' inputs

        Returns:
            Action to take for component specifing whether it should run, wait, skip or be removed

        Raises:
            PipelineRuntimeError: If action to take can't be determined
        """

        # Upstream components/socket pairs the current component is connected to
        input_components = {
            from_node: data["to_socket"].name for from_node, _, data in self.graph.in_edges(name, data=True)
        }
        # Sockets that have received inputs from upstream components
        received_input_sockets = set(inputs.keys())

        # All components inputs, whether they're connected, default or pipeline inputs
        input_sockets: Dict[str, InputSocket] = self.graph.nodes[name]["input_sockets"].keys()
        optional_input_sockets = {
            socket.name for socket in self.graph.nodes[name]["input_sockets"].values() if socket.is_optional
        }
        mandatory_input_sockets = {
            socket.name for socket in self.graph.nodes[name]["input_sockets"].values() if not socket.is_optional
        }

        # Components that are in the inputs buffer and have no inputs assigned are considered skipped
        skipped_components = {n for n, v in inputs_buffer.items() if not v}

        # Sockets that have their upstream component marked as skipped
        skipped_optional_input_sockets = {
            sockets["to_socket"].name
            for from_node, _, sockets in self.graph.in_edges(name, data=True)
            if from_node in skipped_components and sockets["to_socket"].name in optional_input_sockets
        }

        for from_node, socket in input_components.items():
            if socket not in optional_input_sockets:
                continue
            loops_back = networkx.has_path(self.graph, name, from_node)
            has_run = self.graph.nodes[from_node]["visits"] > 0
            if loops_back or has_run:
                # Consider all input components that loop back to current component
                # or that have already run at least once as skipped.
                # This must be done to correctly handle cycles in the pipeline or we
                # would reach a dead lock in components that have multiple inputs and
                # one of these forms a cycle.
                skipped_optional_input_sockets.add(socket)

        ##############
        # RUN CHECKS #
        ##############
        if (
            mandatory_input_sockets.issubset(received_input_sockets)
            and input_sockets == received_input_sockets | mandatory_input_sockets | skipped_optional_input_sockets
        ):
            # We received all mandatory inputs and:
            #   * There are no optional inputs or
            #   * All optional inputs are skipped or
            #   * We received part of the optional inputs, the rest are skipped
            if not optional_input_sockets:
                logger.debug("Component '%s' is ready to run. All mandatory inputs received.", name)
            else:
                logger.debug(
                    "Component '%s' is ready to run. All mandatory inputs received, skipped optional inputs: %s",
                    name,
                    skipped_optional_input_sockets,
                )
            return "run"

        if set(input_components.values()).issubset(received_input_sockets):
            # We have data from each connected input component.
            # We reach this when the current component is the first of the pipeline or
            # when it has defaults and all its input components have run.
            logger.debug("Component '%s' is ready to run. All expected inputs were received.", name)
            return "run"

        ###############
        # WAIT CHECKS #
        ###############
        if mandatory_input_sockets == received_input_sockets and skipped_optional_input_sockets.issubset(
            optional_input_sockets
        ):
            # We received all of the inputs we need, but some optional inputs have not been run or skipped yet
            logger.debug(
                "Component '%s' is waiting. All mandatory inputs received, some optional are not skipped: %s",
                name,
                optional_input_sockets - skipped_optional_input_sockets,
            )
            return "wait"

        if any(self.graph.nodes[n]["visits"] == 0 for n in input_components.keys()):
            # Some upstream component that must send input to the current component has yet to run.
            logger.debug(
                "Component '%s' is waiting. Missing inputs: %s",
                name,
                set(input_components.values()),
            )
            return "wait"

        ###############
        # SKIP CHECKS #
        ###############
        if self.graph.nodes[name]["visits"] == 0:
            # It's the first time visiting this component, if it can't run nor wait
            # it's fine skipping it at this point.
            logger.debug("Component '%s' is skipped. It can't run nor wait.", name)
            return "skip"

        #################
        # REMOVE CHECKS #
        #################
        if self.graph.nodes[name]["visits"] > 0:
            # This component has already been visited at least once. If it can't run nor wait
            # there is no reason to skip it again. So we it must be removed.
            logger.debug("Component '%s' is removed. It can't run, wait or skip.", name)
            return "remove"

        # Can't determine action to take
        raise PipelineRuntimeError(
            f"Can't determine Component '{name}' action. "
            f"Mandatory input sockets: {mandatory_input_sockets}, "
            f"optional input sockets: {optional_input_sockets}, "
            f"received input: {list(inputs.keys())}, "
            f"input components: {list(input_components.keys())}, "
            f"skipped components: {skipped_components}, "
            f"skipped optional inputs: {skipped_optional_input_sockets}."
            f"This is likely a Canals bug. Please open an issue at https://github.com/deepset-ai/canals.",
        )

    def _skip_downstream_unvisited_nodes(self, component_name: str, inputs_buffer: OrderedDict) -> OrderedDict:
        """
        When a component is skipped, put all downstream nodes in the inputs buffer too: the might be skipped too,
        unless they are merge nodes. They will be evaluated later by the pipeline execution loop.
        """
        downstream_nodes = [e[1] for e in self.graph.out_edges(component_name)]
        for downstream_node in downstream_nodes:
            if downstream_node in inputs_buffer:
                continue
            if self.graph.nodes[downstream_node]["visits"] == 0:
                # Skip downstream nodes only if they never been visited
                inputs_buffer[downstream_node] = {}
        return inputs_buffer

    def _run_component(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Once we're confident this component is ready to run, run it and collect the output.
        """
        self.graph.nodes[name]["visits"] += 1
        instance = self.graph.nodes[name]["instance"]
        try:
            logger.info("* Running %s (visits: %s)", name, self.graph.nodes[name]["visits"])
            logger.debug("   '%s' inputs: %s", name, inputs)

            outputs = instance.run(**inputs)

            # Unwrap the output
            logger.debug("   '%s' outputs: %s\n", name, outputs)

            # Make sure the component returned a dictionary
            if not isinstance(outputs, dict):
                raise PipelineRuntimeError(
                    f"Component '{name}' returned a value of type "
                    f"'{getattr(type(outputs), '__name__', str(type(outputs)))}' instead of a dict. "
                    "Components must always return dictionaries: check the the documentation."
                )

        except Exception as e:
            raise PipelineRuntimeError(
                f"{name} raised '{e.__class__.__name__}: {e}' \nInputs: {inputs}\n\n"
                "See the stacktrace above for more information."
            ) from e

        return outputs

    def _route_output(
        self,
        node_name: str,
        node_results: Dict[str, Any],
        inputs_buffer: OrderedDict,
    ) -> OrderedDict:
        """
        Distrubute the outputs of the component into the input buffer of downstream components.

        Returns the updated inputs buffer.
        """
        # This is not a terminal node: find out where the output goes, to which nodes and along which edge
        is_decision_node_for_loop = (
            any(networkx.has_path(self.graph, edge[1], node_name) for edge in self.graph.out_edges(node_name))
            and len(self.graph.out_edges(node_name)) > 1
        )
        for edge_data in self.graph.out_edges(node_name, data=True):
            to_socket = edge_data[2]["to_socket"]
            from_socket = edge_data[2]["from_socket"]
            target_node = edge_data[1]

            # If this is a decision node and a loop is involved, we add to the input buffer only the nodes
            # that received their expected output and we leave the others out of the queue.
            if is_decision_node_for_loop and node_results.get(from_socket.name, None) is None:
                if networkx.has_path(self.graph, target_node, node_name):
                    # In case we're choosing to leave a loop, do not put the loop's node in the buffer.
                    logger.debug(
                        "Not adding '%s' to the inputs buffer: we're leaving the loop.",
                        target_node,
                    )
                else:
                    # In case we're choosing to stay in a loop, do not put the external node in the buffer.
                    logger.debug(
                        "Not adding '%s' to the inputs buffer: we're staying in the loop.",
                        target_node,
                    )
            else:
                # In all other cases, populate the inputs buffer for all downstream nodes.

                # Create the buffer for the downstream node if it's not yet there.
                if target_node not in inputs_buffer:
                    inputs_buffer[target_node] = {}

                # Skip Edges that did not receive any input.
                value_to_route = node_results.get(from_socket.name)
                if value_to_route is None:
                    continue

                # If the socket was marked as variadic, pile up inputs in a list
                if to_socket.is_variadic:
                    inputs_buffer[target_node].setdefault(to_socket.name, []).append(value_to_route)
                # Non-variadic input: just store the value
                else:
                    inputs_buffer[target_node][to_socket.name] = value_to_route

        return inputs_buffer

    def _prepare_inputs_buffer(self, data: Dict[str, Any]) -> OrderedDict:
        """
        Prepare the inputs buffer based on the parameters that were
        passed to run()
        """
        inputs_buffer: OrderedDict = OrderedDict()
        for node_name, input_data in data.items():
            for socket_name, value in input_data.items():
                if value is None:
                    continue
                if self.graph.nodes[node_name]["input_sockets"][socket_name].is_variadic:
                    value = [value]
                inputs_buffer.setdefault(node_name, {})[socket_name] = value
        return inputs_buffer
