# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any, Dict, List, Literal, Union, Set

import os
import json
import datetime
import logging
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from dataclasses import fields

import networkx

from canals.errors import PipelineConnectError, PipelineMaxLoops, PipelineRuntimeError, PipelineValidationError
from canals.draw import draw, convert_for_debug, RenderingEngines
from canals.pipeline.sockets import InputSocket, OutputSocket, find_input_sockets, find_output_sockets
from canals.pipeline.validation import validate_pipeline_input
from canals.pipeline.connections import parse_connection_name, find_unambiguous_connection


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

    def _comparable_nodes_list(self, graph: networkx.MultiDiGraph) -> List[Dict[str, Any]]:
        """
        Replaces instances of nodes with their class name and defaults list in order to make sure they're comparable.
        """
        nodes = []
        for node in graph.nodes:
            comparable_node = graph.nodes[node]
            if hasattr(comparable_node, "defaults"):
                comparable_node["defaults"] = comparable_node["instance"].defaults
            comparable_node["instance"] = comparable_node["instance"].__class__
            nodes.append(comparable_node)
        nodes.sort()
        return nodes

    def add_component(self, name: str, instance: Any) -> None:
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
        if not hasattr(instance, "__canals_component__"):
            raise PipelineValidationError(
                f"'{type(instance)}' doesn't seem to be a component. Is this class decorated with @component?"
            )

        # Find inputs and outputs
        input_sockets = find_input_sockets(instance)
        output_sockets = find_output_sockets(instance)

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
        from_node, from_socket_name = parse_connection_name(connect_from)
        to_node, to_socket_name = parse_connection_name(connect_to)

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
                    + ", ".join([f"{name} (type {socket.type.__name__})" for name, socket in from_sockets.items()])
                )
        if to_socket_name:
            to_socket = to_sockets.get(to_socket_name, None)
            if not to_socket:
                raise PipelineConnectError(
                    f"'{to_node}.{to_socket_name} does not exist. "
                    f"Input connections of {to_node} are: "
                    + ", ".join([f"{name} (type {socket.type.__name__})" for name, socket in to_sockets.items()])
                )

        # If either one of the two sockets is not specified, look for an unambiguous connection
        # Note that if there is more than one possible connection but two sockets match by name, they're paired.
        if not to_socket_name or not from_socket_name:
            from_sockets = [from_socket] if from_socket_name else from_sockets.values()
            to_sockets = [to_socket] if to_socket_name else to_sockets.values()
            from_socket, to_socket = find_unambiguous_connection(
                from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
            )

        # Connect the components on these sockets
        self._direct_connect(from_node=from_node, from_socket=from_socket, to_node=to_node, to_socket=to_socket)

    def _direct_connect(self, from_node: str, from_socket: OutputSocket, to_node: str, to_socket: InputSocket) -> None:
        """
        Directly connect socket to socket.
        """
        # Check that the types match. We need type equality: subclass relationships are not accepted, just like
        # Optionals, Unions, and similar "aggregate" types. See https://github.com/python/typing/issues/570
        if to_socket.type is not Any and not from_socket.type == to_socket.type:
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_socket.name}' with '{to_node}.{to_socket.name}': "
                f"their declared input and output types do not match.\n"
                f" - {from_node}.{from_socket.name}: {from_socket.type.__name__}\n"
                f" - {to_node}.{to_socket.name}: {to_socket.type.__name__}\n"
            )

        # Make sure the receiving socket isn't already connected - sending sockets can be connected as many times as needed,
        # so they don't need this check
        if to_socket.sender:
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_socket.name}' with '{to_node}.{to_socket.name}': "
                f"{to_node}.{to_socket.name} is already connected to {to_socket.sender}.\n"
            )

        # Create the connection
        logger.debug("Connecting '%s.%s' to '%s.%s'", from_node, from_socket.name, to_node, to_socket.name)
        edge_key = f"{from_socket.name}/{to_socket.name}"
        self.graph.add_edge(from_node, to_node, key=edge_key, from_socket=from_socket, to_socket=to_socket)

        # Stores the name of the node that will send its output to this socket
        to_socket.sender = from_node

    def get_component(self, name: str) -> object:
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

    def draw(self, path: Path, engine: RenderingEngines = "mermaid-img") -> None:
        """
        Draws the pipeline. Requires either `graphviz` as a system dependency, or an internet connection for Mermaid.
        Run `pip install canals[graphviz]` or `pip install canals[mermaid]` to install missing dependencies.

        Args:
            path: where to save the diagram.
            engine: which format to save the graph as. Accepts 'graphviz', 'mermaid-text', 'mermaid-img'.
                Default is 'mermaid-img'.

        Returns:
            None

        Raises:
            ImportError: if `engine='graphviz'` and `pygraphviz` is not installed.
            HTTPConnectionError: (and similar) if the internet connection is down or other connection issues.
        """
        draw(graph=deepcopy(self.graph), path=path, engine=engine)

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
        self._clear_visits_count()
        self.warm_up()

        logger.info("Pipeline execution started.")
        inputs_buffer = OrderedDict(
            {
                node: {key: value for key, value in input_data.__dict__.items() if value is not None}
                for node, input_data in data.items()
            }
        )
        skipped_nodes: Set[str] = set()
        pipeline_output: Dict[str, Dict[str, Any]] = {}

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

            component, inputs = inputs_buffer.popitem(last=False)  # FIFO

            # Make sure it didn't run too many times already
            self._check_max_loops(component)

            # if debug:
            #     draw(deepcopy(self.graph), engine="graphviz", path=f"debug/step_{current_step}.jpg", running=component, queued=inputs_buffer.keys())

            # **** IS IT MY TURN YET? ****
            # Check if the node should be run or not
            ready_to_run = self._ready_to_run(name=component, inputs=inputs, skipped_nodes=skipped_nodes)

            # This component is missing data: let's put it back in the queue and wait.
            if ready_to_run == "wait":
                if not inputs_buffer:
                    # What if there are no components to wait for?
                    raise PipelineRuntimeError(
                        f"'{component}' is stuck waiting for input, but there are no other components to run. "
                        "This is likely a Canals bug. Open an issue at https://github.com/deepset-ai/canals."
                    )

                inputs_buffer[component] = inputs
                continue

            # This component did not receive the input it needs: it must be on a skipped branch. Let's not run it.
            if ready_to_run == "skip":
                self.graph.nodes[component]["visits"] += 1
                inputs_buffer = self._skip_downstream_nodes(component=component, inputs_buffer=inputs_buffer)
                skipped_nodes.add(component)
                continue

            # **** RUN THE NODE ****
            # It is our turn! The node is ready to run and all necessary inputs are present
            output = self._run_component(name=component, inputs=inputs)

            # **** PROCESS THE OUTPUT ****
            # The node run successfully. Let's store or distribute the output it produced, if it's valid.
            if not self.graph.out_edges(component):
                # Note: if a node outputs many times (like in loops), the output will be overwritten
                pipeline_output[component] = output
            else:
                inputs_buffer = self._route_output(
                    node_results=output, node_name=component, inputs_buffer=inputs_buffer
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
        mermaid_graph = convert_for_debug(deepcopy(self.graph))
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

    def _ready_to_run(
        self, name: str, inputs: Dict[str, Any], skipped_nodes: Set[str]
    ) -> Literal["run", "wait", "skip"]:
        """
        Verify whether a component is ready to run.

        Returns 'run', 'wait' or 'skip' depending on how the node should be treated and the log message explaining
        the decision.
        """

        # List all the component/socket pairs the current component should be waiting for.
        input_components = {
            from_node: data["to_socket"].name for from_node, _, data in self.graph.in_edges(name, data=True)
        }
        input_sockets = {f.name for f in fields(self.graph.nodes[name]["instance"].__canals_input__)}
        optional_input_sockets = set(self.graph.nodes[name]["instance"].__canals_optional_inputs__)
        mandatory_input_sockets = {i for i in input_sockets if i not in optional_input_sockets}
        skipped_optional_input_sockets = {
            sockets["to_socket"].name
            for from_node, _, sockets in self.graph.in_edges(name, data=True)
            if from_node in skipped_nodes and sockets["to_socket"].name in optional_input_sockets
        }

        def _should_run():
            if mandatory_input_sockets.issubset(set(inputs.keys())) and not optional_input_sockets:
                # We received all the inputs we need and have no optional inputs
                return True
            if (
                mandatory_input_sockets.issubset(set(inputs.keys()))
                and skipped_optional_input_sockets == optional_input_sockets
            ):
                # We received all the inputs we need and all optional inputs are skipped
                return True
            if (
                mandatory_input_sockets.issubset(set(inputs.keys()))
                and input_sockets == set(inputs.keys()) | mandatory_input_sockets | skipped_optional_input_sockets
            ):
                # We received all the inputs we need and some optionals, some other optionals instead are skipped
                return True
            if not input_components:
                # Nothing to wait for, this must be the first input component in the Pipeline
                logger.debug("Component '%s' is ready to run: it's a starting node.", name)
                return True
            if set(input_components.values()).issubset(set(inputs.keys())):
                # Otherwise, just make sure there is one input key for each expected input key
                logger.debug("Component '%s' is ready to run: all expected inputs were received.", name)
                return True

            return False

        def _should_wait():
            if mandatory_input_sockets.issubset(set(inputs.keys())) and skipped_optional_input_sockets.issubset(
                optional_input_sockets
            ):
                # We received all the inputs we need and some optional inputs have not been skipped
                return True
            if not all(
                self.graph.nodes[node_to_wait_for]["visits"] > 0 for node_to_wait_for in input_components.keys()
            ):
                logger.debug(
                    "Putting '%s' back in the queue, some inputs are missing (inputs to wait for: %s, inputs_received: %s)",
                    name,
                    [f"{node}.{socket}" for node, socket in input_components.items()],
                    list(inputs.keys()),
                )
                return True

            return False

        def _should_skip():
            if input_components and set(input_components.keys()).issubset(skipped_nodes):
                # All mandatory inputs are skipped, skipping this too
                return True
            return False

        if _should_run():
            return "run"

        if _should_wait():
            return "wait"

        if _should_skip():
            return "skip"

        logger.debug(
            "Skipping '%s', upstream component didn't produce output "
            "(upstream components: %s, expected inputs: %s, inputs received: %s)",
            name,
            list(input_components.keys()),
            [f"{node}.{socket}" for node, socket in input_components.items()],
            list(inputs.keys()),
        )
        return "skip"

    def _check_received_vs_expected_inputs(
        self, name: str, inputs: Dict[str, Any], expected_inputs: Dict[str, str]
    ) -> bool:
        """
        Check if all the inputs the component is expecting have been received.

        Returns True if all the necessary inputs are received, False otherwise, and a message with the decision.
        """
        # No input sockets are connected: this is an input node and should be always ready to run.
        if not expected_inputs:
            logger.debug("Component '%s' is ready to run: it's a starting node.", name)
            return True

        # Otherwise, just make sure there is one input key for each expected input key
        if set(expected_inputs.values()).issubset(set(inputs.keys())):
            logger.debug("Component '%s' is ready to run: all expected inputs were received.", name)
            return True

        return False

    def _connections_to_wait_for(self, name: str):
        """
        Return all the component/socket pairs this component is waiting for.
        """
        # We should be wait on all edges except for the downstream ones, to support loops.
        # This downstream check is enabled only for nodes taking more than one input
        # (the "entrance" of the loop).
        data_to_wait_for = [
            (from_node, data["to_socket"].name)
            for from_node, _, data in self.graph.in_edges(name, data=True)
            # ... if there's a path in the graph leading back from the current node to the
            # input node, # and only in case this node accepts multiple inputs.
            if networkx.has_path(self.graph, from_node, name)
            # and data["to_socket"].name not in self.graph.nodes[name]["instance"].__canals_optional_inputs__
        ]
        return data_to_wait_for

    def _all_nodes_to_wait_for_run(self, nodes_to_wait_for: List[str]) -> bool:
        """
        Check if all the nodes this component is waiting for has run or not.

        FIXME: checking for `visits>0` might not be enough for all loops.
        """
        return all(self.graph.nodes[node_to_wait_for]["visits"] > 0 for node_to_wait_for in nodes_to_wait_for)

    def _skip_downstream_nodes(self, component: str, inputs_buffer: OrderedDict) -> OrderedDict:
        """
        When a component is skipped, put all downstream nodes in the inputs buffer too: the might be skipped too,
        unless they are merge nodes. They will be evaluated later by the pipeline execution loop.
        """
        downstream_nodes = [e[1] for e in self.graph.out_edges(component)]
        for downstream_node in downstream_nodes:
            if not downstream_node in inputs_buffer:
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

            # Optional fields are defaulted to None so creation of the input dataclass doesn't fail
            # cause we're missing some argument
            optionals = {field: None for field in instance.__canals_optional_inputs__}

            # Pass the inputs as kwargs after adding the component's own defaults to them
            inputs = {**optionals, **instance.defaults, **inputs}
            input_dataclass = instance.input(**inputs)

            output_dataclass = instance.run(input_dataclass)

            # Unwrap the output
            logger.debug("   '%s' outputs: %s\n", name, output_dataclass.__dict__)

        except Exception as e:
            raise PipelineRuntimeError(
                f"{name} raised '{e.__class__.__name__}: {e}' \nInputs: {inputs}\n\n"
                "See the stacktrace above for more information."
            ) from e

        return output_dataclass

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
            if is_decision_node_for_loop and getattr(node_results, from_socket.name) is None:
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
                # In all other cases, populate the inputs buffer for all downstream nodes, setting None to any
                # edge that did not receive input.
                if not target_node in inputs_buffer:
                    inputs_buffer[target_node] = {}  # Create the buffer for the downstream node if it's not there yet

                value_to_route = getattr(node_results, from_socket.name, None)
                if value_to_route:
                    inputs_buffer[target_node][to_socket.name] = value_to_route

        return inputs_buffer
