from typing import Optional, Any, Dict, List, Tuple, Literal

from pathlib import Path
import logging
from copy import deepcopy
from collections import OrderedDict

import networkx as nx

from canals.errors import PipelineConnectError, PipelineMaxLoops, PipelineRuntimeError, PipelineValidationError
from canals.pipeline.draw import render_graphviz, render_mermaid
from canals.pipeline._utils import (
    InputSocket,
    OutputSocket,
    find_sockets,
    find_unambiguous_connection,
    is_subtype,
    parse_connection_name,
    validate_pipeline,
    find_pipeline_inputs,
    find_pipeline_outputs,
)


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
    ):
        """
        Creates the Pipeline.

        Args:
            metadata: arbitrary dictionary to store metadata about this pipeline. Make sure all the values contained in
                this dictionary can be serialized and deserialized if you wish to save this pipeline to file with
                `save_pipelines()/load_pipelines()`.
            max_loops_allowed: how many times the pipeline can run the same node before throwing an exception.
        """
        self.metadata = metadata or {}
        self.max_loops_allowed = max_loops_allowed
        self.graph = nx.MultiDiGraph()

    def __eq__(self, other) -> bool:
        # Equal pipelines share all nodes and metadata instances.
        if not isinstance(other, type(self)):
            return False
        return (
            self.metadata == other.metadata
            and self.max_loops_allowed == other.max_loops_allowed
            and self.graph == other.graph
        )

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
            ValueError: if a component with the same name already exists or `parameters` is not a dictionary
            PipelineValidationError: if the given instance is not a Canals component
        """
        # Component names are unique
        if name in self.graph.nodes:
            raise ValueError(f"Component named '{name}' already exists: choose another name.")

        # Component instances must be components
        if not hasattr(instance, "__canals_component__"):
            raise PipelineValidationError(
                f"'{type(instance)}' doesn't seem to be a component. Is this class decorated with @component?"
            )

        # Find expected inputs and outputs
        input_sockets, output_sockets = find_sockets(instance)

        # Add component to the graph, disconnected
        logger.debug("Adding component '%s' (%s)", name, instance)
        self.graph.add_node(
            name,
            instance=instance,
            input_sockets=input_sockets,
            variadic_input=any(e.variadic for e in input_sockets.values()),
            output_sockets=output_sockets,
            visits=0,
        )

    def _connect(self, from_node: str, from_socket: OutputSocket, to_node: str, to_socket: InputSocket) -> None:
        """
        Directly connect socket to socket.
        """
        if not is_subtype(from_socket.type, to_socket.type):
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_socket.name}' with '{to_node}.{to_socket.name}': "
                f"their declared input and output types do not match.\n"
                f" - {from_node}.{from_socket.name}: {from_socket.type.__name__}\n"
                f" - {to_node}.{to_socket.name}: {to_socket.type.__name__}\n"
            )
        if to_socket.taken_by:
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_socket.name}' with '{to_node}.{to_socket.name}': "
                f"{to_node}.{to_socket.name} is already connected to {to_socket.taken_by}.\n"
            )

        # Create the connection
        logger.debug("Connecting '%s.%s' to '%s.%s'", from_node, from_socket.name, to_node, to_socket.name)
        self.graph.add_edge(
            from_node,
            to_node,
            key=f"{from_socket.name}/{to_socket.name}",
            from_socket=from_socket,
            to_socket=to_socket,
        )
        # Variadic sockets are never fully taken
        if not to_socket.variadic:
            to_socket.taken_by = from_node

    def connect(self, connect_from: str, connect_to: str) -> None:
        """
        Connect components together. All components to connect must exist in the pipeline.
        If connecting to an component that has several output connections, specify its name with
        'component_name.connections_name'.

        Args:
            connect_from: the component that deliver the value. This can be either a single component name or can be
                in the format `component_name.connection_name` if the component has multiple outputs.
            connect_to: the component that receives the value. This can be either a single component name or can be
                in the format `component_name.connection_name` if the component has multiple inputs.

        Returns:
            None

        Raises:
            PipelineConnectError: if the two components cannot be connected (for example if one of the components is
                not present in the pipeline, or the connections don't match by type, and so on).
        """
        # Edges may be named explicitly by passing 'node_name.edge_name' to connect().
        from_node_name, from_socket_name = parse_connection_name(connect_from)
        to_node_name, to_socket_name = parse_connection_name(connect_to)

        # Get the nodes data. This method also ensures that the nodes names are present the pipeline
        from_sockets = self._get_component_data(from_node_name)["output_sockets"]
        to_sockets = self._get_component_data(to_node_name)["input_sockets"]

        # If the names of both edges are given, get the sockets directly
        if from_socket_name:
            from_socket = from_sockets.get(from_socket_name, None)
            if not from_socket:
                raise PipelineConnectError(
                    f"'{from_node_name}.{from_socket_name} does not exist. "
                    f"Output connections of {from_node_name} are: "
                    + ", ".join([f"{name} (type {socket.type.__name__})" for name, socket in from_sockets.items()])
                )
        if to_socket_name:
            to_socket = to_sockets.get(to_socket_name, None)
            if not to_socket:
                raise PipelineConnectError(
                    f"'{to_node_name}.{to_socket_name} does not exist. "
                    f"Input connections of {to_node_name} are: "
                    ", ".join([f"{name} (type {socket.type.__name__})" for name, socket in to_sockets.items()])
                )
        if not to_socket_name or not from_socket_name:
            # Find one pair of sockets that can be connected
            from_sockets = [from_socket] if from_socket_name else from_sockets.values()
            to_sockets = [to_socket] if to_socket_name else to_sockets.values()
            from_socket, to_socket = find_unambiguous_connection(
                from_node=from_node_name, from_sockets=from_sockets, to_node=to_node_name, to_sockets=to_sockets
            )
        # Connect the components on these sockets
        self._connect(from_node=from_node_name, from_socket=from_socket, to_node=to_node_name, to_socket=to_socket)

    def _get_component_data(self, name: str) -> Dict[str, Any]:
        """
        Returns all the data associated with a component.
        """
        candidates = [node for node in self.graph.nodes if node == name]
        if not candidates:
            raise ValueError(f"Component named {name} not found in the pipeline.")
        return self.graph.nodes[candidates[0]]

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
        return self._get_component_data(name)["instance"]

    def draw(self, path: Path, engine: Literal["graphviz", "mermaid"] = "mermaid") -> None:
        """
        Draws the pipeline. Requires `pygraphviz`.
        Run `pip install canals[draw]` to install missing dependencies.

        Args:
            path: where to save the drawing.

        Returns:
            None

        Raises:
            ImportError: if pygraphviz is not installed.
        """
        graph = deepcopy(self.graph)

        input_nodes = find_pipeline_inputs(graph)
        output_nodes = find_pipeline_outputs(graph)

        # Label the edges
        for inp, outp, key, data in graph.edges(keys=True, data=True):
            data["label"] = f"{data['from_socket'].name} -> {data['to_socket'].name}"
            graph.add_edge(inp, outp, key=key, **data)

        # Draw the input
        graph.add_node("input")
        for node, sockets in input_nodes.items():
            for socket in sockets:
                graph.add_edge("input", node, label=socket.name)

        # Draw the output
        graph.add_node("output", shape="plain")
        for node, sockets in output_nodes.items():
            for socket in sockets:
                graph.add_edge(node, "output", label=socket.name)

        if engine == "graphviz":
            render_graphviz(graph=graph, path=path)
        elif engine == "mermaid":
            render_mermaid(graph=graph, path=path)
        else:
            raise ValueError(f"Unknown rendering engine '{engine}'. Choose one from: 'graphviz', 'mermaid'.")

        logger.debug("Pipeline diagram saved at %s", path)

    def warm_up(self):
        """
        Make sure all nodes are warm.

        It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
        without re-initializing everything.
        """
        for node in self.graph.nodes:
            if hasattr(self.graph.nodes[node]["instance"], "warm_up"):
                self.graph.nodes[node]["instance"].warm_up()

    def run(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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
        validate_pipeline(self.graph, inputs_values=data)
        self._clear_visits_count()
        self.warm_up()

        # **** The Pipeline.run() algorithm ****
        #
        # Nodes are run as soon as an input for them appears in the inputs buffer.
        # When there's more than a node at once  in the buffer (which means some
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
        # Data access:
        # - Name of the node       # self.graph.nodes  (List[str])
        # - Node instance          # self.graph.nodes[node]["instance"]
        # - Input nodes            # [e[0] for e in self.graph.in_edges(node)]
        # - Output nodes           # [e[1] for e in self.graph.out_edges(node)]
        # - Output edges           # [e[2]["label"] for e in self.graph.out_edges(node, data=True)]
        #
        logger.info("Pipeline execution started.")

        inputs_buffer = OrderedDict(data)

        # *** PIPELINE EXECUTION LOOP ***
        # We select the nodes to run by checking which keys are set in the
        # inputs buffer. If the key exists, the node might be ready to run.
        pipeline_results: Dict[str, List[Dict[str, Any]]] = {}
        while inputs_buffer:
            logger.debug("> Current component queue: %s", inputs_buffer.keys())
            logger.debug("> Current inputs buffer: %s", inputs_buffer)

            node_name, node_inputs = inputs_buffer.popitem(last=False)  # FIFO

            ready_to_run, inputs_buffer = self._ready_to_run(
                component_name=node_name, component_inputs=node_inputs, inputs_buffer=inputs_buffer
            )
            if not ready_to_run:
                continue

            # **** RUN THE NODE ****
            # It is our turn! The node is ready to run and all inputs are ready
            #
            # Let's raise the visits count
            self.graph.nodes[node_name]["visits"] += 1

            # Get the node
            node_instance = self.graph.nodes[node_name]["instance"]

            # Call the node
            try:
                logger.info("* Running %s (visits: %s)", node_name, self.graph.nodes[node_name]["visits"])
                logger.debug("   '%s' inputs: %s", node_name, node_inputs)

                # Check if any param is variadic and unpack it
                # Note that components either accept one single variadic positional or named kwargs! Not both.
                if self._get_component_data(node_name)["variadic_input"]:
                    node_inputs = list(node_inputs.items())[0][1]
                    node_output = node_instance.run(*node_inputs)
                else:
                    node_inputs = {**node_instance.defaults, **node_inputs}
                    node_output = node_instance.run(**node_inputs)

                node_results = node_output.__dict__
                logger.debug("   '%s' outputs: %s\n", node_name, node_results)
            except Exception as e:
                raise PipelineRuntimeError(
                    f"{node_name} raised '{e.__class__.__name__}: {e}' \ninputs={node_inputs}\n\n"
                    "See the stacktrace above for more information."
                ) from e

            # **** PROCESS THE OUTPUT ****
            # The node run successfully. Let's store or distribute the output it produced, if it's valid.
            #
            # Process the output of the node
            if not self.graph.out_edges(node_name):
                # If there are no output edges, the output of this node is the output of the pipeline:
                # store it in pipeline_results.
                if not node_name in pipeline_results.keys():
                    pipeline_results[node_name] = []
                # If a node outputs many times (like in loops), the output will be overwritten
                pipeline_results[node_name] = node_results
            else:
                inputs_buffer = self._route_output(
                    node_results=node_results, node_name=node_name, inputs_buffer=inputs_buffer
                )

        logger.info("Pipeline executed successfully.")

        # # Simplify output for single edge, single output pipelines
        # pipeline_results = self._unwrap_results(pipeline_results)
        return pipeline_results

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
        self, component_name: str, component_inputs: Dict[str, Any], inputs_buffer: OrderedDict
    ) -> Tuple[bool, OrderedDict]:
        """
        Verify whether a component is ready to run.

        Returns true if the component should run, false otherwise, and the updated inputs buffer.
        """
        # Make sure it didn't run too many times already
        self._check_max_loops(component_name)

        # List all the inputs the current node should be waiting for.
        inputs_received = tuple(component_inputs.keys())

        # We should be wait on all edges except for the downstream ones, to support loops.
        # This downstream check is enabled only for nodes taking more than one input
        # (the "entrance" of the loop).
        is_merge_node = len(self.graph.in_edges(component_name)) != 1

        data_to_wait_for = [
            (from_node, data["to_socket"].name)
            for from_node, _, data in self.graph.in_edges(component_name, data=True)
            # if there's a path in the graph leading back from the current node to the
            # input node, in case of multiple input nodes.
            if not is_merge_node or not nx.has_path(self.graph, component_name, from_node)
        ]
        nodes_to_wait_for, inputs_to_wait_for = zip(*data_to_wait_for) if data_to_wait_for else ([], [])

        if not data_to_wait_for:
            # This is an input node, so it's ready to run.
            logger.debug("'%s' is an input component and it's ready to run.", component_name)
            return (True, inputs_buffer)

        # Do we have all the inputs we expect?
        if self.graph.nodes[component_name]["variadic_input"]:
            # Here we're assuming the variadic nodes take only one argument!
            if inputs_received and len(component_inputs[inputs_received[0]]) == len(inputs_to_wait_for):
                return (True, inputs_buffer)

        elif set(inputs_to_wait_for).issubset(set(inputs_received)):
            return (True, inputs_buffer)

        # This node is missing some inputs.
        #
        # Did all the upstream nodes run?
        if not all(self.graph.nodes[node_to_wait_for]["visits"] > 0 for node_to_wait_for in nodes_to_wait_for):

            if not inputs_buffer:
                # What if there are no components to wait for, though?
                raise PipelineRuntimeError(
                    f"'{component_name}' is stuck waiting for input, but there are no other components to run. "
                    "This is likely a Canals bug. Open an issue at https://github.com/deepset-ai/canals."
                )

            # Some node upstream didn't run yet, so we should wait for them.
            logger.debug(
                "Putting '%s' back in the queue, some inputs are missing "
                "(inputs to wait for: %s, inputs_received: %s)",
                component_name,
                inputs_to_wait_for,
                inputs_received,
            )
            # Put back the node in the inputs buffer at the back...
            inputs_buffer[component_name] = component_inputs
            # ... and do not run this node (yet)
            return (False, inputs_buffer)

        # All upstream nodes run, so it **must** be our turn.
        # However, we're missing data, so this branch probably is being skipped.
        # Let's skip this node and add all downstream nodes to the queue with an equally empty buffer, so they will
        # be skipped in turn, unless they're variadic merge nodes.
        #
        # TODO Merge nodes that are not variadic but tolerate Nones might also exist. We should check for that.
        if self.graph.nodes[component_name]["variadic_input"]:
            logger.debug(
                "Running '%s', even though some upstream component did not produced output. "
                "(upstream components: %s, expected inputs: %s, n. inputs received %s)",
                component_name,
                nodes_to_wait_for,
                inputs_to_wait_for,
                # Here we're assuming the variadic nodes take only one argument!
                len(component_inputs[inputs_received[0]]),
            )
            return (True, inputs_buffer)

        self.graph.nodes[component_name]["visits"] += 1
        logger.debug(
            "Skipping '%s', no upstream component produced output "
            "(upstream components: %s, expected inputs: %s, inputs received %s)",
            component_name,
            nodes_to_wait_for,
            inputs_to_wait_for,
            inputs_received,
        )
        # Put all downstream nodes in the inputs buffer...
        downstream_nodes = [e[1] for e in self.graph.out_edges(component_name)]
        for downstream_node in downstream_nodes:
            if not downstream_node in inputs_buffer:
                inputs_buffer[downstream_node] = {}
        # ... and never run this node
        return (False, inputs_buffer)

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
            any(nx.has_path(self.graph, edge[1], node_name) for edge in self.graph.out_edges(node_name))
            and len(self.graph.out_edges(node_name)) > 1
        )
        for edge_data in self.graph.out_edges(node_name, data=True):
            to_socket = edge_data[2]["to_socket"]
            from_socket = edge_data[2]["from_socket"]
            target_node = edge_data[1]

            # If this is a decision node and a loop is involved, we add to the input buffer only the nodes
            # that received their expected output and we leave the others out of the queue.
            if is_decision_node_for_loop and not to_socket.name in node_results.keys():
                if nx.has_path(self.graph, target_node, node_name):
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
                if node_results.get(from_socket.name, None):
                    if to_socket.variadic:
                        if not to_socket.name in inputs_buffer[target_node].keys():
                            inputs_buffer[target_node][to_socket.name] = []
                        inputs_buffer[target_node][to_socket.name].append(node_results[from_socket.name])
                    else:
                        inputs_buffer[target_node][to_socket.name] = node_results[from_socket.name]

        return inputs_buffer
