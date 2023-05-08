from typing import Optional, Any, Dict, List, Tuple, Literal

import datetime
import logging
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

import networkx

from canals.errors import PipelineConnectError, PipelineMaxLoops, PipelineRuntimeError, PipelineValidationError
from canals.draw import draw, RenderingEngines
from canals.pipeline._utils import (
    InputSocket,
    OutputSocket,
    find_input_sockets,
    find_output_sockets,
    find_unambiguous_connection,
    parse_connection_name,
    validate_pipeline_input,
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
        self.graph = networkx.MultiDiGraph()
        self.debug: Dict[int, Dict[str, Any]] = {}

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
            variadic_input=any(e.variadic for e in input_sockets.values()),
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
        if not from_socket.type == to_socket.type:
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_socket.name}' with '{to_node}.{to_socket.name}': "
                f"their declared input and output types do not match.\n"
                f" - {from_node}.{from_socket.name}: {from_socket.type.__name__}\n"
                f" - {to_node}.{to_socket.name}: {to_socket.type.__name__}\n"
            )

        # Make sure the receiving socket is not taken - sending sockets can be connected as many times as needed,
        # so they don't need this check
        if to_socket.taken_by:
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_socket.name}' with '{to_node}.{to_socket.name}': "
                f"{to_node}.{to_socket.name} is already connected to {to_socket.taken_by}.\n"
            )

        # Create the connection
        logger.debug("Connecting '%s.%s' to '%s.%s'", from_node, from_socket.name, to_node, to_socket.name)
        edge_key = f"{from_socket.name}/{to_socket.name}"
        self.graph.add_edge(from_node, to_node, key=edge_key, from_socket=from_socket, to_socket=to_socket)

        # Mark the receiving socket as taken (unless is variadic - variadic sockets are never "taken")
        if not to_socket.variadic:
            to_socket.taken_by = from_node

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

    def run(self, data: Dict[str, Dict[str, Any]], debug: bool = False) -> Dict[str, Any]:
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

        data = validate_pipeline_input(self.graph, inputs_values=data)
        self._clear_visits_count()
        self.warm_up()

        logger.info("Pipeline execution started.")
        inputs_buffer = OrderedDict(data)
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
                self.debug[step] = {
                    "time": datetime.datetime.now(),
                    "inputs_buffer": list(inputs_buffer.items()),
                    "pipeline_output": pipeline_output,
                }
            logger.debug("> Queue at step %s: %s", step, {k: list(v.keys()) for k, v in inputs_buffer.items()})

            component, inputs = inputs_buffer.popitem(last=False)  # FIFO

            # if debug:
            #     draw(deepcopy(self.graph), engine="graphviz", path=f"debug/step_{current_step}.jpg", running=component, queued=inputs_buffer.keys())

            # **** IS IT MY TURN YET? ****
            # Check if the node should be run or not
            ready_to_run = self._ready_to_run(name=component, inputs=inputs, inputs_buffer=inputs_buffer)

            # This component is missing data: let's put it back in the queue and wait.
            if ready_to_run == "wait":
                inputs_buffer[component] = inputs
                continue

            # This component did not receive the input it needs: it must be on a skipped branch. Let's not run it.
            if ready_to_run == "skip":
                self.graph.nodes[component]["visits"] += 1
                inputs_buffer = self._skip_downstream_nodes(component=component, inputs_buffer=inputs_buffer)
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
            self.debug[step + 1] = {
                "time": datetime.datetime.now(),
                "inputs_buffer": list(inputs_buffer.items()),
                "pipeline_output": pipeline_output,
            }
            pipeline_output["_debug"] = self.debug  # type: ignore

        logger.info("Pipeline executed successfully.")
        return pipeline_output

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
        self, name: str, inputs: Dict[str, Any], inputs_buffer: OrderedDict
    ) -> Literal["run", "wait", "skip"]:
        """
        Verify whether a component is ready to run.

        Returns 'run', 'wait' or 'skip' depending on how the node should be treated and the log message explaining
        the decision.
        """
        # Make sure it didn't run too many times already
        self._check_max_loops(name)

        # List all the component/socket pairs the current component should be waiting for.
        expected_inputs = self._connections_to_wait_for(name=name)

        # Check if the expected inputs were all received
        if self._check_received_vs_expected_inputs(name=name, inputs=inputs, expected_inputs=expected_inputs):
            return "run"

        # This node is missing some inputs. Did all the upstream nodes run?
        nodes_to_wait_for, _ = zip(*expected_inputs) if expected_inputs else ([], [])

        # Some node upstream didn't run yet, so we should wait for them.
        if not self._all_nodes_to_wait_for_run(nodes_to_wait_for=nodes_to_wait_for):

            if not inputs_buffer:
                # What if there are no components to wait for?
                raise PipelineRuntimeError(
                    f"'{name}' is stuck waiting for input, but there are no other components to run. "
                    "This is likely a Canals bug. Open an issue at https://github.com/deepset-ai/canals."
                )

            logger.debug(
                "Putting '%s' back in the queue, some inputs are missing (inputs to wait for: %s, inputs_received: %s)",
                name,
                [f"{node}.{socket}" for node, socket in expected_inputs],
                list(inputs.keys()),
            )
            return "wait"

        # All upstream nodes run, so it **must** be our turn.
        # However we're missing data, so this branch probably is being skipped.
        if inputs and self.graph.nodes[name]["variadic_input"]:
            logger.debug(
                "Running '%s', even though some upstream component did not produced output. "
                "(upstream components: %s, expected inputs: %s, n. of inputs received %s)",
                name,
                list(nodes_to_wait_for),
                [f"{node}.{socket}" for node, socket in expected_inputs],
                len(list(inputs.values())[0]) if inputs else 0,
            )
            return "run"

        logger.debug(
            "Skipping '%s', upstream component didn't produce output "
            "(upstream components: %s, expected inputs: %s, inputs received: %s)",
            name,
            list(nodes_to_wait_for),
            [f"{node}.{socket}" for node, socket in expected_inputs],
            list(inputs.keys()),
        )
        return "skip"

    def _check_received_vs_expected_inputs(
        self, name: str, inputs: Dict[str, Any], expected_inputs: Tuple[str, str]
    ) -> bool:
        """
        Check if all the inputs the component is expecting have been received.

        Returns True if all the necessary inputs are received, False otherwise, and a message with the decision.
        """
        # Variadic nodes expect a single list regardless of how many incoming connections they have,
        # but the length of the list should match the length of incoming connections.
        if self.graph.nodes[name]["variadic_input"]:

            # Variadic nodes need at least two values
            if not inputs or len(inputs) < 2:
                return False

            if len(list(inputs.values())[0]) == len(expected_inputs):
                logger.debug(
                    "Component '%s' is ready to run: all connected inputs were received "
                    "(expecting %s, received %s values).",
                    name,
                    len(expected_inputs),
                    len(list(inputs.values())[0]),
                )
                return True
        else:
            # No input sockets are connected: this is an input node and should be always ready to run.
            if not expected_inputs:
                logger.debug("Component '%s' is ready to run: it's a starting node.", name)
                return True

            # Otherwise, just make sure there is one input key for each expected input key
            _, expected_input_names = zip(*expected_inputs)
            if set(expected_input_names).issubset(set(inputs.keys())):
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
            if not (networkx.has_path(self.graph, name, from_node) and self.graph.nodes[name]["variadic_input"])
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
        unless they are merge/variadic nodes. They will be evaluated later by the pipeline execution loop.
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

            # If the node is variadic, unpack the input
            if self.graph.nodes[name]["variadic_input"]:
                inputs = list(inputs.values())[0]
                output_dataclass = instance.run(*inputs)

            # Otherwise pass the inputs as kwargs after adding the component's own defaults to them
            else:
                inputs = {**instance.defaults, **inputs}
                output_dataclass = instance.run(**inputs)

            # Unwrap the output
            output = output_dataclass.__dict__
            logger.debug("   '%s' outputs: %s\n", name, output)

        except Exception as e:
            raise PipelineRuntimeError(
                f"{name} raised '{e.__class__.__name__}: {e}' \nInputs: {inputs}\n\n"
                "See the stacktrace above for more information."
            ) from e

        return output

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
            if is_decision_node_for_loop and node_results[from_socket.name] is None:
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
                if node_results.get(from_socket.name, None):
                    if to_socket.variadic:
                        if not to_socket.name in inputs_buffer[target_node].keys():
                            inputs_buffer[target_node][to_socket.name] = []
                        inputs_buffer[target_node][to_socket.name].append(node_results[from_socket.name])
                    else:
                        inputs_buffer[target_node][to_socket.name] = node_results[from_socket.name]

        return inputs_buffer
