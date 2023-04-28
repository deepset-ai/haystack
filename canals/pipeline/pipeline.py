from typing import Optional, Any, Dict, List, Tuple

from pathlib import Path
import logging
import inspect
from dataclasses import dataclass, fields
from copy import deepcopy
from collections import OrderedDict

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


logger = logging.getLogger(__name__)


PYGRAPHVIZ_IMPORTED = False
try:
    import pygraphviz  # pylint: disable=unused-import

    PYGRAPHVIZ_IMPORTED = True
except ImportError:
    logger.info(
        "Could not import `pygraphviz`. Please install via: \n"
        "pip install pygraphviz\n"
        "(You might need to run this first: apt install libgraphviz-dev graphviz )"
    )


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    pass


class PipelineConnectError(PipelineError):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelineMaxLoops(PipelineError):
    pass


def locate_pipeline_input_components(graph) -> List[str]:
    """
    Collect the components with no input connections: they receive directly the pipeline inputs.

    Args:
        graph: the pipeline graph.

    Returns:
        A list of components that should directly receive the user's inputs.
    """
    return [node for node in graph.nodes if not graph.in_edges(node)]


def locate_pipeline_output_components(graph) -> List[str]:
    """
    Collect the components with no output connections: these define the output of the pipeline.

    Args:
        graph: the pipeline graph.

    Returns:
        A list of components whose output goes back to the user.
    """
    return [node for node in graph.nodes if not graph.out_edges(node)]


@dataclass
class _Edge:
    name: str
    type: type


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
        self.graph = nx.DiGraph()

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
        run_signature = inspect.signature(instance.run)
        inputs = [
            _Edge(name=run_signature.parameters[param].name, type=run_signature.parameters[param].annotation)
            for param in run_signature.parameters
        ]
        return_annotation = run_signature.return_annotation
        outputs = [_Edge(name=field.name, type=field.type) for field in fields(return_annotation)]

        # Add component to the graph, disconnected
        logger.debug("Adding component '%s' (%s)", name, instance)
        self.graph.add_node(name, instance=instance, inputs=inputs, outputs=outputs, visits=0)

    def _parse_connection_name(self, connection: str) -> Tuple[str, Optional[str]]:
        """
        Returns node, edge pairs from a connection name
        """
        if "." in connection:
            split_str = connection.split(".", maxsplit=1)
            return (split_str[0], split_str[1])
        return connection, None

    def _taken_and_available_edges(
        self, node_name: str, node_inputs: List[_Edge]
    ) -> Tuple[List[str], List[str], List[_Edge]]:
        """
        Iterates over the edges to return which edges are taken, by whom, and which ones are available.
        """
        taken_edges_names_origins = [
            (data["name"], origin_node) for origin_node, __, data in self.graph.in_edges(node_name, data=True)
        ]
        taken_edges_names, taken_edges_origins = (
            zip(*taken_edges_names_origins) if taken_edges_names_origins else ([], [])
        )
        available_edges = [edge for edge in node_inputs if edge.name not in taken_edges_names]

        return taken_edges_names, taken_edges_origins, available_edges

    def _direct_connection(self, connect_from: str, connect_to: str) -> None:
        """
        If both edges are given, try to directly connect them.
        """
        # Edges may be named explicitly by passing 'node_name.edge_name' to connect().
        connect_from_node_name, connect_from_edge_name = self._parse_connection_name(connect_from)
        connect_to_node_name, connect_to_edge_name = self._parse_connection_name(connect_to)

        # Get the nodes data. This method also ensures that the nodes names are present the pipeline
        connect_from_node_data = self._get_component_data(connect_from_node_name)
        connect_to_node_data = self._get_component_data(connect_to_node_name)

        # Find available and taken inputs
        taken_edges_names, taken_edges_origins, _ = self._taken_and_available_edges(
            node_name=connect_to_node_name, node_inputs=connect_to_node_data["inputs"]
        )

        connect_from_edge = connect_from_node_data["outputs"].get(connect_from_edge_name)
        if not connect_from_edge:
            raise PipelineConnectError(
                f"'{connect_from_node_name}.{connect_from_edge_name} does not exist. "
                f"Output connections of {connect_from_node_name} are: "
                + ", ".join([edge.name for edge in connect_from_node_data["outputs"]])
            )

        connect_to_edge = connect_to_node_data["inputs"].get(connect_to_edge_name)
        if not connect_to_edge:
            raise PipelineConnectError(
                f"'{connect_to_node_name}.{connect_to_edge_name} does not exist. "
                f"Input connections of {connect_to_node_name} are: "
                + ", ".join([edge.name for edge in connect_to_node_data["inputs"]])
            )

        if not connect_from_edge.type == connect_to_edge.type:
            raise PipelineConnectError(
                f"Cannot connect '{connect_from_node_name}' with '{connect_to_node_name}': "
                f"their declared input and output types do not match.\n"
                f" - connect_from: {connect_from_node_name}.{connect_from_edge_name}, type '{connect_from_edge.type.__nane__}'\n"
                f" - connect_to: {connect_to_node_name}.{connect_to_edge_name}, type '{connect_to_edge.type.__nane__}'\n"
            )

        if connect_from_edge_name in taken_edges_names:
            used_by = taken_edges_origins[taken_edges_names.index(connect_from_edge_name)]
            raise PipelineConnectError(
                f"Cannot connect '{connect_from_node_name}' with '{connect_to_node_name}': "
                f"{connect_to_node_name}.{connect_from_edge_name} is already connected to {used_by}.\n"
            )

        # Create the connection
        logger.debug(
            "Connecting '%s' to '%s'",
            connect_from,
            connect_to,
        )
        self.graph.add_edge(
            connect_from_node_name,
            connect_to_node_name,
            name=connect_to_edge.name,
            origin_edge=connect_from_edge.name,
        )

    def connect(self, connect_from: str, connect_to: str) -> None:
        """
        Connect components together. All components to connect must exist in the pipeline.
        If connecting to an component that has several output connections, specify its name with
        'component_name.connections_name'.

        Args:
            connect_from: the component that deliver the values. This can be either a single component name or can be
                in the format `component_name.connection_name` if the component has multiple outputs.
            connect_to: the component that receives the values. This is always just the component name.

        Returns:
            None

        Raises:
            PipelineConnectError: if the two components cannot be connected (for example if one of the components is
                not present in the pipeline, or the connections don't match, and so on).
        """
        # Edges may be named explicitly by passing 'node_name.edge_name' to connect().
        connect_from_node_name, connect_from_edge_name = self._parse_connection_name(connect_from)
        connect_to_node_name, connect_to_edge_name = self._parse_connection_name(connect_to)

        # If we have both edges names: try to simply connect them.
        # Types must match and the input must not be connected to anything else already
        if connect_to_edge_name and connect_from_edge_name:
            self._direct_connection(connect_from=connect_from, connect_to=connect_to)

        else:
            # Get the nodes data. This method also ensures that the nodes names are present the pipeline
            connect_from_node_data = self._get_component_data(connect_from_node_name)
            connect_to_node_data = self._get_component_data(connect_to_node_name)

            # Find available and taken inputs
            taken_edges_names, taken_edges_origins, available_edges = self._taken_and_available_edges(
                node_name=connect_to_node_name, node_inputs=connect_to_node_data["inputs"]
            )

            # Otherwise, let's try to find one unambiguous connection between these nodes
            # Get all the possible connection with the available inputs
            possible_connections = [
                pair for pair in zip(connect_from_node_data["outputs"], available_edges) if pair[0].type == pair[1].type
            ]
            if not possible_connections:
                raise PipelineConnectError(
                    f"Cannot connect '{connect_from_node_name}' with '{connect_to_node_name}': "
                    + "there are no matching connections available.\n"
                    + f"'{connect_from_node_name}:\n"
                    + "\n".join(
                        [f" - {edge.name} ({edge.type.__name__})" for edge in connect_from_node_data["outputs"]]
                    )
                    + f"'{connect_to_node_name}:\n"
                    + "\n".join([f" - {edge.name} ({edge.type.__name__})" for edge in available_edges])
                    + "\n".join(
                        [
                            f" - {edge} (taken by {origin_node}) - "
                            for edge, origin_node in zip(taken_edges_names, taken_edges_origins)
                        ]
                    )
                )

            if len(possible_connections) > 1:
                # TODO allow for multiple connections at once if there is no ambiguity?
                raise PipelineConnectError(
                    f"Cannot connect '{connect_from_node_name}' with '{connect_to_node_name}': "
                    + "there are multiple connections possible, please specify one.\n"
                    + f"'{connect_from_node_name}:\n"
                    + "\n".join(
                        [f" - {edge.name} ({edge.type.__name__})" for edge in connect_from_node_data["outputs"]]
                    )
                    + f"'{connect_to_node_name}:\n"
                    + "\n".join([f" - {edge.name} ({edge.type.__name__})" for edge in available_edges])
                    + "\n".join(
                        [
                            f" - {edge} (taken by {origin_node})"
                            for edge, origin_node in zip(taken_edges_names, taken_edges_origins)
                        ]
                    )
                )

            connect_from_edge, connect_to_edge = possible_connections[0]

            # Create the connection
            logger.debug(
                "Connecting component '%s' to component '%s'",
                connect_from_node_name,
                connect_to_node_name,
            )
            self.graph.add_edge(
                connect_from_node_name,
                connect_to_node_name,
                name=connect_to_edge.name,
                origin_edge=connect_from_edge.name,
            )

    def _get_component_data(self, name: str) -> Dict[str, Any]:
        """
        Returns all the data associated with a component.
        """
        candidates = [node for node in self.graph.nodes if node == name]
        if not candidates:
            raise ValueError(f"Component named {name} not found in the pipeline.")
        return self.graph.nodes[candidates[0]]

    def get_component(self, name: str) -> Dict[str, Any]:
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

    def draw(self, path: Path) -> None:
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
        if not PYGRAPHVIZ_IMPORTED:
            raise ImportError(
                "Could not import `pygraphviz`. Please install via: \n"
                "pip install pygraphviz\n"
                "(You might need to run this first: apt install libgraphviz-dev graphviz )"
            )
        graph = deepcopy(self.graph)

        graphviz = to_agraph(graph)
        graphviz.layout("dot")
        graphviz.draw(path)
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

    def run(
        self,
        data: Dict[str, Dict[str, Any]],
        debug: bool = False,
    ) -> Dict[str, Any]:
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
        self.warm_up()
        self._clear_visits_count()
        self._validate_pipeline()

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
        inputs_buffer: OrderedDict = OrderedDict()

        # for node_name in locate_pipeline_input_components(self.graph):
        inputs_buffer = OrderedDict(data)

        # *** PIPELINE EXECUTION LOOP ***
        # We select the nodes to run by checking which keys are set in the
        # inputs buffer. If the key exists, the node might be ready to run.
        pipeline_results: Dict[str, List[Dict[str, Any]]] = {}
        while inputs_buffer:
            logger.debug("> Current component queue: %s", inputs_buffer.keys())
            logger.debug("> Current inputs buffer: %s", inputs_buffer)

            node_name, node_inputs = inputs_buffer.popitem(last=False)  # FIFO

            # Check if we looped over this node too many times
            self._check_max_loops(node_name)

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
            node_node = self.graph.nodes[node_name]["instance"]

            # Call the node
            try:
                logger.info(
                    "* Running %s (visits: %s)",
                    node_name,
                    self.graph.nodes[node_name]["visits"],
                )
                logger.debug("   '%s' inputs: %s", node_name, node_inputs)
                node_output = node_node.run(**node_inputs)
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

    def _validate_pipeline(self):
        """
        Make sure the pipeline has at least one input component and one output component.
        """
        if not locate_pipeline_input_components(self.graph):
            raise ValueError("This pipeline has no input components!")

        if not locate_pipeline_output_components(self.graph):
            raise ValueError("This pipeline has no output components!")

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
        # List all the inputs the current node should be waiting for.
        inputs_received = tuple(component_inputs.keys())

        # We should be wait on all edges except for the downstream ones, to support loops.
        # This downstream check is enabled only for nodes taking more than one input
        # (the "entrance" of the loop).
        is_merge_node = len(self.graph.in_edges(component_name)) != 1
        data_to_wait_for = [
            (e[0], e[2]["name"])  # the node and the edge name
            for e in self.graph.in_edges(component_name, data=True)  # for all input edges
            # if there's a path in the graph leading back from the current node to the
            # input node, in case of multiple input nodes.
            if not is_merge_node or not nx.has_path(self.graph, component_name, e[0])
        ]

        if not data_to_wait_for:
            # This is an input node, so it's ready to run.
            logger.debug("'%s' is an input component and it's ready to run.", component_name)
            return (True, inputs_buffer)

        # Do we have all the inputs we expect?
        nodes_to_wait_for, inputs_to_wait_for = zip(*data_to_wait_for)
        if sorted(inputs_to_wait_for) == sorted(inputs_received):
            return (True, inputs_buffer)

        # This node is missing some inputs.
        #
        # Did all the upstream nodes run?
        if not all(self.graph.nodes[node_to_wait_for]["visits"] > 0 for node_to_wait_for in nodes_to_wait_for):
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
        #
        # Are we missing ALL inputs or just a few?
        if not inputs_received:
            # ALL upstream nodes have been skipped.
            #
            # Let's skip this node and add all downstream nodes to the queue.
            self.graph.nodes[component_name]["visits"] += 1
            logger.debug(
                "Skipping '%s', all input components were skipped and no inputs were received "
                "(skipped components: %s, inputs: %s)",
                component_name,
                nodes_to_wait_for,
                inputs_to_wait_for,
            )
            # Put all downstream nodes in the inputs buffer...
            downstream_nodes = [e[1] for e in self.graph.out_edges(component_name)]
            for downstream_node in downstream_nodes:
                if not downstream_node in inputs_buffer:
                    inputs_buffer[downstream_node] = {}
            # ... and never run this node
            return (False, inputs_buffer)

        # If all nodes upstream have run and we received SOME input,
        # this is a merge node that was waiting on a node that has been skipped, so it's ready to run.
        # Let's pass None on the missing edges and go ahead.
        #
        # Example:
        #
        # --------------- value ----+
        #                           |
        #        +---X--- even ---+ |
        #        |                | |
        # -- parity_check         sum --
        #        |                 |
        #        +------ odd ------+
        #
        # If 'parity_check' produces output only on 'odd', 'sum' should run
        # with 'value' and 'odd' only, because 'even' will never arrive.
        #
        inputs_to_wait_for = list(inputs_to_wait_for)
        for input_expected in inputs_to_wait_for:
            if input_expected in inputs_received:
                inputs_to_wait_for.pop(inputs_to_wait_for.index(input_expected))
        logger.debug(
            "Some components upstream of '%s' were skipped, so some inputs will be None (missing inputs: %s)",
            component_name,
            inputs_to_wait_for,
        )
        for missing_input in inputs_to_wait_for:
            component_inputs[missing_input] = None

        return (True, inputs_buffer)

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
            edge = edge_data[2]["name"]
            source_edge = edge_data[2]["origin_edge"]
            target_node = edge_data[1]

            # If this is a decision node and a loop is involved, we add to the input buffer only the nodes
            # that received their expected output and we leave the others out of the queue.
            if is_decision_node_for_loop and not edge in node_results.keys():
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
                if source_edge in node_results.keys():
                    inputs_buffer[target_node][edge] = node_results[source_edge]

        return inputs_buffer
