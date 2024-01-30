# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import importlib
import itertools
import logging
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Type, TypeVar, Union

import networkx  # type:ignore

from haystack.core.component import Component, InputSocket, OutputSocket, component
from haystack.core.errors import PipelineConnectError, PipelineError, PipelineRuntimeError, PipelineValidationError
from haystack.core.pipeline.descriptions import find_pipeline_inputs, find_pipeline_outputs
from haystack.core.pipeline.draw.draw import RenderingEngines, _draw
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.core.type_utils import _type_name, _types_are_compatible

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

        if getattr(instance, "__haystack_added_to_pipeline__", None):
            msg = (
                "Component has already been added in another Pipeline. "
                "Components can't be shared between Pipelines. Create a new instance instead."
            )
            raise PipelineError(msg)

        # Create the component's input and output sockets
        input_sockets = getattr(instance, "__haystack_input__", {})
        output_sockets = getattr(instance, "__haystack_output__", {})

        setattr(instance, "__haystack_added_to_pipeline__", self)

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

        # Find all possible connections between these two components
        possible_connections = [
            (sender_sock, receiver_sock)
            for sender_sock, receiver_sock in itertools.product(sender_socket_candidates, receiver_socket_candidates)
            if _types_are_compatible(sender_sock.type, receiver_sock.type)
        ]

        # We need this status for error messages, since we might need it in multiple places we calculate it here
        status = _connections_status(
            sender_node=sender,
            sender_sockets=sender_socket_candidates,
            receiver_node=receiver,
            receiver_sockets=receiver_socket_candidates,
        )

        if not possible_connections:
            # There's no possible connection between these two components
            if len(sender_socket_candidates) == len(receiver_socket_candidates) == 1:
                msg = (
                    f"Cannot connect '{sender}.{sender_socket_candidates[0].name}' with '{receiver}.{receiver_socket_candidates[0].name}': "
                    f"their declared input and output types do not match.\n{status}"
                )
            else:
                msg = f"Cannot connect '{sender}' with '{receiver}': " f"no matching connections available.\n{status}"
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
                    f"Cannot connect '{sender}' with '{receiver}': more than one connection is possible "
                    "between these components. Please specify the connection name, like: "
                    f"pipeline.connect('{sender}.{possible_connections[0][0].name}', "
                    f"'{receiver}.{possible_connections[0][1].name}').\n{status}"
                )
                raise PipelineConnectError(msg)

            # Get the only possible match
            sender_socket = name_matches[0][0]
            receiver_socket = name_matches[0][1]

        # Connection must be valid on both sender/receiver sides
        if not sender_socket or not receiver_socket or not sender or not receiver:
            if sender and sender_socket:
                sender_repr = f"{sender}.{sender_socket.name} ({_type_name(sender_socket.type)})"
            else:
                sender_repr = "input needed"

            if receiver and receiver_socket:
                receiver_repr = f"({_type_name(receiver_socket.type)}) {receiver}.{receiver_socket.name}"
            else:
                receiver_repr = "output"
            msg = f"Connection must have both sender and receiver: {sender_repr} -> {receiver_repr}"
            raise PipelineConnectError(msg)

        logger.debug("Connecting '%s.%s' to '%s.%s'", sender, sender_socket.name, receiver, receiver_socket.name)

        if receiver in sender_socket.receivers and sender in receiver_socket.senders:
            # This is already connected, nothing to do
            return

        if receiver_socket.senders and not receiver_socket.is_variadic:
            # Only variadic input sockets can receive from multiple senders
            msg = (
                f"Cannot connect '{sender}.{sender_socket.name}' with '{receiver}.{receiver_socket.name}': "
                f"{receiver}.{receiver_socket.name} is already connected to {receiver_socket.senders}.\n"
            )
            raise PipelineConnectError(msg)

        # Update the sockets with the new connection
        sender_socket.receivers.append(receiver)
        receiver_socket.senders.append(sender)

        # Create the new connection
        self.graph.add_edge(
            sender,
            receiver,
            key=f"{sender_socket.name}/{receiver_socket.name}",
            conn_type=_type_name(sender_socket.type),
            from_socket=sender_socket,
            to_socket=receiver_socket,
            mandatory=receiver_socket.is_mandatory,
        )

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

    def _validate_input(self, data: Dict[str, Any]):
        """
        Validates that data:
        * Each Component name actually exists in the Pipeline
        * Each Component is not missing any input
        * Each Component has only one input per input socket, if not variadic
        * Each Component doesn't receive inputs that are already sent by another Component

        Raises ValueError if any of the above is not true.
        """
        for component_name, component_inputs in data.items():
            if component_name not in self.graph.nodes:
                raise ValueError(f"Component named {component_name} not found in the pipeline.")
            instance = self.graph.nodes[component_name]["instance"]
            for socket_name, socket in instance.__haystack_input__.items():
                if socket.senders == [] and socket.is_mandatory and socket_name not in component_inputs:
                    raise ValueError(f"Missing input for component {component_name}: {socket_name}")
            for input_name in component_inputs.keys():
                if input_name not in instance.__haystack_input__:
                    raise ValueError(f"Input {input_name} not found in component {component_name}.")

        for component_name in self.graph.nodes:
            instance = self.graph.nodes[component_name]["instance"]
            for socket_name, socket in instance.__haystack_input__.items():
                component_inputs = data.get(component_name, {})
                if socket.senders == [] and socket.is_mandatory and socket_name not in component_inputs:
                    raise ValueError(f"Missing input for component {component_name}: {socket_name}")
                if socket.senders and socket_name in component_inputs and not socket.is_variadic:
                    raise ValueError(
                        f"Input {socket_name} for component {component_name} is already sent by {socket.senders}."
                    )

    # TODO: We're ignoring this linting rules for the time being, after we properly optimize this function we'll remove the noqa
    def run(  # noqa: C901, PLR0912 pylint: disable=too-many-branches
        self, data: Dict[str, Any], debug: bool = False
    ) -> Dict[str, Any]:
        # NOTE: We're assuming data is formatted like so as of now
        # data = {
        #     "comp1": {"input1": 1, "input2": 2},
        # }
        #
        # TODO: Support also this format:
        # data = {
        #     "input1": 1, "input2": 2,
        # }

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        # Raise if input is malformed in some way
        self._validate_input(data)

        # NOTE: The above NOTE and TODO are technically not true.
        # This implementation of run supports only the first format, but the second format is actually
        # never received by this method. It's handled by the `run()` method of the `Pipeline` class
        # defined in `haystack/pipeline.py`.
        # As of now we're ok with this, but we'll need to merge those two classes at some point.
        for component_name, component_inputs in data.items():
            if component_name not in self.graph.nodes:
                # This is not a component name, it must be the name of one or more input sockets.
                # Those are handled in a different way, so we skip them here.
                continue
            instance = self.graph.nodes[component_name]["instance"]
            for component_input, input_value in component_inputs.items():
                # Handle mutable input data
                data[component_name][component_input] = copy(input_value)
                if instance.__haystack_input__[component_input].is_variadic:
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

            if len(component.__haystack_input__) == 0:
                # Component has no input, can run right away
                to_run.append((node_name, component))
                continue

            for socket in component.__haystack_input__.values():
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

        # This is what we'll return at the end
        final_outputs = {}
        while len(to_run) > 0:
            name, comp = to_run.pop(0)

            if any(socket.is_variadic for socket in comp.__haystack_input__.values()) and not getattr(  # type: ignore
                comp, "is_greedy", False
            ):
                there_are_non_variadics = False
                for _, other_comp in to_run:
                    if not any(socket.is_variadic for socket in other_comp.__haystack_input__.values()):  # type: ignore
                        there_are_non_variadics = True
                        break

                if there_are_non_variadics:
                    if (name, comp) not in waiting_for_input:
                        waiting_for_input.append((name, comp))
                    continue

            if name in last_inputs and len(comp.__haystack_input__) == len(last_inputs[name]):  # type: ignore
                # This component has all the inputs it needs to run
                res = comp.run(**last_inputs[name])

                if not isinstance(res, Mapping):
                    raise PipelineRuntimeError(
                        f"Component '{name}' didn't return a dictionary. "
                        "Components must always return dictionaries: check the the documentation."
                    )

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
                        is_variadic = any(socket.is_variadic for socket in comp.__haystack_input__.values())  # type: ignore
                        if is_variadic and not getattr(comp, "is_greedy", False):
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
                    is_variadic = any(socket.is_variadic for socket in comp.__haystack_input__.values())  # type: ignore
                    if is_variadic and not getattr(comp, "is_greedy", False):
                        there_are_only_lazy_variadics = True
                        for other_name, other_comp in waiting_for_input:
                            if name == other_name:
                                continue
                            there_are_only_lazy_variadics &= any(
                                socket.is_variadic for socket in other_comp.__haystack_input__.values()  # type: ignore
                            ) and not getattr(other_comp, "is_greedy", False)

                        if not there_are_only_lazy_variadics:
                            continue

                    # Find the first component that has all the inputs it needs to run
                    has_enough_inputs = True
                    for input_socket in comp.__haystack_input__.values():  # type: ignore
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
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None
