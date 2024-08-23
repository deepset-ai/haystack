# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from warnings import warn

import networkx as nx

from haystack import logging, tracing
from haystack.core.component import Component, InputSocket, OutputSocket
from haystack.core.errors import PipelineMaxComponentRuns, PipelineRuntimeError
from haystack.core.pipeline.base import (
    _dequeue_component,
    _dequeue_waiting_component,
    _enqueue_component,
    _enqueue_waiting_component,
)
from haystack.telemetry import pipeline_running

from .base import PipelineBase, _add_missing_input_defaults, _is_lazy_variadic

logger = logging.getLogger(__name__)


class Pipeline(PipelineBase):
    """
    Synchronous version of the orchestration engine.

    Orchestrates component execution according to the execution graph, one after the other.
    """

    def _run_component(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a Component with the given inputs.

        :param name: Name of the Component as defined in the Pipeline.
        :param inputs: Inputs for the Component.
        :raises PipelineRuntimeError: If Component doesn't return a dictionary.
        :return: The output of the Component.
        """
        instance: Component = self.graph.nodes[name]["instance"]

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={
                "haystack.component.name": name,
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
        ) as span:
            span.set_content_tag("haystack.component.input", inputs)
            logger.info("Running component {component_name}", component_name=name)
            res: Dict[str, Any] = instance.run(**inputs)
            self.graph.nodes[name]["visits"] += 1

            # After a Component that has variadic inputs is run, we need to reset the variadic inputs that were consumed
            for socket in instance.__haystack_input__._sockets_dict.values():  # type: ignore
                if socket.name not in inputs:
                    continue
                if socket.is_variadic:
                    inputs[socket.name] = []

            if not isinstance(res, Mapping):
                raise PipelineRuntimeError(
                    f"Component '{name}' didn't return a dictionary. "
                    "Components must always return dictionaries: check the documentation."
                )
            span.set_tag("haystack.component.visits", self.graph.nodes[name]["visits"])
            span.set_content_tag("haystack.component.output", res)

            return res

    def _run_subgraph(
        self,
        execution_graph: nx.MultiDiGraph,
        cycle: List[str],
        component_name: str,
        components_inputs: Dict[str, Dict[str, Any]],
    ):
        # simple_paths = nx.all_simple_paths(execution_graph, start_component, end_component)
        # nodes = set()
        # for paths in simple_paths:
        #     nodes.update(paths)
        # before_last_waiting_queue: Optional[Set[str]] = None
        # last_waiting_queue: Optional[Set[str]] = None
        # cycle_graph = execution_graph.subgraph(nodes)
        # if not nx.is_directed_acyclic_graph(cycle_graph):
        #     # TODO: This must not happen, we'll see how to handle this
        #     raise PipelineRuntimeError("Cycle detected in the subgraph")
        # sorted_graph = nx.topological_sort(cycle_graph)

        waiting_queue: List[Tuple[str, Component]] = []
        run_queue: List[Tuple[str, Component]] = []

        # Create the run queue starting with the component that needs to run first
        start_index = cycle.index(component_name)
        for node in cycle[start_index:]:
            run_queue.append((node, self.graph.nodes[node]["instance"]))

        # Find all the connections to Components that are not part of the cycle subgraph
        # TODO: Do we really need this? Let's see.
        # exit_edges = {}
        # for component_name, comp in run_queue:
        #     for socket_name, socket in comp.__haystack_output__._sockets_dict.items():
        #         for receiver in socket.receivers:
        #             if receiver not in nodes:
        #                 if component_name not in exit_edges:
        #                     exit_edges[component_name] = []
        #                 exit_edges[component_name].append(socket_name)

        # cycle_edges = {}
        # for component_name, comp in run_queue:
        #     for socket_name, socket in comp.__haystack_output__._sockets_dict.items():
        #         for receiver in socket.receivers:
        #             if receiver in nodes:
        #                 if component_name not in cycle_edges:
        #                     cycle_edges[component_name] = []
        #                 cycle_edges[component_name].append(socket_name)

        subgraph_outputs = {}

        # This variable is used to keep track if we style need to run the cycle or not.
        # TODO: Find a nicer name
        exit_edge_reached = False
        while not exit_edge_reached:
            # Here we run the Components
            name, comp = run_queue.pop(0)
            if _is_lazy_variadic(comp) and not all(_is_lazy_variadic(comp) for _, comp in run_queue):
                # We run Components with lazy variadic inputs only if there only Components with
                # lazy variadic inputs left to run
                _enqueue_waiting_component((name, comp), waiting_queue)
                continue
            # Whenever a Component is run we get its outpur edges

            # As soon as a Component returns only output that is not part of the cycle, we can stop
            if self._component_has_enough_inputs_to_run(name, components_inputs):
                if self.graph.nodes[name]["visits"] > self.max_loops_allowed:
                    msg = f"Maximum run count {self._max_runs_per_component} reached for component '{name}'"
                    raise PipelineMaxComponentRuns(msg)

                res: Dict[str, Any] = self._run_component(name, components_inputs[name])

                # TODO: Handle `include_outputs_from` here

                # Reset the waiting for input previous states, we managed to run a component
                before_last_waiting_queue = None
                last_waiting_queue = None

                component_exits_cycle = True
                for output_socket in res.keys():
                    for receiver in comp.__haystack_output__._sockets_dict[output_socket].receivers:
                        if receiver in cycle:
                            component_exits_cycle = False
                            break
                    if not component_exits_cycle:
                        break

                # TODO: This is redundant
                if component_exits_cycle:
                    # We stop only if the Component we just ran doesn't send any output to sockets that
                    # are part of the cycle.
                    exit_edge_reached = True

                # We manage to run this component that was in the waiting list, we can remove it.
                # This happens when a component was put in the waiting list but we reached it from another edge.
                _dequeue_waiting_component((name, comp), waiting_queue)
                for pair in self._find_components_that_will_receive_no_input(name, res):
                    _dequeue_component(pair, run_queue, waiting_queue)

                # - Add the output from the Component that just ran to components_inputs

                to_remove_from_component_result = set()
                for _, receiver_name, connection in self.graph.edges(nbunch=name, data=True):
                    sender_socket: OutputSocket = connection["from_socket"]
                    receiver_socket: InputSocket = connection["to_socket"]
                    if sender_socket.name not in res:
                        # This output wasn't created by the sender, nothing we can do.
                        #
                        # Some Components might have conditional outputs, so we need to check if they actually returned
                        # some output while iterating over their output sockets.
                        #
                        # A perfect example of this would be the ConditionalRouter, which will have an output for each
                        # condition it has been initialized with.
                        # Though it will return only one output at a time.
                        continue

                    if receiver_name not in cycle:
                        # This receiver is not part of the cycle, we can ignore it
                        continue

                    if receiver_name not in components_inputs:
                        components_inputs[receiver_name] = {}

                    # We keep track of the keys that were distributed to other Components.
                    # This key will be removed from component_result at the end of the loop.
                    to_remove_from_component_result.add(sender_socket.name)

                    value = res[sender_socket.name]

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
                    is_greedy = getattr(receiver, "__haystack_is_greedy__", False)
                    if receiver_socket.is_variadic:
                        if is_greedy:
                            # If the receiver is greedy, we can run it as soon as possible.
                            # First we remove it from the status lists it's in if it's there or
                            # we risk running it multiple times.
                            if pair in run_queue:
                                run_queue.remove(pair)
                            if pair in waiting_queue:
                                waiting_queue.remove(pair)
                            run_queue.append(pair)
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

                    res = {k: v for k, v in res.items() if k not in to_remove_from_component_result}

                # - Remove the Components that are not part of the cycle from the run_queue and waiting_queue
                # We do this just to avoid duplicating code for the time being.
                # to_remove = []
                # for pair in run_queue:
                #     if pair[0] not in cycle:
                #         to_remove.append(pair)

                # for pair in to_remove:
                #     run_queue.remove(pair)

                # to_remove = []
                # for pair in waiting_queue:
                #     if pair[0] not in cycle:
                #         to_remove.append(pair)

                # for pair in to_remove:
                #     waiting_queue.remove(pair)

                # - Add remaining Component output to the subgraph output
                if len(res) > 0:
                    subgraph_outputs[name] = res
            else:
                # This component doesn't have enough inputs so we can't run it yet
                _enqueue_waiting_component((name, comp), waiting_queue)

            if len(run_queue) == 0 and len(waiting_queue) > 0:
                # Check if we're stuck in a loop.
                # It's important to check whether previous waitings are None as it could be that no
                # Component has actually been run yet.
                if (
                    before_last_waiting_queue is not None
                    and last_waiting_queue is not None
                    and before_last_waiting_queue == last_waiting_queue
                ):
                    if self._is_stuck_in_a_loop(waiting_queue):
                        # We're stuck! We can't make any progress.
                        msg = (
                            "Pipeline is stuck running in a loop. Partial outputs will be returned. "
                            "Check the Pipeline graph for possible issues."
                        )
                        warn(RuntimeWarning(msg))
                        break

                    (name, comp) = self._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
                    _add_missing_input_defaults(name, comp, components_inputs)
                    _enqueue_component((name, comp), run_queue, waiting_queue)
                    continue

                before_last_waiting_queue = last_waiting_queue.copy() if last_waiting_queue is not None else None
                last_waiting_queue = {item[0] for item in waiting_queue}

                (name, comp) = self._find_next_runnable_component(components_inputs, waiting_queue)
                _add_missing_input_defaults(name, comp, components_inputs)
                _enqueue_component((name, comp), run_queue, waiting_queue)

        return subgraph_outputs

    def run(  # noqa: PLR0915
        self, data: Dict[str, Any], include_outputs_from: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs the pipeline with given input data.

        :param data:
            A dictionary of inputs for the pipeline's components. Each key is a component name
            and its value is a dictionary of that component's input parameters:
            ```
            data = {
                "comp1": {"input1": 1, "input2": 2},
            }
            ```
            For convenience, this format is also supported when input names are unique:
            ```
            data = {
                "input1": 1, "input2": 2,
            }
            ```

        :param include_outputs_from:
            Set of component names whose individual outputs are to be
            included in the pipeline's output. For components that are
            invoked multiple times (in a loop), only the last-produced
            output is included.
        :returns:
            A dictionary where each entry corresponds to a component name
            and its output. If `include_outputs_from` is `None`, this dictionary
            will only contain the outputs of leaf components, i.e., components
            without outgoing connections.

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

        # Reset the visits count for each component
        self._init_graph()

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        # normalize `data`
        data = self._prepare_component_input_data(data)

        # Raise if input is malformed in some way
        self._validate_input(data)

        # Initialize the inputs state
        components_inputs: Dict[str, Dict[str, Any]] = self._init_inputs_state(data)

        # Take all components that:
        # - have no inputs
        # - receive input from the user
        # - have at least one input not connected
        # - have at least one input that is variadic
        # run_queue: List[Tuple[str, Component]] = self._init_run_queue(data)

        # These variables are used to detect when we're stuck in a loop.
        # Stuck loops can happen when one or more components are waiting for input but
        # no other component is going to run.
        # This can happen when a whole branch of the graph is skipped for example.
        # When we find that two consecutive iterations of the loop where the waiting_queue is the same,
        # we know we're stuck in a loop and we can't make any progress.
        #
        # They track the previous two states of the waiting_queue. So if waiting_queue would n,
        # before_last_waiting_queue would be n-2 and last_waiting_queue would be n-1.
        # When we run a component, we reset both.
        before_last_waiting_queue: Optional[Set[str]] = None
        last_waiting_queue: Optional[Set[str]] = None

        # The waiting_for_input list is used to keep track of components that are waiting for input.
        waiting_queue: List[Tuple[str, Component]] = []

        include_outputs_from = set() if include_outputs_from is None else include_outputs_from

        # This is what we'll return at the end
        final_outputs: Dict[Any, Any] = {}

        execution_graph = self.graph.copy()
        cycles = nx.recursive_simple_cycles(self.graph)
        # edges_removed = []
        edges_removed = {}
        # This keeps track of all the cycles that a component is part of.
        components_in_cycles = {}
        for cycle in cycles:
            for comp in cycle:
                if comp not in components_in_cycles:
                    components_in_cycles[comp] = []
                components_in_cycles[comp].append(cycle)

            cycle = zip(cycle, cycle[1:] + cycle[:1])
            for sender_comp, receiver_comp in cycle:
                edge = execution_graph.get_edge_data(sender_comp, receiver_comp)
                # TODO: This is a bad assumption to make but for the time being it makes things easier.
                # We need to find all the variadic edges and remove them.
                assert len(edge.keys()) == 1
                # It's just one in any case
                edge_key = list(edge.keys())[0]
                edge = list(edge.values())[0]
                # For fucks sake these are still called like shit, we need to change this in `connect`
                if edge["to_socket"].is_variadic:
                    break
            else:
                continue
            # We found the variadic edge
            if sender_comp not in edges_removed:
                edges_removed[sender_comp] = []
            edges_removed[sender_comp].append(edge["from_socket"].name)

            execution_graph.remove_edge(sender_comp, receiver_comp, edge_key)
            if nx.is_directed_acyclic_graph(execution_graph):
                # We removed all the cycles, nice
                break

        run_queue: List[Tuple[str, Component]] = []
        for node in nx.topological_sort(execution_graph):
            run_queue.append((node, self.graph.nodes[node]["instance"]))


        # Set defaults inputs for those sockets that don't receive input neither from the user
        # nor from other Components.
        # If they have no default nothing is done.
        # This is important to ensure correct order execution, otherwise some variadic
        # Components that receive input from the user might be run before than they should.
        for name, comp in self.graph.nodes(data="instance"):
            if name not in components_inputs:
                components_inputs[name] = {}
            for socket_name, socket in comp.__haystack_input__._sockets_dict.items():
                if socket_name in components_inputs[name]:
                    continue
                if not socket.senders:
                    value = socket.default_value
                    if socket.is_variadic:
                        value = [value]
                    components_inputs[name][socket_name] = value


        with tracing.tracer.trace(
            "haystack.pipeline.run",
            tags={
                "haystack.pipeline.input_data": data,
                "haystack.pipeline.output_data": final_outputs,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_runs_per_component": self._max_runs_per_component,
            },
        ):
            # Cache for extra outputs, if enabled.
            extra_outputs: Dict[Any, Any] = {}

            while len(run_queue) > 0:
                name, comp = run_queue.pop(0)
                pass

                if _is_lazy_variadic(comp) and not all(_is_lazy_variadic(comp) for _, comp in run_queue):
                    # We run Components with lazy variadic inputs only if there only Components with
                    # lazy variadic inputs left to run
                    _enqueue_waiting_component((name, comp), waiting_queue)
                    continue
                if self._component_has_enough_inputs_to_run(name, components_inputs) and components_in_cycles.get(
                    name, []
                ):
                    cycles = components_in_cycles.get(name, [])

                    # This component is part of one or more cycles, let's get the first one and run it.
                    # TODO: Explain why it's fine taking the first one
                    subgraph_output = self._run_subgraph(execution_graph, cycles[0], name, components_inputs)

                    run_queue = []
                    # waiting_queue = []

                    for component_name, component_output in subgraph_output.items():
                        component_output = self._distribute_output(
                            component_name, component_output, components_inputs, run_queue, waiting_queue
                        )

                        if len(component_output) > 0:
                            final_outputs[component_name] = component_output

                    # for component_name, component_output in subgraph_output.items():
                    #     comp = self.graph.nodes[component_name]["instance"]
                    #     for socket_name, value in component_output.items():
                    #         receivers = comp.__haystack_output__._sockets_dict[socket_name].receivers
                    #         if not receivers:
                    #             if component_name not in final_outputs:
                    #                 final_outputs[component_name] = {}
                    #             final_outputs[component_name][socket_name] = value
                    #             continue
                    #         for receiver in receivers:
                    #             if receiver not in components_inputs:
                    #                 components_inputs[receiver] = {}
                    #             components_inputs[receiver][socket_name] = value

                    #             if receiver in run_queue:
                    #                 continue
                    #             run_queue.append((receiver, self.graph.nodes[receiver]["instance"]))

                    # TODO: Given the subgraph output we should understand which components need to run next.
                    # We also need to understand if the subgraph output is actually an output of the Pipeline or not.

                elif self._component_has_enough_inputs_to_run(name, components_inputs):
                    if self.graph.nodes[name]["visits"] > self._max_runs_per_component:
                        msg = f"Maximum run count {self._max_runs_per_component} reached for component '{name}'"
                        raise PipelineMaxComponentRuns(msg)

                    res: Dict[str, Any] = self._run_component(name, components_inputs[name])

                    if name in include_outputs_from:
                        # Deepcopy the outputs to prevent downstream nodes from modifying them
                        # We don't care about loops - Always store the last output.
                        extra_outputs[name] = deepcopy(res)

                    # Reset the waiting for input previous states, we managed to run a component
                    before_last_waiting_queue = None
                    last_waiting_queue = None

                    # We manage to run this component that was in the waiting list, we can remove it.
                    # This happens when a component was put in the waiting list but we reached it from another edge.
                    _dequeue_waiting_component((name, comp), waiting_queue)

                    for pair in self._find_components_that_will_receive_no_input(name, res, components_inputs):
                        _dequeue_component(pair, run_queue, waiting_queue)
                    res = self._distribute_output(name, res, components_inputs, run_queue, waiting_queue)

                    if len(res) > 0:
                        final_outputs[name] = res
                else:
                    # This component doesn't have enough inputs so we can't run it yet
                    _enqueue_waiting_component((name, comp), waiting_queue)

                if len(run_queue) == 0 and len(waiting_queue) > 0:
                    # Check if we're stuck in a loop.
                    # It's important to check whether previous waitings are None as it could be that no
                    # Component has actually been run yet.
                    if (
                        before_last_waiting_queue is not None
                        and last_waiting_queue is not None
                        and before_last_waiting_queue == last_waiting_queue
                    ):
                        if self._is_stuck_in_a_loop(waiting_queue):
                            # We're stuck! We can't make any progress.
                            msg = (
                                "Pipeline is stuck running in a loop. Partial outputs will be returned. "
                                "Check the Pipeline graph for possible issues."
                            )
                            warn(RuntimeWarning(msg))
                            break

                        (name, comp) = self._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
                        _add_missing_input_defaults(name, comp, components_inputs)
                        _enqueue_component((name, comp), run_queue, waiting_queue)
                        continue

                    before_last_waiting_queue = last_waiting_queue.copy() if last_waiting_queue is not None else None
                    last_waiting_queue = {item[0] for item in waiting_queue}

                    (name, comp) = self._find_next_runnable_component(components_inputs, waiting_queue)
                    _add_missing_input_defaults(name, comp, components_inputs)
                    _enqueue_component((name, comp), run_queue, waiting_queue)

            if len(include_outputs_from) > 0:
                for name, output in extra_outputs.items():
                    inner = final_outputs.get(name)
                    if inner is None:
                        final_outputs[name] = output
                    else:
                        # Let's not override any keys that are already
                        # in the final_outputs as they might be different
                        # from what we cached in extra_outputs, e.g. when loops
                        # are involved.
                        for k, v in output.items():
                            if k not in inner:
                                inner[k] = v

            return final_outputs
