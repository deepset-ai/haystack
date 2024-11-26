# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from warnings import warn

import networkx as nx

from haystack import logging, tracing
from haystack.core.component import Component
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

    def _run_component(
        self, name: str, inputs: Dict[str, Any], parent_span: Optional[tracing.Span] = None
    ) -> Dict[str, Any]:
        """
        Runs a Component with the given inputs.

        :param name: Name of the Component as defined in the Pipeline.
        :param inputs: Inputs for the Component.
        :param parent_span: The parent span to use for the newly created span.
            This is to allow tracing to be correctly linked to the pipeline run.
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
            parent_span=parent_span,
        ) as span:
            # We deepcopy the inputs otherwise we might lose that information
            # when we delete them in case they're sent to other Components
            span.set_content_tag("haystack.component.input", deepcopy(inputs))
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

    def _run_subgraph(  # noqa: PLR0915
        self,
        cycle: List[str],
        component_name: str,
        components_inputs: Dict[str, Dict[str, Any]],
        *,
        include_outputs_from: Optional[Set[str]] = None,
        parent_span: Optional[tracing.Span] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs a `cycle` in the Pipeline starting from `component_name`.

        This will return once there are no inputs for the Components in `cycle`.

        This is an internal method meant to be used in `Pipeline.run()` only.

        :param cycle:
            List of Components that are part of the cycle being run
        :param component_name:
            Name of the Component that will start execution of the cycle
        :param components_inputs:
            Components inputs, this might include inputs for Components that are not part
            of the cycle but part of the wider Pipeline's graph
        :param include_outputs_from:
            Set of component names whose individual outputs are to be
            included in the cycle's output. In case a Component is executed multiple times
            only the last-produced output is included.
        :returns:
            Outputs of all the Components that are not connected to other Components in `cycle`.
            If `include_outputs_from` is set those Components' outputs will be included.
        :raises PipelineMaxComponentRuns:
            If a Component reaches the maximum number of times it can be run in this Pipeline
        """
        waiting_queue: List[Tuple[str, Component]] = []
        run_queue: List[Tuple[str, Component]] = []

        # Create the run queue starting with the component that needs to run first
        start_index = cycle.index(component_name)
        for node in cycle[start_index:]:
            run_queue.append((node, self.graph.nodes[node]["instance"]))

        include_outputs_from = set() if include_outputs_from is None else include_outputs_from

        before_last_waiting_queue: Optional[Set[str]] = None
        last_waiting_queue: Optional[Set[str]] = None

        subgraph_outputs = {}
        # These are outputs that are sent to other Components but the user explicitly
        # asked to include them in the final output.
        extra_outputs = {}

        # This variable is used to keep track if we still need to run the cycle or not.
        # When a Component doesn't send outputs to another Component
        # that's inside the subgraph, we stop running this subgraph.
        cycle_received_inputs = False

        while not cycle_received_inputs:
            # Here we run the Components
            name, comp = run_queue.pop(0)
            if _is_lazy_variadic(comp) and not all(_is_lazy_variadic(comp) for _, comp in run_queue):
                # We run Components with lazy variadic inputs only if there only Components with
                # lazy variadic inputs left to run
                _enqueue_waiting_component((name, comp), waiting_queue)
                continue

            # As soon as a Component returns only output that is not part of the cycle, we can stop
            if self._component_has_enough_inputs_to_run(name, components_inputs):
                if self.graph.nodes[name]["visits"] > self._max_runs_per_component:
                    msg = f"Maximum run count {self._max_runs_per_component} reached for component '{name}'"
                    raise PipelineMaxComponentRuns(msg)

                res: Dict[str, Any] = self._run_component(name, components_inputs[name], parent_span=parent_span)

                # Delete the inputs that were consumed by the Component and are not received from
                # the user or from Components that are part of this cycle
                sockets = list(components_inputs[name].keys())
                for socket_name in sockets:
                    senders = comp.__haystack_input__._sockets_dict[socket_name].senders  # type: ignore
                    if not senders:
                        # We keep inputs that came from the user
                        continue
                    all_senders_in_cycle = all(sender in cycle for sender in senders)
                    if all_senders_in_cycle:
                        # All senders are in the cycle, we can remove the input.
                        # We'll receive it later at a certain point.
                        del components_inputs[name][socket_name]

                if name in include_outputs_from:
                    # Deepcopy the outputs to prevent downstream nodes from modifying them
                    # We don't care about loops - Always store the last output.
                    extra_outputs[name] = deepcopy(res)

                # Reset the waiting for input previous states, we managed to run a component
                before_last_waiting_queue = None
                last_waiting_queue = None

                # Check if a component doesn't send any output to components that are part of the cycle
                final_output_reached = False
                for output_socket in res.keys():
                    for receiver in comp.__haystack_output__._sockets_dict[output_socket].receivers:  # type: ignore
                        if receiver in cycle:
                            final_output_reached = True
                            break
                    if final_output_reached:
                        break

                if not final_output_reached:
                    # We stop only if the Component we just ran doesn't send any output to sockets that
                    # are part of the cycle
                    cycle_received_inputs = True

                # We manage to run this component that was in the waiting list, we can remove it.
                # This happens when a component was put in the waiting list but we reached it from another edge.
                _dequeue_waiting_component((name, comp), waiting_queue)
                for pair in self._find_components_that_will_receive_no_input(name, res, components_inputs):
                    _dequeue_component(pair, run_queue, waiting_queue)

                receivers = [item for item in self._find_receivers_from(name) if item[0] in cycle]

                res = self._distribute_output(receivers, res, components_inputs, run_queue, waiting_queue)

                # We treat a cycle as a completely independent graph, so we keep track of output
                # that is not sent inside the cycle.
                # This output is going to get distributed to the wider graph after we finish running
                # a cycle.
                # All values that are left at this point go outside the cycle.
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

        return subgraph_outputs, extra_outputs

    def run(  # noqa: PLR0915, PLR0912
        self, data: Dict[str, Any], include_outputs_from: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs the Pipeline with given input data.

        Usage:
        ```python
        from haystack import Pipeline, Document
        from haystack.utils import Secret
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.components.generators import OpenAIGenerator
        from haystack.components.builders.answer_builder import AnswerBuilder
        from haystack.components.builders.prompt_builder import PromptBuilder

        # Write documents to InMemoryDocumentStore
        document_store = InMemoryDocumentStore()
        document_store.write_documents([
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome.")
        ])

        prompt_template = \"\"\"
        Given these documents, answer the question.
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        \"\"\"

        retriever = InMemoryBM25Retriever(document_store=document_store)
        prompt_builder = PromptBuilder(template=prompt_template)
        llm = OpenAIGenerator(api_key=Secret.from_token(api_key))

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Ask a question
        question = "Who lives in Paris?"
        results = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
            }
        )

        print(results["llm"]["replies"])
        # Jean lives in Paris
        ```

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
            If the Pipeline contains cycles with unsupported connections that would cause
            it to get stuck and fail running.
            Or if a Component fails or returns output in an unsupported type.
        :raises PipelineMaxComponentRuns:
            If a Component reaches the maximum number of times it can be run in this Pipeline.
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

        # Normalize the input data
        components_inputs: Dict[str, Dict[str, Any]] = self._normalize_varidiac_input_data(data)

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

        # Break cycles in case there are, this is a noop if no cycle is found.
        # This will raise if a cycle can't be broken.
        graph_without_cycles, components_in_cycles = self._break_supported_cycles_in_graph()

        run_queue: List[Tuple[str, Component]] = []
        for node in nx.topological_sort(graph_without_cycles):
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
        ) as span:
            # Cache for extra outputs, if enabled.
            extra_outputs: Dict[Any, Any] = {}

            while len(run_queue) > 0:
                name, comp = run_queue.pop(0)

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
                    # We can reliably pick any of the cycles if there are multiple ones, the way cycles
                    # are run doesn't make a different whether we pick the first or any of the others a
                    # Component is part of.
                    subgraph_output, subgraph_extra_output = self._run_subgraph(
                        cycles[0], name, components_inputs, include_outputs_from=include_outputs_from, parent_span=span
                    )

                    # After a cycle is run the previous run_queue can't be correct anymore cause it's
                    # not modified when running the subgraph.
                    # So we reset it given the output returned by the subgraph.
                    run_queue = []

                    # Reset the waiting for input previous states, we managed to run at least one component
                    before_last_waiting_queue = None
                    last_waiting_queue = None

                    # Merge the extra outputs
                    extra_outputs.update(subgraph_extra_output)

                    for component_name, component_output in subgraph_output.items():
                        receivers = self._find_receivers_from(component_name)
                        component_output = self._distribute_output(
                            receivers, component_output, components_inputs, run_queue, waiting_queue
                        )

                        if len(component_output) > 0:
                            final_outputs[component_name] = component_output

                elif self._component_has_enough_inputs_to_run(name, components_inputs):
                    if self.graph.nodes[name]["visits"] > self._max_runs_per_component:
                        msg = f"Maximum run count {self._max_runs_per_component} reached for component '{name}'"
                        raise PipelineMaxComponentRuns(msg)

                    res: Dict[str, Any] = self._run_component(name, components_inputs[name], parent_span=span)

                    # Delete the inputs that were consumed by the Component and are not received from the user
                    sockets = list(components_inputs[name].keys())
                    for socket_name in sockets:
                        senders = comp.__haystack_input__._sockets_dict[socket_name].senders
                        if senders:
                            # Delete all inputs that are received from other Components
                            del components_inputs[name][socket_name]
                        # We keep inputs that came from the user

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
                    receivers = self._find_receivers_from(name)
                    res = self._distribute_output(receivers, res, components_inputs, run_queue, waiting_queue)

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
