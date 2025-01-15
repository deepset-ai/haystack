# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from haystack import logging, tracing
from haystack.core.component import Component, InputSocket
from haystack.core.errors import PipelineMaxComponentRuns, PipelineRuntimeError
from haystack.core.pipeline.base import PipelineBase
from haystack.core.pipeline.component_checks import (
    _NO_OUTPUT_PRODUCED,
    all_predecessors_executed,
    are_all_lazy_variadic_sockets_resolved,
    can_component_run,
    is_any_greedy_socket_ready,
    is_socket_lazy_variadic,
)
from haystack.core.pipeline.utils import FIFOPriorityQueue
from haystack.telemetry import pipeline_running

logger = logging.getLogger(__name__)


class ComponentPriority(IntEnum):
    HIGHEST = 1
    READY = 2
    DEFER = 3
    DEFER_LAST = 4
    BLOCKED = 5


class Pipeline(PipelineBase):
    """
    Synchronous version of the orchestration engine.

    Orchestrates component execution according to the execution graph, one after the other.
    """

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

    def _run_component(
        self, component: Dict[str, Any], inputs: Dict[str, Any], parent_span: Optional[tracing.Span] = None
    ) -> Tuple[Dict, Dict]:
        """
        Runs a Component with the given inputs.

        :param component: Component with component metadata.
        :param inputs: Inputs for the Component.
        :param parent_span: The parent span to use for the newly created span.
            This is to allow tracing to be correctly linked to the pipeline run.
        :raises PipelineRuntimeError: If Component doesn't return a dictionary.
        :return: The output of the Component and the new state of inputs.
        """
        instance: Component = component["instance"]
        component_name = self.get_component_name(instance)
        component_inputs, inputs = self._consume_component_inputs(
            component_name=component_name, component=component, inputs=inputs
        )

        # We need to add missing defaults using default values from input sockets because the run signature
        # might not provide these defaults for components with inputs defined dynamically upon component initialization
        component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={
                "haystack.component.name": component_name,
                "haystack.component.type": instance.__class__.__name__,
                "haystack.component.input_types": {k: type(v).__name__ for k, v in component_inputs.items()},
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
            span.set_content_tag("haystack.component.input", deepcopy(component_inputs))
            logger.info("Running component {component_name}", component_name=component_name)
            component_output = instance.run(**component_inputs)
            component["visits"] += 1

            if not isinstance(component_output, Mapping):
                raise PipelineRuntimeError(
                    f"Component '{component_name}' didn't return a dictionary. "
                    "Components must always return dictionaries: check the documentation."
                )

            span.set_tag("haystack.component.visits", component["visits"])
            span.set_content_tag("haystack.component.output", component_output)

            return component_output, inputs

    @staticmethod
    def _consume_component_inputs(component_name: str, component: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
        """
        Extracts the inputs needed to run for the component and removes them from the global inputs state.

        :param component: Component with component metadata.
        :param inputs: Global inputs state.
        :returns: The inputs for the component and the new state of global inputs.
        """
        component_inputs = inputs.get(component_name, {})
        consumed_inputs = {}
        greedy_inputs_to_remove = set()
        for socket_name, socket in component["input_sockets"].items():
            socket_inputs = component_inputs.get(socket_name, [])
            socket_inputs = [sock["value"] for sock in socket_inputs if sock["value"] != _NO_OUTPUT_PRODUCED]
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

        return consumed_inputs, inputs

    @staticmethod
    def _convert_from_legacy_format(pipeline_inputs: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
        """
        Converts the inputs to the pipeline to the format that is needed for the internal `Pipeline.run` logic.

        :param pipeline_inputs: Inputs to the pipeline.
        :returns: Converted inputs that can be used by the internal `Pipeline.run` logic.
        """
        inputs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for component_name, socket_dict in pipeline_inputs.items():
            inputs[component_name] = {}
            for socket_name, value in socket_dict.items():
                inputs[component_name][socket_name] = [{"sender": None, "value": value}]

        return inputs

    def _fill_queue(self, component_names: List[str], inputs: Dict[str, Any]) -> FIFOPriorityQueue:
        """
        Calculates the execution priority for each component and inserts it into the priority queue.

        :param component_names: Names of the components to put into the queue.
        :param inputs: Inputs to the components.
        :returns: A prioritized queue of component names.
        """
        priority_queue = FIFOPriorityQueue()
        for component_name in component_names:
            component = self._get_component_with_graph_metadata(component_name)
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
        elif is_any_greedy_socket_ready(component, inputs):
            return ComponentPriority.HIGHEST
        elif all_predecessors_executed(component, inputs):
            return ComponentPriority.READY
        elif are_all_lazy_variadic_sockets_resolved(component, inputs):
            return ComponentPriority.DEFER
        else:
            return ComponentPriority.DEFER_LAST

    def _get_component_with_graph_metadata(self, component_name: str) -> Dict[str, Any]:
        return self.graph.nodes[component_name]

    def _get_next_runnable_component(
        self, priority_queue: FIFOPriorityQueue
    ) -> Union[Tuple[ComponentPriority, str, Dict[str, Any]], None]:
        """
        Returns the next runnable component alongside its metadata from the priority queue.

        :param priority_queue: Priority queue of component names.
        :returns: The next runnable component, the component name, and its priority
            or None if no component in the queue can run.
        :raises: PipelineMaxComponentRuns if the next runnable component has exceeded the maximum number of runs.
        """
        priority_and_component_name: Union[Tuple[ComponentPriority, str], None] = priority_queue.get()

        if priority_and_component_name is not None and priority_and_component_name[0] != ComponentPriority.BLOCKED:
            priority, component_name = priority_and_component_name
            component = self._get_component_with_graph_metadata(component_name)
            if component["visits"] > self._max_runs_per_component:
                msg = f"Maximum run count {self._max_runs_per_component} reached for component '{component_name}'"
                raise PipelineMaxComponentRuns(msg)

            return priority, component_name, component

        return None

    @staticmethod
    def _write_component_outputs(
        component_name, component_outputs, inputs, receivers, include_outputs_from
    ) -> Tuple[Dict, Dict]:
        """
        Distributes the outputs of a component to the input sockets that it is connected to.

        :param component_name: The name of the component.
        :param component_outputs: The outputs of the component.
        :param inputs: The current global input state.
        :param receivers: List of receiver_name, sender_socket, receiver_socket for connected components.
        :param include_outputs_from: List of component names that should always return an output from the pipeline.
        """
        for receiver_name, sender_socket, receiver_socket in receivers:
            # We either get the value that was produced by the actor or we use the _NO_OUTPUT_PRODUCED class to indicate
            # that the sender did not produce an output for this socket.
            # This allows us to track if a pre-decessor already ran but did not produce an output.
            value = component_outputs.get(sender_socket.name, _NO_OUTPUT_PRODUCED)
            if receiver_name not in inputs:
                inputs[receiver_name] = {}

            # If we have a non-variadic or a greedy variadic receiver socket, we can just overwrite any inputs
            # that might already exist (to be reconsidered but mirrors current behavior).
            if not is_socket_lazy_variadic(receiver_socket):
                inputs[receiver_name][receiver_socket.name] = [{"sender": component_name, "value": value}]

            # If the receiver socket is lazy variadic, and it already has an input, we need to append the new input.
            # Lazy variadic sockets can collect multiple inputs.
            else:
                if not inputs[receiver_name].get(receiver_socket.name):
                    inputs[receiver_name][receiver_socket.name] = []

                inputs[receiver_name][receiver_socket.name].append({"sender": component_name, "value": value})

        # If we want to include all outputs from this actor in the final outputs, we don't need to prune any consumed
        # outputs
        if component_name in include_outputs_from:
            return component_outputs, inputs

        # We prune outputs that were consumed by any receiving sockets.
        # All remaining outputs will be added to the final outputs of the pipeline.
        consumed_outputs = {sender_socket.name for _, sender_socket, __ in receivers}
        pruned_outputs = {key: value for key, value in component_outputs.items() if key not in consumed_outputs}

        return pruned_outputs, inputs

    @staticmethod
    def _merge_component_and_pipeline_outputs(
        component_name: str, component_outputs: Dict, pipeline_outputs: Dict
    ) -> Dict:
        """
        Merges the outputs of a component with the current pipeline outputs.

        :param component_name: The name of the component.
        :param component_outputs: The outputs of the component.
        :param pipeline_outputs: The pipeline outputs.
        :returns: New pipeline outputs.
        """
        if not component_outputs:
            return pipeline_outputs
        elif component_name not in pipeline_outputs:
            pipeline_outputs[component_name] = component_outputs
        else:
            for key, value in component_outputs.items():
                if key not in pipeline_outputs[component_name]:
                    pipeline_outputs[component_name][key] = value

        return pipeline_outputs

    @staticmethod
    def _is_queue_stale(priority_queue: FIFOPriorityQueue) -> bool:
        """
        Checks if the priority queue needs to be recomputed because the priorities might have changed.

        :param priority_queue: Priority queue of component names.
        """
        return len(priority_queue) == 0 or priority_queue.peek()[0] > ComponentPriority.READY

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

        if include_outputs_from is None:
            include_outputs_from = set()

        # We create a list of components in the pipeline sorted by name, so that the algorithm runs deterministically
        # and independent of insertion order into the pipeline.
        ordered_component_names = sorted(self.graph.nodes.keys())

        # We need to access a component's receivers multiple times during a pipeline run.
        # We store them here for easy access.
        cached_receivers = {name: self._find_receivers_from(name) for name in ordered_component_names}

        pipeline_outputs: Dict[str, Any] = {}
        with tracing.tracer.trace(
            "haystack.pipeline.run",
            tags={
                "haystack.pipeline.input_data": data,
                "haystack.pipeline.output_data": pipeline_outputs,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_runs_per_component": self._max_runs_per_component,
            },
        ) as span:
            inputs = self._convert_from_legacy_format(pipeline_inputs=data)

            priority_queue = self._fill_queue(ordered_component_names, inputs)

            while True:
                candidate = self._get_next_runnable_component(priority_queue)
                if candidate is None:
                    break

                _, component_name, component = candidate
                component_outputs, inputs = self._run_component(component, inputs, parent_span=span)
                component_pipeline_outputs, inputs = self._write_component_outputs(
                    component_name=component_name,
                    component_outputs=component_outputs,
                    inputs=inputs,
                    receivers=cached_receivers[component_name],
                    include_outputs_from=include_outputs_from,
                )
                # TODO check original logic in pipeline, it looks like we don't want to override existing outputs
                # e.g. for cycles but the tests check if intermediate outputs from components in cycles are overwritten
                if component_pipeline_outputs:
                    pipeline_outputs[component_name] = component_pipeline_outputs
                if self._is_queue_stale(priority_queue):
                    priority_queue = self._fill_queue(ordered_component_names, inputs)

            return pipeline_outputs
