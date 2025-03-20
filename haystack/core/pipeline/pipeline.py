# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from pprint import pprint
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, cast

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline.base import ComponentPriority, PipelineBase
from haystack.telemetry import pipeline_running

logger = logging.getLogger(__name__)


class Pipeline(PipelineBase):
    """
    Synchronous version of the orchestration engine.

    Orchestrates component execution according to the execution graph, one after the other.
    """

    def _run_component(
        self,
        component: Dict[str, Any],
        inputs: Dict[str, Any],
        component_visits: Dict[str, int],
        breakpoints: Optional[Set[Tuple[str, int]]] = None,
        ordered_component_names: Optional[Set[str]] = None,
        original_input_data: Optional[Dict[str, Any]] = None,
        parent_span: Optional[tracing.Span] = None,
    ) -> Dict[str, Any]:
        """
        Runs a Component with the given inputs.

        :param component: Component with component metadata.
        :param inputs: Inputs for the Component.
        :param component_visits: Current state of component visits.
        :param parent_span: The parent span to use for the newly created span.
            This is to allow tracing to be correctly linked to the pipeline run.
        :raises PipelineRuntimeError: If Component doesn't return a dictionary.
        :return: The output of the Component.
        """
        instance: Component = component["instance"]
        component_name = self.get_component_name(instance)
        component_inputs = self._consume_component_inputs(
            component_name=component_name, component=component, inputs=inputs
        )

        # We need to add missing defaults using default values from input sockets because the run signature
        # might not provide these defaults for components with inputs defined dynamically upon component initialization
        component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

        if breakpoints is not None and (component_name, component_visits[component_name]) in breakpoints:
            self.save_state(inputs, component_name, component_visits, ordered_component_names, original_input_data)
            sys.exit()  # ToDo: do this in a more graceful way

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
            component_visits[component_name] += 1

            if not isinstance(component_output, Mapping):
                raise PipelineRuntimeError(
                    f"Component '{component_name}' didn't return a dictionary. "
                    "Components must always return dictionaries: check the documentation."
                )

            span.set_tag("haystack.component.visits", component_visits[component_name])
            span.set_content_tag("haystack.component.output", component_output)

            return cast(Dict[Any, Any], component_output)

    def run(  # noqa: PLR0915, PLR0912
        self,
        data: Dict[str, Any],
        include_outputs_from: Optional[Set[str]] = None,
        breakpoints: Optional[Set[Tuple[str, int]]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
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
        rag_pipeline.add_component("llm", llm)debug_state
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

        :param breakpoints:
            Set of tuples of component names and visit counts at which the pipeline should break execution.

        :param resume_state:
            A dictionary containing the state of a previously saved pipeline execution.

        :returns:
            A dictionary where each entry corresponds to a component name
            and its output. If `include_outputs_from` is `None`, this dictionary
            will only contain the outputs of leaf components, i.e., components
            without outgoing connections.

        :raises ValueError:
            If invalid inputs are provided to the pipeline.
        :raises PipelineRuntimeError:
            If the Pipeline contains cycles with unsupported connections that would cause
            it to get stuck and fail running.
            Or if a Component fails or returns output in an unsupported type.
        :raises PipelineMaxComponentRuns:
            If a Component reaches the maximum number of times it can be run in this Pipeline.
        """
        pipeline_running(self)

        if breakpoints:
            breakpoints = self._validate_breakpoints(breakpoints)

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        if include_outputs_from is None:
            include_outputs_from = set()

        if not resume_state:
            # normalize `data`
            data = self._prepare_component_input_data(data)

            # Raise ValueError if input is malformed in some way
            self._validate_input(data)

            # We create a list of components in the pipeline sorted by name, so that the algorithm runs
            # deterministically and independent of insertion order into the pipeline.
            ordered_component_names = sorted(self.graph.nodes.keys())

            # We track component visits to decide if a component can run.
            component_visits = dict.fromkeys(ordered_component_names, 0)

            # We need to access a component's receivers multiple times during a pipeline run.
            # We store them here for easy access.
            cached_receivers = {name: self._find_receivers_from(name) for name in ordered_component_names}

        else:
            # ToDo: validate that this is a valid resume_state
            data = self._prepare_component_input_data(resume_state["pipeline_state"]["inputs"])
            component_visits = resume_state["pipeline_state"]["component_visits"]
            ordered_component_names = resume_state["pipeline_state"]["ordered_component_names"]
            cached_receivers = {name: self._find_receivers_from(name) for name in ordered_component_names}

            print(
                f"\n\nResuming pipeline from component: {resume_state['breakpoint']['component']} visit count: "
                f"{resume_state['breakpoint']['visits']}"
            )

        cached_topological_sort = None

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
            inputs = self._convert_to_internal_format(pipeline_inputs=data)
            priority_queue = self._fill_queue(ordered_component_names, inputs, component_visits)

            # check if pipeline is blocked before execution
            self.validate_pipeline(priority_queue)

            while True:
                candidate = self._get_next_runnable_component(priority_queue, component_visits)
                if candidate is None:
                    break

                priority, component_name, component = candidate
                if len(priority_queue) > 0 and priority in [ComponentPriority.DEFER, ComponentPriority.DEFER_LAST]:
                    component_name, topological_sort = self._tiebreak_waiting_components(
                        component_name=component_name,
                        priority=priority,
                        priority_queue=priority_queue,
                        topological_sort=cached_topological_sort,
                    )
                    cached_topological_sort = topological_sort
                    component = self._get_component_with_graph_metadata_and_visits(
                        component_name, component_visits[component_name]
                    )

                component_outputs = self._run_component(
                    component, inputs, component_visits, breakpoints, ordered_component_names, data, parent_span=span
                )

                # Updates global input state with component outputs and returns outputs that should go to
                # pipeline outputs.

                component_pipeline_outputs = self._write_component_outputs(
                    component_name=component_name,
                    component_outputs=component_outputs,
                    inputs=inputs,
                    receivers=cached_receivers[component_name],
                    include_outputs_from=include_outputs_from,
                )

                if component_pipeline_outputs:
                    pipeline_outputs[component_name] = deepcopy(component_pipeline_outputs)
                if self._is_queue_stale(priority_queue):
                    priority_queue = self._fill_queue(ordered_component_names, inputs, component_visits)

            return pipeline_outputs

    def _validate_breakpoints(self, breakpoints: Set[Tuple[str, Optional[int]]]) -> Set[Tuple[str, int]]:
        """
        Validates the breakpoints passed to the pipeline.

        Make sure they are all valid components registered in the pipeline,
        If the visit is not given, it is assumed to be 0, it will break on the first visit.
        If a negative number is given it means it will break on all visits, e.g.: a component running in a loop.

        :param breakpoints: Set of tuples of component names and visit counts at which the pipeline should stop.
        :returns:
            Set of valid breakpoints.
        """

        processed_breakpoints = set()

        for break_point in breakpoints:
            if break_point[0] not in self.graph.nodes:
                raise ValueError(f"Breakpoint {break_point} is not a registered component in the pipeline")
            if break_point[1] is None:
                break_point = (break_point[0], 0)
            processed_breakpoints.add(break_point)

        return processed_breakpoints

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """
        Tries to serialise any type of input that can be passed to as input to a pipeline component.
        """
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            return value.to_dict()

        # this is a hack to serialize inputs that don't have a to_dict
        elif hasattr(value, "__dict__"):
            return {
                "_type": value.__class__.__name__,
                "_module": value.__class__.__module__,
                "attributes": value.__dict__,
            }

        elif isinstance(value, dict):  # for inputs in dictionary values
            return {k: Pipeline._serialize_value(v) for k, v in value.items()}

        elif isinstance(value, (list, tuple)):  # for inputs in lists or tuples
            return [Pipeline._serialize_value(item) for item in value]

        return value

    @staticmethod
    def save_state(
        inputs: Dict[str, Any],
        component_name: str,
        component_visits: Dict[str, int],
        ordered_component_names: Optional[Set[str]],
        original_input_data,
        callback_fun: Callable = None,
    ) -> None:
        """
        Saves the state of the pipeline at a given component visit count.
        """

        import json
        from datetime import datetime

        dt = datetime.now()
        timestamp = f"{component_name}_state_{dt.strftime('%Y_%m_%d_%H_%M_%S')}.json"

        state = {
            "input_data": original_input_data,
            "timestamp": dt.isoformat(),
            "breakpoint": {"component": component_name, "visits": component_visits[component_name]},
            "pipeline_state": {
                "inputs": inputs,
                "component_visits": component_visits,
                "ordered_component_names": list(ordered_component_names) if ordered_component_names else None,
            },
        }
        try:
            serialized_inputs = Pipeline._serialize_value(state["pipeline_state"]["inputs"])
            state["pipeline_state"]["inputs"] = serialized_inputs

            with open(timestamp, "w") as f_out:
                json.dump(state, f_out, indent=2)

            logger.info(f"Pipeline state saved at: {timestamp}")

            # pass the state to some user-defined callback function
            if callback_fun:
                callback_fun(state)

        except Exception as e:
            logger.error(f"Failed to save pipeline state: {str(e)}")
            raise
