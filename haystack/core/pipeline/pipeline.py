# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Mapping, Optional, Set, cast

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline.base import (
    _COMPONENT_INPUT,
    _COMPONENT_OUTPUT,
    _COMPONENT_VISITS,
    ComponentPriority,
    PipelineBase,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.telemetry import pipeline_running

logger = logging.getLogger(__name__)


class Pipeline(PipelineBase):
    """
    Synchronous version of the orchestration engine.

    Orchestrates component execution according to the execution graph, one after the other.
    """

    @staticmethod
    def _run_component(
        component_name: str,
        component: Dict[str, Any],
        inputs: Dict[str, Any],
        component_visits: Dict[str, int],
        parent_span: Optional[tracing.Span] = None,
    ) -> Dict[str, Any]:
        """
        Runs a Component with the given inputs.

        :param component_name: Name of the Component.
        :param component: Component with component metadata.
        :param inputs: Inputs for the Component.
        :param component_visits: Current state of component visits.
        :param parent_span: The parent span to use for the newly created span.
            This is to allow tracing to be correctly linked to the pipeline run.
        :raises PipelineRuntimeError: If Component doesn't return a dictionary.
        :return: The output of the Component.
        """
        instance: Component = component["instance"]

        with PipelineBase._create_component_span(
            component_name=component_name, instance=instance, inputs=inputs, parent_span=parent_span
        ) as span:
            # We deepcopy the inputs otherwise we might lose that information
            # when we delete them in case they're sent to other Components
            span.set_content_tag(_COMPONENT_INPUT, _deepcopy_with_exceptions(inputs))
            logger.info("Running component {component_name}", component_name=component_name)
            try:
                component_output = instance.run(**inputs)
            except Exception as error:
                raise PipelineRuntimeError.from_exception(component_name, instance.__class__, error) from error
            component_visits[component_name] += 1

            if not isinstance(component_output, Mapping):
                raise PipelineRuntimeError.from_invalid_output(component_name, instance.__class__, component_output)

            span.set_tag(_COMPONENT_VISITS, component_visits[component_name])
            span.set_content_tag(_COMPONENT_OUTPUT, component_output)

            return cast(Dict[Any, Any], component_output)

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

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        # normalize `data`
        data = self._prepare_component_input_data(data)

        # Raise ValueError if input is malformed in some way
        self._validate_input(data)

        if include_outputs_from is None:
            include_outputs_from = set()

        # We create a list of components in the pipeline sorted by name, so that the algorithm runs deterministically
        # and independent of insertion order into the pipeline.
        ordered_component_names = sorted(self.graph.nodes.keys())

        # We track component visits to decide if a component can run.
        component_visits = dict.fromkeys(ordered_component_names, 0)

        # We need to access a component's receivers multiple times during a pipeline run.
        # We store them here for easy access.
        cached_receivers = {name: self._find_receivers_from(name) for name in ordered_component_names}

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

                component_inputs = self._consume_component_inputs(
                    component_name=component_name, component=component, inputs=inputs
                )
                # We need to add missing defaults using default values from input sockets because the run signature
                # might not provide these defaults for components with inputs defined dynamically upon component
                # initialization
                component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

                component_outputs = self._run_component(
                    component_name=component_name,
                    component=component,
                    inputs=component_inputs,
                    component_visits=component_visits,
                    parent_span=span,
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
                    pipeline_outputs[component_name] = _deepcopy_with_exceptions(component_pipeline_outputs)
                if self._is_queue_stale(priority_queue):
                    priority_queue = self._fill_queue(ordered_component_names, inputs, component_visits)

            return pipeline_outputs
