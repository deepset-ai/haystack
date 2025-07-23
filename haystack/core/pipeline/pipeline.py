# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Set, Union

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.core.errors import BreakpointException, PipelineInvalidPipelineSnapshotError, PipelineRuntimeError
from haystack.core.pipeline.base import (
    _COMPONENT_INPUT,
    _COMPONENT_OUTPUT,
    _COMPONENT_VISITS,
    ComponentPriority,
    PipelineBase,
)
from haystack.core.pipeline.breakpoint import (
    _create_pipeline_snapshot,
    _trigger_break_point,
    _validate_break_point_against_pipeline,
    _validate_pipeline_snapshot_against_pipeline,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, PipelineSnapshot
from haystack.telemetry import pipeline_running
from haystack.utils import _deserialize_value_with_schema

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
    ) -> Mapping[str, Any]:
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
            except BreakpointException as error:
                # Re-raise BreakpointException to preserve the original exception context
                # This is important when Agent components internally use Pipeline._run_component
                # and trigger breakpoints that need to bubble up to the main pipeline
                raise error
            except Exception as error:
                raise PipelineRuntimeError.from_exception(component_name, instance.__class__, error) from error

            component_visits[component_name] += 1

            if not isinstance(component_output, Mapping):
                raise PipelineRuntimeError.from_invalid_output(component_name, instance.__class__, component_output)

            span.set_tag(_COMPONENT_VISITS, component_visits[component_name])
            span.set_content_tag(_COMPONENT_OUTPUT, component_output)

            return component_output

    def run(  # noqa: PLR0915, PLR0912, C901, pylint: disable=too-many-branches
        self,
        data: Dict[str, Any],
        include_outputs_from: Optional[Set[str]] = None,
        *,
        break_point: Optional[Union[Breakpoint, AgentBreakpoint]] = None,
        pipeline_snapshot: Optional[PipelineSnapshot] = None,
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

        :param break_point:
            A set of breakpoints that can be used to debug the pipeline execution.

        :param pipeline_snapshot:
            A dictionary containing a snapshot of a previously saved pipeline execution.

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
        :raises PipelineBreakpointException:
            When a pipeline_breakpoint is triggered. Contains the component name, state, and partial results.
        """
        pipeline_running(self)

        if break_point and pipeline_snapshot:
            msg = (
                "pipeline_breakpoint and pipeline_snapshot cannot be provided at the same time. "
                "The pipeline run will be aborted."
            )
            raise PipelineInvalidPipelineSnapshotError(message=msg)

        # make sure all breakpoints are valid, i.e. reference components in the pipeline
        if break_point:
            _validate_break_point_against_pipeline(break_point, self.graph)

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        if include_outputs_from is None:
            include_outputs_from = set()

        pipeline_outputs: Dict[str, Any] = {}

        if not pipeline_snapshot:
            # normalize `data`
            data = self._prepare_component_input_data(data)

            # Raise ValueError if input is malformed in some way
            self.validate_input(data)

            # We create a list of components in the pipeline sorted by name, so that the algorithm runs
            # deterministically and independent of insertion order into the pipeline.
            ordered_component_names = sorted(self.graph.nodes.keys())

            # We track component visits to decide if a component can run.
            component_visits = dict.fromkeys(ordered_component_names, 0)

        else:
            # Validate the pipeline snapshot against the current pipeline graph
            _validate_pipeline_snapshot_against_pipeline(pipeline_snapshot, self.graph)

            # Handle resuming the pipeline from a snapshot
            component_visits = pipeline_snapshot.pipeline_state.component_visits
            ordered_component_names = pipeline_snapshot.ordered_component_names
            data = _deserialize_value_with_schema(pipeline_snapshot.pipeline_state.inputs)

            # include_outputs_from from the snapshot when resuming
            include_outputs_from = pipeline_snapshot.include_outputs_from

            # also intermediate_outputs from the snapshot when resuming
            pipeline_outputs = pipeline_snapshot.pipeline_state.pipeline_outputs

        cached_topological_sort = None
        # We need to access a component's receivers multiple times during a pipeline run.
        # We store them here for easy access.
        cached_receivers = {name: self._find_receivers_from(name) for name in ordered_component_names}

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

                # If there are no runnable components left, we can exit the loop
                if candidate is None:
                    break

                priority, component_name, component = candidate

                # If the next component is blocked, we do a check to see if the pipeline is possibly blocked and raise
                # a warning if it is.
                if priority == ComponentPriority.BLOCKED:
                    if self._is_pipeline_possibly_blocked(current_pipeline_outputs=pipeline_outputs):
                        # Pipeline is most likely blocked (most likely a configuration issue) so we raise a warning.
                        logger.warning(
                            "Cannot run pipeline - the next component that is meant to run is blocked.\n"
                            "Component name: '{component_name}'\n"
                            "Component type: '{component_type}'\n"
                            "This typically happens when the component is unable to receive all of its required "
                            "inputs.\nCheck the connections to this component and ensure all required inputs are "
                            "provided.",
                            component_name=component_name,
                            component_type=component["instance"].__class__.__name__,
                        )
                    # We always exit the loop since we cannot run the next component.
                    break

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

                if pipeline_snapshot:
                    if isinstance(pipeline_snapshot.break_point, AgentBreakpoint):
                        name_to_check = pipeline_snapshot.break_point.agent_name
                    else:
                        name_to_check = pipeline_snapshot.break_point.component_name
                    is_resume = name_to_check == component_name
                else:
                    is_resume = False
                component_inputs = self._consume_component_inputs(
                    component_name=component_name, component=component, inputs=inputs, is_resume=is_resume
                )

                # We need to add missing defaults using default values from input sockets because the run signature
                # might not provide these defaults for components with inputs defined dynamically upon component
                # initialization
                component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

                # Scenario 1: Pipeline snapshot is provided to resume the pipeline at a specific component
                # Deserialize the component_inputs if they are passed in the pipeline_snapshot.
                # this check will prevent other component_inputs generated at runtime from being deserialized
                if pipeline_snapshot and component_name in pipeline_snapshot.pipeline_state.inputs.keys():
                    for key, value in component_inputs.items():
                        component_inputs[key] = _deserialize_value_with_schema(value)

                # If we are resuming from an AgentBreakpoint, we inject the agent_snapshot into the Agents inputs
                if (
                    pipeline_snapshot
                    and isinstance(pipeline_snapshot.break_point, AgentBreakpoint)
                    and component_name == pipeline_snapshot.break_point.agent_name
                ):
                    component_inputs["snapshot"] = pipeline_snapshot.agent_snapshot
                    component_inputs["break_point"] = None

                # Scenario 2: A breakpoint is provided to stop the pipeline at a specific component
                if break_point:
                    should_trigger_breakpoint = False
                    should_create_snapshot = False

                    # Scenario 2.1: an AgentBreakpoint is provided to stop the pipeline at a specific component
                    if isinstance(break_point, AgentBreakpoint) and component_name == break_point.agent_name:
                        should_create_snapshot = True
                        component_inputs["break_point"] = break_point

                    # Scenario 2.2: a regular breakpoint is provided to stop the pipeline at a specific component and
                    # visit count
                    elif (
                        isinstance(break_point, Breakpoint)
                        and break_point.component_name == component_name
                        and break_point.visit_count == component_visits[component_name]
                    ):
                        should_trigger_breakpoint = True
                        should_create_snapshot = True

                    if should_create_snapshot:
                        pipeline_snapshot_inputs_serialised = deepcopy(inputs)
                        pipeline_snapshot_inputs_serialised[component_name] = deepcopy(component_inputs)
                        new_pipeline_snapshot = _create_pipeline_snapshot(
                            inputs=pipeline_snapshot_inputs_serialised,
                            break_point=break_point,
                            component_visits=component_visits,
                            original_input_data=data,
                            ordered_component_names=ordered_component_names,
                            include_outputs_from=include_outputs_from,
                            pipeline_outputs=pipeline_outputs,
                        )

                        # add the parent_snapshot to agent inputs if needed
                        if isinstance(break_point, AgentBreakpoint) and component_name == break_point.agent_name:
                            component_inputs["parent_snapshot"] = new_pipeline_snapshot

                        # trigger the breakpoint if needed
                        if should_trigger_breakpoint:
                            _trigger_break_point(
                                pipeline_snapshot=new_pipeline_snapshot, pipeline_outputs=pipeline_outputs
                            )

                component_outputs = self._run_component(
                    component_name=component_name,
                    component=component,
                    inputs=component_inputs,  # the inputs to the current component
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
                    pipeline_outputs[component_name] = deepcopy(component_pipeline_outputs)
                if self._is_queue_stale(priority_queue):
                    priority_queue = self._fill_queue(ordered_component_names, inputs, component_visits)

            if isinstance(break_point, Breakpoint):
                logger.warning(
                    "The given breakpoint {break_point} was never triggered. This is because:\n"
                    "1. The provided component is not a part of the pipeline execution path.\n"
                    "2. The component did not reach the visit count specified in the pipeline_breakpoint",
                    pipeline_breakpoint=break_point,
                )

            return pipeline_outputs
