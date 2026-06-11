# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any, ClassVar, cast

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.core.errors import BreakpointException, PipelineRuntimeError
from haystack.core.pipeline.base import (
    _COMPONENT_INPUT,
    _COMPONENT_OUTPUT,
    _COMPONENT_VISITS,
    ComponentPriority,
    PipelineBase,
    _validate_component_output_keys,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses import AsyncStreamingCallbackT, StreamingCallbackT, StreamingChunk, select_streaming_callback
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.dataclasses.streaming_chunk import _invoke_streaming_callback
from haystack.telemetry import pipeline_running
from haystack.utils.async_utils import _run_component_async

logger = logging.getLogger(__name__)


class _EndOfStream:
    """Sentinel type indicating no more chunks will arrive on the stream."""


class PipelineStreamHandle:
    """
    Handle returned by `AsyncPipeline.stream()`.

    Async-iterable over `StreamingChunk`s produced by streaming components in the pipeline. After iteration ends,
    `result` holds the final pipeline output dict.

    By default, iteration cleans up automatically: if the consumer abandons iteration, the underlying pipeline task is
    cancelled. `aclose()` is also available for explicit cleanup.
    """

    _END_OF_STREAM: ClassVar[_EndOfStream] = _EndOfStream()
    _CLEANUP_TIMEOUT_SECONDS: ClassVar[float] = 1.0

    def __init__(
        self,
        queue: asyncio.Queue["StreamingChunk | _EndOfStream"],
        task: asyncio.Task[dict[str, Any]],
        cancel_on_abandon: bool = True,
    ) -> None:
        self._queue = queue
        self._task = task
        self._cancel_on_abandon = cancel_on_abandon

    async def __aiter__(self) -> AsyncIterator[StreamingChunk]:
        """
        Drain the queue and cancel the pipeline task if iteration is abandoned.

        `__aiter__` is an async generator function: each `async for` call gets a generator and `try/finally` runs on
        exit. When `cancel_on_abandon` is True (default), abandoned iteration cancels the pipeline task; when False,
        the task is left running to completion.
        """
        try:
            while True:
                item = await self._queue.get()
                if item is self._END_OF_STREAM:
                    await self._task  # called to make exceptions surface
                    return
                yield cast(StreamingChunk, item)  # at this point, item is guaranteed to be a StreamingChunk

        finally:
            if self._cancel_on_abandon:
                await self.aclose()

    @property
    def result(self) -> dict[str, Any]:
        """
        Final pipeline output dict, available only after a successful, complete run.

        Raises a `RuntimeError` if the pipeline has not finished or was cancelled. If the pipeline failed, re-raises the
        original exception.
        """
        if not self._task.done():
            raise RuntimeError("Pipeline has not finished; iterate the handle first.")
        if self._task.cancelled():
            raise RuntimeError("Pipeline was cancelled; no result available.")
        exc = self._task.exception()
        if exc is not None:
            raise exc
        return self._task.result()

    async def aclose(self) -> None:
        """
        Cancel the underlying pipeline task.

        Bounded by `_CLEANUP_TIMEOUT_SECONDS` so that components cannot block cleanup indefinitely.
        """
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(self._task, timeout=self._CLEANUP_TIMEOUT_SECONDS)


class AsyncPipeline(PipelineBase):
    """
    Asynchronous version of the Pipeline orchestration engine.

    Manages components in a pipeline allowing for concurrent processing when the pipeline's execution graph permits.
    This enables efficient processing of components by minimizing idle time and maximizing resource utilization.
    """

    @staticmethod
    async def _run_component_async(
        component_name: str,
        component: dict[str, Any],
        component_inputs: dict[str, Any],
        component_visits: dict[str, int],
        parent_span: tracing.Span | None = None,
        *,
        break_point: Breakpoint | None = None,
    ) -> Mapping[str, Any]:
        """
        Executes a single component asynchronously.

        If the component supports async execution, it is awaited directly as it will run async;
        otherwise the component is offloaded to executor.

        The method also updates the `visits` count of the component, writes outputs to `inputs_state`,
        and returns pruned outputs that get stored in `pipeline_outputs`.

        :param component_name: The name of the component.
        :param component_inputs: Inputs for the component.
        :returns: Outputs from the component that can be yielded from run_async_generator.
        """
        if (
            isinstance(break_point, Breakpoint)
            and break_point.component_name == component_name
            and break_point.visit_count == component_visits[component_name]
        ):
            raise BreakpointException.from_triggered_breakpoint(break_point=break_point)

        instance: Component = component["instance"]

        with PipelineBase._create_component_span(
            component_name=component_name, instance=instance, inputs=component_inputs, parent_span=parent_span
        ) as span:
            # deepcopy inputs before passing to the tracer so that even if a tracer mutates them
            # the component always receives the original unmodified values
            component_inputs_copy = _deepcopy_with_exceptions(component_inputs)
            span.set_content_tag(_COMPONENT_INPUT, component_inputs)
            logger.info("Running component {component_name}", component_name=component_name)

            try:
                # For sync-only components, _run_component_async dispatches to a thread via asyncio.to_thread,
                # which copies the current contextvars context — preserving e.g. the active tracing span.
                outputs = await _run_component_async(instance, **component_inputs_copy)
            except Exception as error:
                raise PipelineRuntimeError.from_exception(component_name, instance.__class__, error) from error

            component_visits[component_name] += 1

            if not isinstance(outputs, Mapping):
                raise PipelineRuntimeError.from_invalid_output(component_name, instance.__class__, outputs)

            _validate_component_output_keys(component_name, component, outputs)

            span.set_tag(_COMPONENT_VISITS, component_visits[component_name])
            span.set_content_tag(_COMPONENT_OUTPUT, outputs)

            return outputs

    @staticmethod
    async def _wait_for_tasks(
        running_tasks: dict[asyncio.Task, str], scheduled_components: set[str], *, return_when: str
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Waits for running tasks to finish and yields their partial outputs.

        :param running_tasks: Mapping of in-flight tasks to the name of the component they run. Finished tasks are
            removed in place.
        :param scheduled_components: Set of component names that are scheduled but not yet finished. Finished
            components are discarded in place.
        :param return_when: Either `asyncio.FIRST_COMPLETED` to wait for a single task or `asyncio.ALL_COMPLETED` to
            wait for every running task.
        :returns: An async iterator of partial outputs, one per finished component that produced an output.
        """
        if not running_tasks:
            return

        done, _pending = await asyncio.wait(running_tasks.keys(), return_when=return_when)
        for finished in done:
            finished_component_name = running_tasks.pop(finished)
            try:
                partial_result = finished.result()
            except Exception:
                # A component failed. Cancel and drain the remaining in-flight tasks so they don't keep running in
                # the background (and leak) after the run is aborted, then re-raise the original error.
                await AsyncPipeline._cancel_in_flight_tasks(running_tasks, scheduled_components)
                raise
            scheduled_components.discard(finished_component_name)
            if partial_result:
                yield {finished_component_name: _deepcopy_with_exceptions(partial_result)}

    @staticmethod
    async def _cancel_in_flight_tasks(running_tasks: dict[asyncio.Task, str], scheduled_components: set[str]) -> None:
        """
        Cancels all in-flight tasks and waits for the cancellations to settle.

        Called when a component fails or when the run is abandoned early so that sibling tasks don't keep running in
        the background after the pipeline run is aborted. Exceptions from the cancelled tasks are suppressed since we
        are already unwinding.

        Note: cancellation is only effective for components that run natively async. Sync components are offloaded to
        a thread via `asyncio.to_thread` and a running thread cannot be interrupted: cancelling its task abandons
        the await, but the thread keeps running until the component's `run` returns. Its outputs are then discarded
        (the task never writes them to the pipeline state), so state stays consistent, but side effects (e.g. API
        calls) still complete and the thread can outlive this cleanup.

        :param running_tasks: Mapping of in-flight tasks to component names. Cleared in place.
        :param scheduled_components: Set of scheduled-but-unfinished component names. Cleared in place.
        """
        for task in running_tasks:
            task.cancel()
        # return_exceptions=True so a failing or cancelled sibling doesn't mask the original error we re-raise.
        await asyncio.gather(*running_tasks.keys(), return_exceptions=True)
        for component_name in running_tasks.values():
            scheduled_components.discard(component_name)
        running_tasks.clear()

    async def _run_component_in_isolation(
        self,
        *,
        component_name: str,
        inputs: dict[str, dict[str, list[dict[str, Any]]]],
        pipeline_outputs: dict[str, Any],
        component_visits: dict[str, int],
        running_tasks: dict[asyncio.Task, str],
        scheduled_components: set[str],
        cached_receivers: dict[str, Any],
        include_outputs_from: set[str],
        parent_span: tracing.Span | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Runs a component with HIGHEST priority in isolation.

        We need to run components with HIGHEST priority (i.e. components with a GreedyVariadic input socket) by
        themselves, without any other components running concurrently. Otherwise, downstream components could produce
        additional inputs for the GreedyVariadic socket.

        :param component_name: The name of the component to run.
        :param inputs: The global input state shared by all components. Mutated in place.
        :param pipeline_outputs: The accumulated pipeline outputs. Mutated in place.
        :param component_visits: Current state of component visits. Mutated in place.
        :param running_tasks: Mapping of in-flight tasks to component names. Drained in place before running.
        :param scheduled_components: Set of scheduled-but-unfinished component names. Mutated in place.
        :param cached_receivers: Precomputed mapping of component name to its downstream receivers.
        :param include_outputs_from: Set of component names whose outputs should always be included in the output.
        :param parent_span: The parent tracing span for the pipeline run.
        :returns: An async iterator of partial outputs.
        """
        # 1) Wait for all in-flight tasks to finish so the HIGHEST component runs alone.
        async for partial_outputs in self._wait_for_tasks(
            running_tasks, scheduled_components, return_when=asyncio.ALL_COMPLETED
        ):
            yield partial_outputs

        if component_name in scheduled_components:
            # If it's already scheduled for some reason, skip.
            return

        # 2) Run the HIGHEST component by itself.
        scheduled_components.add(component_name)
        component = self._get_component_with_graph_metadata_and_visits(component_name, component_visits[component_name])
        component_inputs = self._consume_component_inputs(component_name, component, inputs)
        component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

        component_outputs = await self._run_component_async(
            component_name=component_name,
            component=component,
            component_inputs=component_inputs,
            component_visits=component_visits,
            parent_span=parent_span,
        )

        pruned = self._write_component_outputs(
            component_name=component_name,
            component_outputs=component_outputs,
            inputs=inputs,
            receivers=cached_receivers[component_name],
            include_outputs_from=include_outputs_from,
        )
        if pruned or component_name in include_outputs_from:
            pipeline_outputs[component_name] = pruned

        scheduled_components.remove(component_name)
        if pruned or component_name in include_outputs_from:
            yield {component_name: _deepcopy_with_exceptions(pruned)}

    def _schedule_component(
        self,
        *,
        component_name: str,
        inputs: dict[str, dict[str, list[dict[str, Any]]]],
        pipeline_outputs: dict[str, Any],
        component_visits: dict[str, int],
        running_tasks: dict[asyncio.Task, str],
        scheduled_components: set[str],
        ready_sem: asyncio.Semaphore,
        cached_receivers: dict[str, Any],
        include_outputs_from: set[str],
        parent_span: tracing.Span | None,
    ) -> None:
        """
        Schedules a component to run as a background task without waiting for it to finish.

        Inputs are consumed synchronously here (before the task is created) so that other components scheduled in the
        same iteration of the scheduling loop observe the updated input state and don't race for the same inputs.

        :param component_name: The name of the component to schedule.
        :param inputs: The global input state shared by all components. Mutated in place.
        :param pipeline_outputs: The accumulated pipeline outputs. Mutated in place by the task once it finishes.
        :param component_visits: Current state of component visits. Mutated in place by the task once it finishes.
        :param running_tasks: Mapping of in-flight tasks to component names. The new task is registered here.
        :param scheduled_components: Set of scheduled-but-unfinished component names. Mutated in place.
        :param ready_sem: Semaphore bounding how many components run concurrently.
        :param cached_receivers: Precomputed mapping of component name to its downstream receivers.
        :param include_outputs_from: Set of component names whose outputs should always be included in the output.
        :param parent_span: The parent tracing span for the pipeline run.
        """
        if component_name in scheduled_components:
            return  # already scheduled, do nothing

        scheduled_components.add(component_name)

        component = self._get_component_with_graph_metadata_and_visits(component_name, component_visits[component_name])
        component_inputs = self._consume_component_inputs(component_name, component, inputs)
        component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

        async def _runner() -> Mapping[str, Any]:
            async with ready_sem:
                component_outputs = await self._run_component_async(
                    component_name=component_name,
                    component=component,
                    component_inputs=component_inputs,
                    component_visits=component_visits,
                    parent_span=parent_span,
                )

            pruned = self._write_component_outputs(
                component_name=component_name,
                component_outputs=component_outputs,
                inputs=inputs,
                receivers=cached_receivers[component_name],
                include_outputs_from=include_outputs_from,
            )
            if pruned or component_name in include_outputs_from:
                pipeline_outputs[component_name] = pruned

            scheduled_components.remove(component_name)
            return pruned

        task = asyncio.create_task(_runner())
        running_tasks[task] = component_name

    async def run_async_generator(  # noqa: PLR0915,C901
        self, data: dict[str, Any], include_outputs_from: set[str] | None = None, concurrency_limit: int = 4
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Executes the pipeline step by step asynchronously, yielding partial outputs when any component finishes.

        Usage:
        ```python
        from haystack import Document
        from haystack.components.builders import ChatPromptBuilder
        from haystack.dataclasses import ChatMessage
        from haystack.utils import Secret
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.components.builders.prompt_builder import PromptBuilder
        from haystack import AsyncPipeline
        import asyncio

        # Write documents to InMemoryDocumentStore
        document_store = InMemoryDocumentStore()
        document_store.write_documents([
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome.")
        ])

        prompt_template = [
            ChatMessage.from_user(
                '''
                Given these documents, answer the question.
                Documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}
                Question: {{question}}
                Answer:
                ''')
        ]

        # Create and connect pipeline components
        retriever = InMemoryBM25Retriever(document_store=document_store)
        prompt_builder = ChatPromptBuilder(template=prompt_template)
        llm = OpenAIChatGenerator()

        rag_pipeline = AsyncPipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Prepare input data
        question = "Who lives in Paris?"
        data = {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }


        # Process results as they become available
        async def process_results():
            async for partial_output in rag_pipeline.run_async_generator(
                    data=data,
                    include_outputs_from={"retriever", "llm"}
            ):
                # Each partial_output contains the results from a completed component
                if "retriever" in partial_output:
                    print("Retrieved documents:", len(partial_output["retriever"]["documents"]))
                if "llm" in partial_output:
                    print("Generated answer:", partial_output["llm"]["replies"][0])


        asyncio.run(process_results())
        ```

        :param data: Initial input data to the pipeline.
        :param concurrency_limit: The maximum number of components that are allowed to run concurrently.
        :param include_outputs_from:
            Set of component names whose individual outputs are to be
            included in the pipeline's output. For components that are
            invoked multiple times (in a loop), only the last-produced
            output is included.
        :return: An async iterator containing partial (and final) outputs.

        :raises ValueError:
            If invalid inputs are provided to the pipeline.
        :raises PipelineMaxComponentRuns:
            If a component exceeds the maximum number of allowed executions within the pipeline.
        :raises PipelineRuntimeError:
            If the Pipeline contains cycles with unsupported connections that would cause
            it to get stuck and fail running.
            Or if a Component fails or returns output in an unsupported type.
        """
        pipeline_running(self)  # telemetry

        # warm up the pipeline by running each component's warm_up method
        self.warm_up()

        if include_outputs_from is None:
            include_outputs_from = set()

        pipeline_outputs: dict[str, Any] = {}

        # Normalize `data` and raise ValueError if the input is malformed in some way.
        data = self._prepare_component_input_data(data)

        # Raise ValueError if input is malformed in some way
        self.validate_input(data)

        # We create a list of components in the pipeline sorted by name, so that the algorithm runs
        # deterministically and independent of insertion order into the pipeline.
        ordered_component_names = sorted(self.graph.nodes.keys())

        # We track component visits to decide if a component can run.
        component_visits = dict.fromkeys(ordered_component_names, 0)

        cached_topological_sort = None
        # We need to access a component's receivers multiple times during a pipeline run.
        # We store them here for easy access.
        cached_receivers = {name: self._find_receivers_from(name) for name in ordered_component_names}

        # Ephemeral concurrency state shared (and mutated in place) by the scheduling helpers below.
        ready_sem = asyncio.Semaphore(max(1, concurrency_limit))
        running_tasks: dict[asyncio.Task, str] = {}
        # A set of component names that have been scheduled but not finished.
        scheduled_components: set[str] = set()

        with tracing.tracer.trace(
            "haystack.async_pipeline.run",
            tags={
                "haystack.pipeline.input_data": data,
                "haystack.pipeline.output_data": pipeline_outputs,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_runs_per_component": self._max_runs_per_component,
            },
        ) as parent_span:
            inputs = self._convert_to_internal_format(pipeline_inputs=data)

            # check if pipeline is blocked before execution
            self.validate_pipeline(self._fill_queue(ordered_component_names, inputs, component_visits))

            try:
                while True:
                    # We rebuild the priority queue every iteration: each iteration waits for one or more concurrent
                    # tasks to finish, which mutates `inputs` and can change many components' priorities at once, so
                    # we rebuild to give every scheduling decision an up-to-date view.
                    priority_queue = self._fill_queue(ordered_component_names, inputs, component_visits)
                    candidate = self._get_next_runnable_component(priority_queue, component_visits)

                    # If we can't make progress with the queue but tasks are running, we wait for one to finish and
                    # retry to potentially unblock the priority queue.
                    if (candidate is None or candidate[0] == ComponentPriority.BLOCKED) and running_tasks:
                        async for partial_outputs in self._wait_for_tasks(
                            running_tasks, scheduled_components, return_when=asyncio.FIRST_COMPLETED
                        ):
                            yield partial_outputs
                        continue

                    # If there are no runnable components left and nothing is running, we can exit the loop.
                    if candidate is None and not running_tasks:
                        break

                    priority, component_name, component = candidate  # type: ignore

                    # If the next component is blocked, we do a check to see if the pipeline is possibly blocked and
                    # raise a warning if it is.
                    if priority == ComponentPriority.BLOCKED and not running_tasks:
                        if self._is_pipeline_possibly_blocked(current_pipeline_outputs=pipeline_outputs):
                            # Pipeline is most likely blocked (most likely a configuration issue) so we raise a warning.
                            self._find_components_blocking_pipeline(
                                priority_queue=priority_queue, component_visits=component_visits, inputs=inputs
                            )
                        # We always exit the loop since we cannot run the next component.
                        break

                    # If the next component is already scheduled, we wait for a task to finish to make progress.
                    if component_name in scheduled_components:
                        async for partial_outputs in self._wait_for_tasks(
                            running_tasks, scheduled_components, return_when=asyncio.FIRST_COMPLETED
                        ):
                            yield partial_outputs
                        continue

                    if priority == ComponentPriority.HIGHEST:
                        # A HIGHEST priority component must run alone, so we hand off to the isolation helper.
                        async for partial_outputs in self._run_component_in_isolation(
                            component_name=component_name,
                            inputs=inputs,
                            pipeline_outputs=pipeline_outputs,
                            component_visits=component_visits,
                            running_tasks=running_tasks,
                            scheduled_components=scheduled_components,
                            cached_receivers=cached_receivers,
                            include_outputs_from=include_outputs_from,
                            parent_span=parent_span,
                        ):
                            yield partial_outputs
                        continue

                    if priority == ComponentPriority.READY:
                        # Schedule this component, then schedule as many additional READY components as concurrency
                        # allows.
                        self._schedule_component(
                            component_name=component_name,
                            inputs=inputs,
                            pipeline_outputs=pipeline_outputs,
                            component_visits=component_visits,
                            running_tasks=running_tasks,
                            scheduled_components=scheduled_components,
                            ready_sem=ready_sem,
                            cached_receivers=cached_receivers,
                            include_outputs_from=include_outputs_from,
                            parent_span=parent_span,
                        )

                        # Possibly schedule more READY tasks if concurrency not fully used
                        while len(priority_queue) > 0 and not ready_sem.locked():
                            peek_priority, peek_name = priority_queue.peek()
                            if peek_priority != ComponentPriority.READY:
                                # We stop scheduling: the next component is BLOCKED (can't run), HIGHEST (must run
                                # alone), or DEFER (waiting for more inputs - we only schedule it once it becomes
                                # READY).
                                break
                            priority_queue.pop()
                            self._schedule_component(
                                component_name=peek_name,
                                inputs=inputs,
                                pipeline_outputs=pipeline_outputs,
                                component_visits=component_visits,
                                running_tasks=running_tasks,
                                scheduled_components=scheduled_components,
                                ready_sem=ready_sem,
                                cached_receivers=cached_receivers,
                                include_outputs_from=include_outputs_from,
                                parent_span=parent_span,
                            )

                    # We only schedule components with priority DEFER when no other tasks are running.
                    elif priority == ComponentPriority.DEFER and not running_tasks:
                        if len(priority_queue) > 0:
                            component_name, cached_topological_sort = self._tiebreak_waiting_components(
                                component_name=component_name,
                                priority=priority,
                                priority_queue=priority_queue,
                                topological_sort=cached_topological_sort,
                            )

                        self._schedule_component(
                            component_name=component_name,
                            inputs=inputs,
                            pipeline_outputs=pipeline_outputs,
                            component_visits=component_visits,
                            running_tasks=running_tasks,
                            scheduled_components=scheduled_components,
                            ready_sem=ready_sem,
                            cached_receivers=cached_receivers,
                            include_outputs_from=include_outputs_from,
                            parent_span=parent_span,
                        )

                    # To make progress, we wait for one task to complete before restarting the loop.
                    async for partial_outputs in self._wait_for_tasks(
                        running_tasks, scheduled_components, return_when=asyncio.FIRST_COMPLETED
                    ):
                        yield partial_outputs

                # Safety net: drain any leftover tasks once the scheduling loop has finished. With the current loop
                # both `break` paths require `running_tasks` to be empty, so this is a no-op. We keep it so that a
                # future change adding a `break` that leaves tasks in flight doesn't lose outputs.
                async for partial_outputs in self._wait_for_tasks(
                    running_tasks, scheduled_components, return_when=asyncio.ALL_COMPLETED
                ):
                    yield partial_outputs

                # Yield the final pipeline outputs.
                yield pipeline_outputs
            finally:
                # If iteration is abandoned early (e.g. the consumer stops iterating the generator and closes it) or
                # the run is cancelled, cancel any tasks still in flight so they don't leak.
                # This is a no-op on normal completion and on a component error, since no tasks are left running by then
                await self._cancel_in_flight_tasks(running_tasks, scheduled_components)

    async def run_async(
        self, data: dict[str, Any], include_outputs_from: set[str] | None = None, concurrency_limit: int = 4
    ) -> dict[str, Any]:
        """
        Provides an asynchronous interface to run the pipeline with provided input data.

        This method allows the pipeline to be integrated into an asynchronous workflow, enabling non-blocking
        execution of pipeline components.

        Usage:
        ```python
        import asyncio

        from haystack import Document
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.core.pipeline import AsyncPipeline
        from haystack.dataclasses import ChatMessage
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        # Write documents to InMemoryDocumentStore
        document_store = InMemoryDocumentStore()
        document_store.write_documents([
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome.")
        ])

        prompt_template = [
            ChatMessage.from_user(
                '''
                Given these documents, answer the question.
                Documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}
                Question: {{question}}
                Answer:
                ''')
        ]

        retriever = InMemoryBM25Retriever(document_store=document_store)
        prompt_builder = ChatPromptBuilder(template=prompt_template)
        llm = OpenAIChatGenerator()

        rag_pipeline = AsyncPipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Ask a question
        question = "Who lives in Paris?"

        async def run_inner(data, include_outputs_from):
            return await rag_pipeline.run_async(data=data, include_outputs_from=include_outputs_from)

        data = {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }

        results = asyncio.run(run_inner(data, include_outputs_from={"retriever", "llm"}))

        print(results["llm"]["replies"])
        # [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='Jean lives in Paris.')],
        # _name=None, _meta={'model': 'gpt-5-mini', 'index': 0, 'finish_reason': 'stop', 'usage':
        # {'completion_tokens': 6, 'prompt_tokens': 69, 'total_tokens': 75,
        # 'completion_tokens_details': CompletionTokensDetails(accepted_prediction_tokens=0,
        # audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), 'prompt_tokens_details':
        # PromptTokensDetails(audio_tokens=0, cached_tokens=0)}})]
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
        :param concurrency_limit: The maximum number of components that should be allowed to run concurrently.
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
        final: dict[str, Any] = {}
        async for partial in self.run_async_generator(
            data=data, concurrency_limit=concurrency_limit, include_outputs_from=include_outputs_from
        ):
            final = partial
        return final or {}

    def stream(
        self,
        data: dict[str, Any],
        *,
        streaming_components: list[str] | None = None,
        include_outputs_from: set[str] | None = None,
        concurrency_limit: int = 4,
        cancel_on_abandon: bool = True,
    ) -> PipelineStreamHandle:
        """
        Run the pipeline and return a handle that streams `StreamingChunk`s as they arrive.

        Iterate the handle with `async for` to consume chunks; after iteration ends, `handle.result` holds the final
        pipeline output dict (same as `run_async`). By default, if iteration is abandoned, the underlying pipeline task
        is cancelled automatically. Pass `cancel_on_abandon=False` to instead let the pipeline run to completion.

        For every async-capable component that exposes a `streaming_callback` input socket, a forwarder is injected at
        runtime that pushes chunks onto the handle's queue. If a `streaming_callback` is provided at component init or
        at runtime (inside `data`, e.g. `data={"llm": {"streaming_callback": cb}}`), it is also invoked for each chunk.
        Async callbacks are preferred; a sync callback is accepted but will run synchronously on the event loop and
        may block it.

        Usage:
        ```python
        import asyncio

        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.core.pipeline import AsyncPipeline
        from haystack.dataclasses import ChatMessage

        pipe = AsyncPipeline()
        pipe.add_component(
            "prompt_builder",
            ChatPromptBuilder(template=[ChatMessage.from_user("Tell me about {{topic}}")]),
        )
        pipe.add_component("llm", OpenAIChatGenerator())
        pipe.connect("prompt_builder.prompt", "llm.messages")

        async def main():
            handle = pipe.stream(data={"prompt_builder": {"topic": "Italy"}})
            async for chunk in handle:
                print(chunk.content, end="", flush=True)
            return handle.result

        result = asyncio.run(main())
        print(result["llm"]["replies"])
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
        :param streaming_components: Names of components to stream from. If `None` (default), every streaming-capable
            component is forwarded. If a list, only the listed components are forwarded; unknown names or names of
            components that do not support streaming raise `ValueError`.
        :param include_outputs_from:
            Set of component names whose individual outputs are to be
            included in the pipeline's output. For components that are
            invoked multiple times (in a loop), only the last-produced
            output is included.
        :param concurrency_limit: The maximum number of components that should be allowed to run concurrently.
        :param cancel_on_abandon: If `True` (default), the underlying pipeline task is cancelled when iteration is
            abandoned. If `False`, the pipeline runs to completion even when the consumer stops reading.
        :returns:
            A `PipelineStreamHandle` that is async-iterable over `StreamingChunk`s. After iteration ends,
            `handle.result` holds the final pipeline output dict (same shape as `run_async`).

        :raises ValueError:
            If `streaming_components` contains unknown component names or components that do not support streaming,
            or if invalid inputs are provided to the pipeline.
        :raises PipelineRuntimeError:
            Surfaced during iteration. If the Pipeline contains cycles with unsupported connections that would cause
            it to get stuck and fail running, or if a Component fails or returns output in an unsupported type.
        :raises PipelineMaxComponentRuns:
            Surfaced during iteration. If a Component reaches the maximum number of times it can be run in this
            Pipeline.
        """
        streaming_capable = {
            name
            for name in self.graph.nodes
            if getattr(self.graph.nodes[name]["instance"], "__haystack_supports_async__", False)
            and "streaming_callback" in self.graph.nodes[name]["instance"].__haystack_input__
        }
        if streaming_components is not None:
            requested = set(streaming_components)
            unknown = requested - set(self.graph.nodes)
            non_streaming = requested - unknown - streaming_capable
            if unknown:
                raise ValueError(f"Unknown components in streaming_components: {sorted(unknown)}")
            if non_streaming:
                raise ValueError(f"These components do not support streaming: {sorted(non_streaming)}")

        queue: asyncio.Queue[StreamingChunk | _EndOfStream] = asyncio.Queue()

        def make_forwarder(user_callback: StreamingCallbackT | None) -> AsyncStreamingCallbackT:
            async def forwarder(chunk: StreamingChunk) -> None:
                await queue.put(chunk)
                if user_callback is not None:
                    await _invoke_streaming_callback(user_callback, chunk)

            return forwarder

        new_data: dict[str, Any] = self._prepare_component_input_data(data)
        for name in streaming_capable:
            if streaming_components is not None and name not in streaming_components:
                continue
            instance = self.graph.nodes[name]["instance"]
            comp_inputs = new_data.setdefault(name, {})

            user_callback = select_streaming_callback(
                init_callback=getattr(instance, "streaming_callback", None),
                runtime_callback=comp_inputs.get("streaming_callback"),
                requires_async=True,
            )
            comp_inputs["streaming_callback"] = make_forwarder(user_callback)

        async def runner() -> dict[str, Any]:
            try:
                return await self.run_async(
                    new_data, include_outputs_from=include_outputs_from, concurrency_limit=concurrency_limit
                )
            finally:
                await queue.put(PipelineStreamHandle._END_OF_STREAM)

        task = asyncio.create_task(runner())
        return PipelineStreamHandle(queue=queue, task=task, cancel_on_abandon=cancel_on_abandon)

    def run(
        self, data: dict[str, Any], include_outputs_from: set[str] | None = None, concurrency_limit: int = 4
    ) -> dict[str, Any]:
        """
        Provides a synchronous interface to run the pipeline with given input data.

        Internally, the pipeline components are executed asynchronously, but the method itself
        will block until the entire pipeline execution is complete.

        In case you need asynchronous methods, consider using `run_async` or `run_async_generator`.

        Usage:
        ```python
        from haystack import Document
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.core.pipeline import AsyncPipeline
        from haystack.dataclasses import ChatMessage
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        # Write documents to InMemoryDocumentStore
        document_store = InMemoryDocumentStore()
        document_store.write_documents([
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome.")
        ])

        prompt_template = [
            ChatMessage.from_user(
                '''
                Given these documents, answer the question.
                Documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}
                Question: {{question}}
                Answer:
                ''')
        ]


        retriever = InMemoryBM25Retriever(document_store=document_store)
        prompt_builder = ChatPromptBuilder(template=prompt_template)
        llm = OpenAIChatGenerator()

        rag_pipeline = AsyncPipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Ask a question
        question = "Who lives in Paris?"

        data = {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }

        results = rag_pipeline.run(data)

        print(results["llm"]["replies"])
        # [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='Jean lives in Paris.')],
        # _name=None, _meta={'model': 'gpt-5-mini', 'index': 0, 'finish_reason': 'stop', 'usage':
        # {'completion_tokens': 6, 'prompt_tokens': 69, 'total_tokens': 75, 'completion_tokens_details':
        # CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0,
        # rejected_prediction_tokens=0), 'prompt_tokens_details': PromptTokensDetails(audio_tokens=0,
        # cached_tokens=0)}})]
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
        :param concurrency_limit: The maximum number of components that should be allowed to run concurrently.

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
        :raises RuntimeError:
            If called from within an async context. Use `run_async` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: safe to use asyncio.run()
            return asyncio.run(
                self.run_async(
                    data=data, include_outputs_from=include_outputs_from, concurrency_limit=concurrency_limit
                )
            )
        else:
            # Running loop present: do not create the coroutine and do not call asyncio.run()
            raise RuntimeError(
                "Cannot call run() from within an async context. Use 'await pipeline.run_async(...)' instead."
            )
