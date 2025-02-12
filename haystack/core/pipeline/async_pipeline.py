# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.core.errors import PipelineMaxComponentRuns, PipelineRuntimeError
from haystack.core.pipeline.base import ComponentPriority, PipelineBase
from haystack.telemetry import pipeline_running

logger = logging.getLogger(__name__)


class AsyncPipeline(PipelineBase):
    """
        Asynchronous version of the Pipeline orchestration engine.

        Manages components in a pipeline allowing for concurrent processing when the pipeline's execution graph permits.
    .   This enables efficient processing of components by minimizing idle time and maximizing resource utilization.
    """

    async def run_async_generator(  # noqa: PLR0915,C901
        self, data: Dict[str, Any], include_outputs_from: Optional[Set[str]] = None, concurrency_limit: int = 4
    ) -> AsyncIterator[Dict[str, Any]]:
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
        if include_outputs_from is None:
            include_outputs_from = set()

        # 0) Basic pipeline init
        pipeline_running(self)  # telemetry
        self.warm_up()  # optional warm-up (if needed)

        # 1) Prepare ephemeral state
        ready_sem = asyncio.Semaphore(max(1, concurrency_limit))
        inputs_state: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        pipeline_outputs: Dict[str, Any] = {}
        running_tasks: Dict[asyncio.Task, str] = {}

        # A set of component names that have been scheduled but not finished:
        scheduled_components: Set[str] = set()

        # 2) Convert input data
        prepared_data = self._prepare_component_input_data(data)

        # raises ValueError if input is malformed in some way
        self._validate_input(prepared_data)
        inputs_state = self._convert_to_internal_format(prepared_data)

        # For quick lookup of downstream receivers
        ordered_names = sorted(self.graph.nodes.keys())
        cached_receivers = {n: self._find_receivers_from(n) for n in ordered_names}
        component_visits = {component_name: 0 for component_name in ordered_names}

        # We fill the queue once and raise if all components are BLOCKED
        self.validate_pipeline(self._fill_queue(ordered_names, inputs_state, component_visits))

        # Single parent span for entire pipeline execution
        with tracing.tracer.trace(
            "haystack.async_pipeline.run",
            tags={
                "haystack.pipeline.input_data": data,
                "haystack.pipeline.output_data": pipeline_outputs,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_runs_per_component": self._max_runs_per_component,
            },
        ) as parent_span:
            # -------------------------------------------------
            # We define some functions here so that they have access to local runtime state
            # (inputs, tasks, scheduled components) via closures.
            # -------------------------------------------------
            async def _run_component_async(component_name: str, component_inputs: Dict[str, Any]) -> Dict[str, Any]:
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
                if component_visits[component_name] > self._max_runs_per_component:
                    raise PipelineMaxComponentRuns(f"Max runs for '{component_name}' reached.")

                instance: Component = self.get_component(component_name)
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
                    span.set_content_tag("haystack.component.input", deepcopy(component_inputs))
                    logger.info("Running component {component_name}", component_name=component_name)

                    if getattr(instance, "__haystack_supports_async__", False):
                        outputs = await instance.run_async(**component_inputs)  # type: ignore
                    else:
                        loop = asyncio.get_running_loop()
                        outputs = await loop.run_in_executor(None, lambda: instance.run(**component_inputs))

                component_visits[component_name] += 1

                if not isinstance(outputs, dict):
                    raise PipelineRuntimeError(
                        f"Component '{component_name}' returned an invalid output type. "
                        f"Expected a dict, but got {type(outputs).__name__} instead. "
                    )

                span.set_tag("haystack.component.visits", component_visits[component_name])
                span.set_content_tag("haystack.component.outputs", deepcopy(outputs))

                # Distribute outputs to downstream inputs; also prune outputs based on `include_outputs_from`
                pruned = self._write_component_outputs(
                    component_name=component_name,
                    component_outputs=outputs,
                    inputs=inputs_state,
                    receivers=cached_receivers[component_name],
                    include_outputs_from=include_outputs_from,
                )
                if pruned:
                    pipeline_outputs[component_name] = pruned

                return pruned

            async def _run_highest_in_isolation(component_name: str) -> AsyncIterator[Dict[str, Any]]:
                """
                Runs a component with HIGHEST priority in isolation.

                We need to run components with HIGHEST priority (i.e. components with GreedyVariadic input socket)
                because otherwise, downstream components could produce additional inputs for the GreedyVariadic socket.

                :param component_name: The name of the component.
                :return: An async iterator of partial outputs.
                """
                # 1) Wait for all in-flight tasks to finish
                while running_tasks:
                    done, _pending = await asyncio.wait(running_tasks.keys(), return_when=asyncio.ALL_COMPLETED)
                    for finished in done:
                        finished_component_name = running_tasks.pop(finished)
                        partial_result = finished.result()
                        scheduled_components.discard(finished_component_name)
                        if partial_result:
                            yield_dict = {finished_component_name: deepcopy(partial_result)}
                            yield yield_dict  # partial outputs

                if component_name in scheduled_components:
                    # If it's already scheduled for some reason, skip
                    return

                # 2) Run the HIGHEST component by itself
                scheduled_components.add(component_name)
                comp_dict = self._get_component_with_graph_metadata_and_visits(
                    component_name, component_visits[component_name]
                )
                component_inputs = self._consume_component_inputs(component_name, comp_dict, inputs_state)
                component_inputs = self._add_missing_input_defaults(component_inputs, comp_dict["input_sockets"])
                result = await _run_component_async(component_name, component_inputs)
                scheduled_components.remove(component_name)
                if result:
                    yield {component_name: deepcopy(result)}

            async def _schedule_task(component_name: str) -> None:
                """
                Schedule a component to run.

                We do NOT wait for it to finish here. This allows us to run other components concurrently.

                :param component_name: The name of the component.
                """

                if component_name in scheduled_components:
                    return  # already scheduled, do nothing

                scheduled_components.add(component_name)

                comp_dict = self._get_component_with_graph_metadata_and_visits(
                    component_name, component_visits[component_name]
                )
                component_inputs = self._consume_component_inputs(component_name, comp_dict, inputs_state)
                component_inputs = self._add_missing_input_defaults(component_inputs, comp_dict["input_sockets"])

                async def _runner():
                    async with ready_sem:
                        result = await _run_component_async(component_name, component_inputs)

                    scheduled_components.remove(component_name)
                    return result

                task = asyncio.create_task(_runner())
                running_tasks[task] = component_name

            async def _wait_for_one_task_to_complete() -> AsyncIterator[Dict[str, Any]]:
                """
                Wait for exactly one running task to finish, yield partial outputs.

                If no tasks are running, does nothing.
                """
                if running_tasks:
                    done, _ = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                    for finished in done:
                        finished_component_name = running_tasks.pop(finished)
                        partial_result = finished.result()
                        scheduled_components.discard(finished_component_name)
                        if partial_result:
                            yield {finished_component_name: deepcopy(partial_result)}

            async def _wait_for_all_tasks_to_complete() -> AsyncIterator[Dict[str, Any]]:
                """
                Wait for all running tasks to finish, yield partial outputs.
                """
                if running_tasks:
                    done, _ = await asyncio.wait(running_tasks.keys(), return_when=asyncio.ALL_COMPLETED)
                    for finished in done:
                        finished_component_name = running_tasks.pop(finished)
                        partial_result = finished.result()
                        scheduled_components.discard(finished_component_name)
                        if partial_result:
                            yield {finished_component_name: deepcopy(partial_result)}

            # -------------------------------------------------
            # MAIN SCHEDULING LOOP
            # -------------------------------------------------
            while True:
                # 2) Build the priority queue of candidates
                priority_queue = self._fill_queue(ordered_names, inputs_state, component_visits)
                candidate = self._get_next_runnable_component(priority_queue, component_visits)
                if candidate is None and running_tasks:
                    # We need to wait for one task to finish to make progress and potentially unblock the priority_queue
                    async for partial_result in _wait_for_one_task_to_complete():
                        yield partial_result
                    continue

                if candidate is None and not running_tasks:
                    # done
                    break

                priority, component_name, _ = candidate  # type: ignore

                if component_name in scheduled_components:
                    # We need to wait for one task to finish to make progress
                    async for partial_result in _wait_for_one_task_to_complete():
                        yield partial_result
                    continue

                if priority == ComponentPriority.HIGHEST:
                    # 1) run alone
                    async for partial_result in _run_highest_in_isolation(component_name):
                        yield partial_result
                    # then continue the loop
                    continue

                if priority == ComponentPriority.READY:
                    # 1) schedule this one
                    await _schedule_task(component_name)

                    # 2) Possibly schedule more READY tasks if concurrency not fully used
                    while len(priority_queue) > 0 and not ready_sem.locked():
                        peek_prio, peek_name = priority_queue.peek()
                        if peek_prio in (ComponentPriority.BLOCKED, ComponentPriority.HIGHEST):
                            # can't run or must run alone => skip
                            break
                        if peek_prio == ComponentPriority.READY:
                            priority_queue.pop()
                            await _schedule_task(peek_name)
                            # keep adding while concurrency is not locked
                            continue

                        # The next is DEFER/DEFER_LAST => we only schedule it if it "becomes READY"
                        # We'll handle it in the next iteration or with incremental waiting
                        break

                # We only schedule components with priority DEFER or DEFER_LAST when no other tasks are running
                elif priority in (ComponentPriority.DEFER, ComponentPriority.DEFER_LAST) and not running_tasks:
                    await _schedule_task(component_name)

                # To make progress, we wait for one task to complete before re-starting the loop
                async for partial_result in _wait_for_one_task_to_complete():
                    yield partial_result

            # End main loop

            # 3) Drain leftover tasks
            async for partial_result in _wait_for_all_tasks_to_complete():
                yield partial_result

            # 4) Yield final pipeline outputs
            yield deepcopy(pipeline_outputs)

    async def run_async(
        self, data: Dict[str, Any], include_outputs_from: Optional[Set[str]] = None, concurrency_limit: int = 4
    ) -> Dict[str, Any]:
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
        # _name=None, _meta={'model': 'gpt-4o-mini-2024-07-18', 'index': 0, 'finish_reason': 'stop', 'usage':
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
        final: Dict[str, Any] = {}
        async for partial in self.run_async_generator(
            data=data, concurrency_limit=concurrency_limit, include_outputs_from=include_outputs_from
        ):
            final = partial
        return final or {}

    def run(
        self, data: Dict[str, Any], include_outputs_from: Optional[Set[str]] = None, concurrency_limit: int = 4
    ) -> Dict[str, Any]:
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

        async def run_inner(data, include_outputs_from):
            return await rag_pipeline.run_async(data=data, include_outputs_from=include_outputs_from)

        data = {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }

        results = rag_pipeline.run(data)

        print(results["llm"]["replies"])
        # [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='Jean lives in Paris.')],
        # _name=None, _meta={'model': 'gpt-4o-mini-2024-07-18', 'index': 0, 'finish_reason': 'stop', 'usage':
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
        """
        return asyncio.run(
            self.run_async(data=data, include_outputs_from=include_outputs_from, concurrency_limit=concurrency_limit)
        )
