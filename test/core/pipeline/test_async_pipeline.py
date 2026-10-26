# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import logging
from dataclasses import replace
from typing import Any

import pytest

from haystack import Document, Pipeline, component
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import BranchJoiner, DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.routers import ConditionalRouter
from haystack.components.routers.conditional_router import Route
from haystack.core.component.types import Variadic
from haystack.core.errors import PipelineRuntimeError
from haystack.document_stores.in_memory import InMemoryDocumentStore

_test_context_var: contextvars.ContextVar[str] = contextvars.ContextVar("_test_context_var", default="unset")


def test_async_pipeline_reentrance(waiting_component, spying_tracer):
    pp = Pipeline()
    pp.add_component("wait", waiting_component())

    run_data = [{"wait_for": 0.001}, {"wait_for": 0.002}]

    async def run_all():
        # Create concurrent tasks for each pipeline run
        tasks = [pp.run_async(data) for data in run_data]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())
    component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run"]
    assert len(component_spans) == 2
    for span in component_spans:
        assert span.tags["haystack.component.visits"] == 1


def test_run_in_sync_context(waiting_component):
    pp = Pipeline()
    pp.add_component("wait", waiting_component())

    result = pp.run({"wait_for": 0.001})

    assert result == {"wait": {"waited_for": 0.001}}


def test_run_async_accepts_dict_valued_flat_input():
    @component
    class DictEcho:
        @component.output_types(result=dict)
        def run(self, payload: dict) -> dict:
            return {"result": payload}

    pipeline = Pipeline()
    pipeline.add_component("echo", DictEcho())
    expected = {"echo": {"result": {"x": 1}}}

    assert asyncio.run(pipeline.run_async({"payload": {"x": 1}})) == expected


def test_run_async_with_invalid_concurrency_limit():
    pp = Pipeline()
    with pytest.raises(ValueError, match="concurrency_limit must be greater than or equal to 1"):
        asyncio.run(pp.run_async({}, concurrency_limit=0))


def test_component_with_empty_dict_as_output_appears_in_results():
    """Test that components that return an empty dict as output appear in results as an empty dict"""

    @component
    class Producer:
        def __init__(self, prefix: str):
            self.prefix = prefix

        @component.output_types(value=str | None)
        def run(self, text: str | None) -> dict[str, str | None]:
            return {"value": f"{self.prefix}: {text}"}

        @component.output_types(value=str | None)
        async def run_async(self, text: str | None) -> dict[str, str | None]:
            return {"value": f"{self.prefix}: {text}"}

    @component
    class EmptyProcessor:
        @component.output_types()
        def run(self, sources: list[str]) -> dict:
            # Returns empty dict when sources is empty
            return {}

        @component.output_types()
        async def run_async(self, sources: list[str]) -> dict:
            # Returns empty dict when sources is empty
            return {}

    @component
    class Combiner:
        @component.output_types(combined=str)
        def run(self, input_a: str | None, input_b: str | None) -> dict[str, str]:
            if input_a is None:
                input_a = ""
            if input_b is None:
                input_b = ""
            return {"combined": f"{input_a} | {input_b}"}

        @component.output_types(combined=str)
        async def run_async(self, input_a: str | None, input_b: str | None) -> dict[str, str]:
            if input_a is None:
                input_a = ""
            if input_b is None:
                input_b = ""
            return {"combined": f"{input_a} | {input_b}"}

    pp = Pipeline()
    pp.add_component("producer_a", Producer("A"))
    pp.add_component("producer_b", Producer("B"))
    pp.add_component("empty_processor", EmptyProcessor())
    pp.add_component("combiner", Combiner())

    pp.connect("producer_a.value", "combiner.input_a")
    pp.connect("producer_b.value", "combiner.input_b")

    result = asyncio.run(
        pp.run_async(
            {"producer_a": {"text": "hello"}, "producer_b": {"text": "world"}, "empty_processor": {"sources": []}},
            include_outputs_from={"producer_a", "empty_processor", "combiner"},
        )
    )

    # Producer A should appear in results because it's in include_outputs_from
    assert "producer_a" in result
    assert result["producer_a"] == {"value": "A: hello"}
    # Producer B should NOT appear since it's not in include_outputs_from
    assert "producer_b" not in result
    # Combiner should appear in results
    assert "combiner" in result
    assert result["combiner"] == {"combined": "A: hello | B: world"}
    # Empty processor should appear in results even though it returns an empty dict
    # because it's in include_outputs_from
    assert "empty_processor" in result
    assert result["empty_processor"] == {}


@pytest.mark.asyncio
async def test__run_component_async_warns_on_extra_output_keys(caplog):
    """Test that a warning is raised when a component returns undeclared output keys."""
    caplog.set_level(logging.WARNING)

    @component
    class ExtraKeyComponent:
        @component.output_types(output=str)
        def run(self, value: str) -> dict[str, str]:
            return {"output": value, "extra_key": "unexpected"}

    pp = Pipeline()
    pp.add_component("extra", ExtraKeyComponent())

    await pp._run_component_async(
        component_name="extra",
        component=pp._get_component_with_graph_metadata_and_visits("extra", 0),
        component_inputs={"value": "test"},
        component_visits={"extra": 0},
    )
    assert "returned output keys" in caplog.text
    assert "extra_key" in caplog.text
    assert "not declared" in caplog.text


@pytest.mark.asyncio
async def test__run_component_async_no_warning_on_correct_output_keys(caplog):
    """Test that no warning is raised when a component returns the correct output keys."""
    caplog.set_level(logging.WARNING)

    @component
    class CorrectComponent:
        @component.output_types(output=str)
        def run(self, value: str) -> dict[str, str]:
            return {"output": value}

    pp = Pipeline()
    pp.add_component("correct", CorrectComponent())

    await pp._run_component_async(
        component_name="correct",
        component=pp._get_component_with_graph_metadata_and_visits("correct", 0),
        component_inputs={"value": "test"},
        component_visits={"correct": 0},
    )
    assert "returned output keys" not in caplog.text
    assert "did not produce output keys" not in caplog.text


def test_async_pipeline_is_possibly_blocked_warning_message(caplog):
    """
    Test that the pipeline raises a warning when it is possibly blocked due to missing inputs.

    The situation below looks a little contrived, but it has happened in practice that users create pipelines
    and accidentally made a mistake in their component code.
    """
    caplog.set_level(logging.WARNING)

    @component
    class MisconfiguredComponent:
        # Here we purposely declare other_output which is not actually returned by the run() method
        @component.output_types(output=str, other_output=str)
        def run(self, required_input: str) -> dict[str, str]:
            return {"output": "test"}

    @component
    class SimpleComponentTwoInputs:
        @component.output_types(output=str)
        def run(self, required_input: str, second_required_input: str) -> dict[str, str]:
            return {"output": "test"}

    pp = Pipeline()
    pp.add_component("first", MisconfiguredComponent())
    pp.add_component("second", SimpleComponentTwoInputs())

    # NOTE: We connect both outputs from the first component to the second component, but the first component
    # doesn't actually produce other_output, so the second component will be blocked due to missing input.
    pp.connect("first.output", "second.required_input")
    pp.connect("first.other_output", "second.second_required_input")

    asyncio.run(pp.run_async({"first": {"required_input": "test"}}))
    assert "Cannot run pipeline - the pipeline appears to be blocked." in caplog.text
    assert " - 'second' (SimpleComponentTwoInputs)" in caplog.text


def test_async_pipeline_ensure_inputs_are_deep_copied():
    """
    Test to ensure that async pipeline deep copies the inputs before passing them to components.

    This is important to prevent unintended side effects when components modify their inputs especially when
    the output from one component is passed to multiple other components.

    Some other notes about how this situation can arise in practice:
    - When a component returns a mutable object (like a Document) and that output is passed to multiple other
      components.
    - This doesn't happen when using output types like strings or integers, because they are not shared by
      reference so we will only commonly see this for objects like our dataclasses.
    """

    @component
    class SimpleComponent:
        @component.output_types(output=Document)
        def run(self, document: Document) -> dict[str, Document]:
            # Creates a new document to avoid modifying in place
            new_document = Document(content=document.content)
            return {"output": new_document}

    @component
    class ModifyingComponent:
        @component.output_types(output=Document)
        def run(self, document: Document) -> dict[str, Document]:
            return {"output": replace(document, content="modified")}

    pp = Pipeline()
    pp.add_component("first", SimpleComponent())
    pp.add_component("modifier", ModifyingComponent())
    # It's important that the following component has a name lower down the alphabetical order than "modifier",
    # since the pipeline runs components in a first-in-first-out manner based on ordered_component_names which is
    # sorted alphabetically.
    pp.add_component("second", SimpleComponent())

    pp.connect("first.output", "modifier.document")
    pp.connect("first.output", "second.document")

    result = asyncio.run(pp.run_async({"first": {"document": Document(content="original")}}))

    assert result["modifier"]["output"].content == "modified"
    # Without deep copying the inputs, the second component would also see the modified document and produce
    # "modified" instead of "original"
    assert result["second"]["output"].content == "original"


def test_async_pipeline_does_not_corrupt_outputs():
    """
    Test that a component's output collected via include_outputs_from is not corrupted when a downstream
    component receives and mutates the same data in-place.
    """

    @component
    class Producer:
        @component.output_types(doc=Document)
        def run(self) -> dict:
            return {"doc": Document(content="original")}

    @component
    class Mutator:
        @component.output_types(doc=Document)
        def run(self, doc: Document) -> dict:
            return {"doc": replace(doc, content="mutated")}

    pipe = Pipeline()
    pipe.add_component("producer", Producer())
    pipe.add_component("mutator", Mutator())
    pipe.connect("producer.doc", "mutator.doc")

    result = asyncio.run(pipe.run_async({}, include_outputs_from={"producer"}))

    assert result["producer"]["doc"].content == "original"
    assert result["mutator"]["doc"].content == "mutated"


def test_run_async_passes_variadic_inputs_in_same_order_as_run():
    @component
    class Sender:
        def __init__(self, value: str, delay: float):
            self.value = value
            self.delay = delay

        @component.output_types(value=str)
        def run(self) -> dict:
            return {"value": self.value}

        @component.output_types(value=str)
        async def run_async(self) -> dict:
            await asyncio.sleep(self.delay)
            return {"value": self.value}

    @component
    class Collector:
        @component.output_types(values=list[str])
        def run(self, values: Variadic[str]) -> dict:
            return {"values": list(values)}

    pipe = Pipeline()
    # "a" sorts first and runs first in run(), but finishes last in run_async()
    pipe.add_component("a", Sender(value="a", delay=0.05))
    pipe.add_component("b", Sender(value="b", delay=0))
    pipe.add_component("collector", Collector())
    pipe.connect("a.value", "collector.values")
    pipe.connect("b.value", "collector.values")

    sync_result = pipe.run({})
    async_result = asyncio.run(pipe.run_async({}))

    assert sync_result == {"collector": {"values": ["a", "b"]}}
    assert async_result == sync_result


def test_run_async_passes_variadic_inputs_in_connection_order_on_every_loop_iteration():
    @component
    class Tagger:
        def __init__(self, tag: str, delay: float):
            self.tag = tag
            self.delay = delay

        @component.output_types(tagged=str)
        def run(self, count: int) -> dict:
            return {"tagged": f"{self.tag}{count}"}

        @component.output_types(tagged=str)
        async def run_async(self, count: int) -> dict:
            await asyncio.sleep(self.delay)
            return {"tagged": f"{self.tag}{count}"}

    @component
    class RoundCollector:
        def __init__(self, rounds: int):
            self.rounds = rounds
            self.collected: list[list[str]] = []

        @component.output_types(next_count=int, collected=list[list[str]])
        def run(self, tagged: Variadic[str], count: int) -> dict:
            self.collected.append(list(tagged))
            if count + 1 < self.rounds:
                return {"next_count": count + 1}
            return {"collected": self.collected}

    def build_pipeline() -> Pipeline:
        pipe = Pipeline()
        pipe.add_component("start", BranchJoiner(int))
        # "a" sorts first and runs first in run(), "b" finishes last in run_async(), and "b" is connected first
        pipe.add_component("a", Tagger(tag="a", delay=0))
        pipe.add_component("b", Tagger(tag="b", delay=0.05))
        pipe.add_component("collector", RoundCollector(rounds=3))
        pipe.connect("start.value", "a.count")
        pipe.connect("start.value", "b.count")
        pipe.connect("start.value", "collector.count")
        pipe.connect("b.tagged", "collector.tagged")
        pipe.connect("a.tagged", "collector.tagged")
        pipe.connect("collector.next_count", "start.value")
        return pipe

    data = {"start": {"value": 0}}
    sync_result = build_pipeline().run(data)
    async_result = asyncio.run(build_pipeline().run_async(data))

    expected = {"collector": {"collected": [["b0", "a0"], ["b1", "a1"], ["b2", "a2"]]}}
    assert sync_result == expected
    assert async_result == expected


@component
class _QueryEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict:
        return {"embedding": [1.0, 0.0, 0.0]}


def _programming_document_store() -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore()
    document_store.write_documents(
        [
            Document(
                id="1", content="Python is a programming language", embedding=[1.0, 0.0, 0.0], meta={"year": 2024}
            ),
            Document(id="2", content="Java is a programming language", embedding=[0.0, 1.0, 0.0], meta={"year": 2023}),
            Document(id="3", content="Pythons are large snakes", embedding=[0.8, 0.0, 0.6], meta={"year": 2024}),
        ]
    )
    return document_store


def _ids_and_scores(documents: list[Document]) -> list[tuple[str, float | None]]:
    return [(doc.id, doc.score) for doc in documents]


def _expected_joined_ids_and_scores(
    result: dict, joiner: DocumentJoiner, retrievers: list[str]
) -> list[tuple[str, float | None]]:
    # Joins the retrievers' outputs directly, in the given order
    return _ids_and_scores(joiner.run(documents=[result[name]["documents"] for name in retrievers])["documents"])


@pytest.mark.parametrize(
    "connection_order",
    [["dense_retriever", "sparse_retriever"], ["sparse_retriever", "dense_retriever"]],
    ids=["dense_first", "sparse_first"],
)
def test_run_async_applies_document_joiner_weights_to_the_retrievers_in_connection_order(connection_order):
    @component
    class SlowBM25Retriever(InMemoryBM25Retriever):
        @component.output_types(documents=list[Document])
        async def run_async(
            self,
            query: str,
            filters: dict[str, Any] | None = None,
            top_k: int | None = None,
            scale_score: bool | None = None,
        ) -> dict[str, list[Document]]:
            await asyncio.sleep(0.05)
            # @component rebuilds the class, which breaks a zero-argument super()
            return await InMemoryBM25Retriever.run_async(
                self, query=query, filters=filters, top_k=top_k, scale_score=scale_score
            )

    document_store = _programming_document_store()
    pipe = Pipeline()
    # "dense_retriever" sits one step deeper than "sparse_retriever", so it runs last in run(), and the sparse retriever
    # finishes last in run_async(): in each connection order, one run mode finishes the retrievers in the other order
    pipe.add_component("embedder", _QueryEmbedder())
    pipe.add_component("dense_retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    pipe.add_component("sparse_retriever", SlowBM25Retriever(document_store=document_store))
    joiner = DocumentJoiner(join_mode="merge", weights=[0.7, 0.3])
    pipe.add_component("joiner", joiner)
    pipe.connect("embedder.embedding", "dense_retriever.query_embedding")
    for retriever in connection_order:
        pipe.connect(f"{retriever}.documents", "joiner.documents")

    data = {"embedder": {"text": "python programming"}, "sparse_retriever": {"query": "python programming"}}
    include_outputs_from = {"dense_retriever", "sparse_retriever"}
    sync_result = pipe.run(data, include_outputs_from=include_outputs_from)
    async_result = asyncio.run(pipe.run_async(data, include_outputs_from=include_outputs_from))

    expected = _expected_joined_ids_and_scores(result=sync_result, joiner=joiner, retrievers=connection_order)
    assert _ids_and_scores(sync_result["joiner"]["documents"]) == expected
    assert _ids_and_scores(async_result["joiner"]["documents"]) == expected


@pytest.mark.parametrize("year", [None, 2024], ids=["filter_branch_skipped", "filter_branch_runs"])
def test_run_async_applies_document_joiner_weights_in_connection_order_when_an_optional_filter_branch_is_skipped(year):
    routes: list[Route] = [
        {"condition": "{{ year is not none }}", "output": "{{ year }}", "output_name": "year", "output_type": int},
        {
            "condition": "{{ year is none }}",
            "output": "{{ year }}",
            "output_name": "no_year",
            "output_type": type(None),
        },
    ]

    document_store = _programming_document_store()
    pipe = Pipeline()
    pipe.add_component("router", ConditionalRouter(routes=routes, optional_variables=["year"]))
    pipe.add_component(
        "filter_builder",
        OutputAdapter(
            template="{{ {'field': 'meta.year', 'operator': '==', 'value': year} }}", output_type=dict[str, Any]
        ),
    )
    pipe.add_component("embedder", _QueryEmbedder())
    pipe.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    pipe.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=document_store))
    joiner = DocumentJoiner(join_mode="merge", weights=[0.7, 0.3])
    pipe.add_component("joiner", joiner)
    pipe.connect("router.year", "filter_builder.year")
    pipe.connect("filter_builder.output", "bm25_retriever.filters")
    pipe.connect("embedder.embedding", "embedding_retriever.query_embedding")
    pipe.connect("embedding_retriever.documents", "joiner.documents")
    pipe.connect("bm25_retriever.documents", "joiner.documents")

    # Without a year, filter_builder never runs, so run() defers bm25_retriever until after embedding_retriever even
    # though bm25_retriever only consumes user input
    data = {
        "router": {"year": year} if year is not None else {},
        "embedder": {"text": "python programming"},
        "bm25_retriever": {"query": "python programming"},
    }
    include_outputs_from = {"embedding_retriever", "bm25_retriever"}
    sync_result = pipe.run(data, include_outputs_from=include_outputs_from)
    async_result = asyncio.run(pipe.run_async(data, include_outputs_from=include_outputs_from))

    expected = _expected_joined_ids_and_scores(
        result=sync_result, joiner=joiner, retrievers=["embedding_retriever", "bm25_retriever"]
    )
    assert _ids_and_scores(sync_result["joiner"]["documents"]) == expected
    assert _ids_and_scores(async_result["joiner"]["documents"]) == expected


@component
class _Doubler:
    """Minimal component used to exercise the isolation helper."""

    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, int]:
        return {"value": value * 2}


def _build_isolation_state(pipeline: Pipeline, data: dict) -> dict:
    """
    Build the ephemeral run state that `_run_component_in_isolation` expects.

    Mirrors the setup `run_async_generator` performs before the scheduling loop.
    """
    inputs = pipeline._convert_to_internal_format(pipeline._prepare_component_input_data(data))
    names = sorted(pipeline.graph.nodes.keys())
    return {
        "inputs": inputs,
        "pipeline_outputs": {},
        "component_visits": dict.fromkeys(names, 0),
        "running_tasks": {},
        "scheduled_components": set(),
        "cached_receivers": {name: pipeline._find_receivers_from(name) for name in names},
        "include_outputs_from": set(),
        "parent_span": None,
    }


class TestRunComponentInIsolation:
    @pytest.mark.asyncio
    async def test_runs_component_and_yields_output(self):
        pp = Pipeline()
        pp.add_component("doubler", _Doubler())
        state = _build_isolation_state(pp, {"doubler": {"value": 3}})

        results = [out async for out in pp._run_component_in_isolation(component_name="doubler", **state)]

        assert results == [{"doubler": {"value": 6}}]
        assert state["pipeline_outputs"] == {"doubler": {"value": 6}}
        assert state["component_visits"]["doubler"] == 1
        # The component is added to and removed from scheduled_components over the course of the run.
        assert state["scheduled_components"] == set()

    @pytest.mark.asyncio
    async def test_runs_greedy_component_consuming_single_input(self):
        pp = Pipeline()
        pp.add_component("joiner", BranchJoiner(type_=int))
        state = _build_isolation_state(pp, {})
        # Two values are queued on the greedy variadic socket; greedy consumption keeps only the first.
        state["inputs"]["joiner"] = {"value": [{"sender": None, "value": 1}, {"sender": None, "value": 2}]}

        results = [out async for out in pp._run_component_in_isolation(component_name="joiner", **state)]

        assert results == [{"joiner": {"value": 1}}]
        assert state["component_visits"]["joiner"] == 1

    @pytest.mark.asyncio
    async def test_drains_in_flight_tasks_before_running(self):
        pp = Pipeline()
        pp.add_component("doubler", _Doubler())
        state = _build_isolation_state(pp, {"doubler": {"value": 3}})

        async def _in_flight() -> dict:
            return {"value": 99}

        task = asyncio.create_task(_in_flight())
        state["running_tasks"][task] = "other"
        state["scheduled_components"].add("other")

        results = [out async for out in pp._run_component_in_isolation(component_name="doubler", **state)]

        # The in-flight task is drained (and its output yielded) before the isolated component runs.
        assert {"other": {"value": 99}} in results
        assert {"doubler": {"value": 6}} in results
        assert results.index({"other": {"value": 99}}) < results.index({"doubler": {"value": 6}})
        assert state["running_tasks"] == {}
        assert "other" not in state["scheduled_components"]

    @pytest.mark.asyncio
    async def test_skips_when_component_already_scheduled(self):
        pp = Pipeline()
        pp.add_component("doubler", _Doubler())
        state = _build_isolation_state(pp, {"doubler": {"value": 3}})
        state["scheduled_components"].add("doubler")

        results = [out async for out in pp._run_component_in_isolation(component_name="doubler", **state)]

        # Already scheduled: the component is not run.
        assert results == []
        assert state["component_visits"]["doubler"] == 0
        assert state["pipeline_outputs"] == {}
        assert "doubler" in state["scheduled_components"]

    @pytest.mark.asyncio
    async def test_distributes_outputs_downstream_and_prunes_consumed(self):
        pp = Pipeline()
        pp.add_component("first", _Doubler())
        pp.add_component("second", _Doubler())
        pp.connect("first.value", "second.value")
        state = _build_isolation_state(pp, {"first": {"value": 3}})

        results = [out async for out in pp._run_component_in_isolation(component_name="first", **state)]

        # `first`'s output is consumed by `second`, so it is pruned: nothing is yielded or stored as a pipeline output.
        assert results == []
        assert state["pipeline_outputs"] == {}
        # `second` can now consume the distributed value.
        second = pp._get_component_with_graph_metadata_and_visits("second", 0)
        assert pp._consume_component_inputs("second", second, state["inputs"]) == {"value": 6}

    @pytest.mark.asyncio
    async def test_include_outputs_from_yields_even_when_consumed(self):
        pp = Pipeline()
        pp.add_component("first", _Doubler())
        pp.add_component("second", _Doubler())
        pp.connect("first.value", "second.value")
        state = _build_isolation_state(pp, {"first": {"value": 3}})
        state["include_outputs_from"] = {"first"}

        results = [out async for out in pp._run_component_in_isolation(component_name="first", **state)]

        # Even though `first`'s output is consumed by `second`, include_outputs_from forces it to be surfaced.
        assert results == [{"first": {"value": 6}}]
        assert state["pipeline_outputs"] == {"first": {"value": 6}}


class TestInFlightTaskCleanupOnError:
    @pytest.mark.asyncio
    async def test_sibling_tasks_cancelled_when_a_component_errors(self):
        """When a component fails, the other in-flight tasks must be cancelled and not leak."""
        slow_started = asyncio.Event()
        slow_cancelled = False

        @component
        class Slow:
            @component.output_types(value=str)
            def run(self, text: str) -> dict[str, str]:
                return {"value": text}

            @component.output_types(value=str)
            async def run_async(self, text: str) -> dict[str, str]:
                nonlocal slow_cancelled
                slow_started.set()
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    slow_cancelled = True
                    raise
                return {"value": text}

        @component
        class Failing:
            @component.output_types(value=str)
            def run(self, text: str) -> dict[str, str]:
                raise RuntimeError("boom")

            @component.output_types(value=str)
            async def run_async(self, text: str) -> dict[str, str]:
                # Fail only once the sibling is actually running, so there is an in-flight task to clean up.
                await slow_started.wait()
                raise RuntimeError("boom")

        pp = Pipeline()
        pp.add_component("slow", Slow())
        pp.add_component("failing", Failing())

        with pytest.raises(PipelineRuntimeError):
            await pp.run_async({"slow": {"text": "x"}, "failing": {"text": "y"}}, concurrency_limit=2)

        assert slow_cancelled is True

    @pytest.mark.asyncio
    async def test_in_flight_tasks_cancelled_when_generator_iteration_is_abandoned(self):
        """When the consumer stops iterating run_async_generator early, in-flight tasks must be cancelled."""
        slow_started = asyncio.Event()
        slow_cancelled = False

        @component
        class Fast:
            @component.output_types(value=str)
            def run(self, text: str) -> dict[str, str]:
                return {"value": text}

            @component.output_types(value=str)
            async def run_async(self, text: str) -> dict[str, str]:
                # Yield an output only once the sibling is actually running, so it is in flight when we abandon.
                await slow_started.wait()
                return {"value": text}

        @component
        class Slow:
            @component.output_types(value=str)
            def run(self, text: str) -> dict[str, str]:
                return {"value": text}

            @component.output_types(value=str)
            async def run_async(self, text: str) -> dict[str, str]:
                nonlocal slow_cancelled
                slow_started.set()
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    slow_cancelled = True
                    raise
                return {"value": text}

        pp = Pipeline()
        pp.add_component("fast", Fast())
        pp.add_component("slow", Slow())

        generator = pp.run_async_generator({"fast": {"text": "x"}, "slow": {"text": "y"}}, concurrency_limit=2)
        async for _partial in generator:
            break  # abandon iteration after the first partial output
        await generator.aclose()

        assert slow_cancelled is True


@pytest.mark.asyncio
async def test_sync_component_run_in_thread_receives_contextvars():
    """
    Regression test: contextvars set in the calling async context (e.g. the active tracing span) must propagate
    to sync-only components, which the async run path dispatches to a thread. `asyncio.to_thread` guarantees this by
    copying the current context; a plain `loop.run_in_executor` would not.
    """

    @component
    class SyncContextVarReader:
        @component.output_types(value=str)
        def run(self, text: str) -> dict[str, str]:
            # Read inside the executor thread — only visible if the calling context was copied
            return {"value": _test_context_var.get()}

    pp = Pipeline()
    pp.add_component("reader", SyncContextVarReader())

    _test_context_var.set("propagated")
    result = await pp.run_async({"reader": {"text": "irrelevant"}})

    assert result["reader"]["value"] == "propagated"


@pytest.mark.asyncio
async def test_run_async_raises_when_multi_element_list_is_unwrapped_at_runtime():
    @component
    class MultiStrProducer:
        @component.output_types(texts=list[str])
        def run(self) -> dict[str, list[str]]:
            return {"texts": ["first", "second", "third"]}

    @component
    class SingleStrConsumer:
        @component.output_types(out=str)
        def run(self, text: str) -> dict[str, str]:
            return {"out": text}

    pipe = Pipeline()
    pipe.add_component("producer", MultiStrProducer())
    pipe.add_component("consumer", SingleStrConsumer())
    pipe.connect("producer.texts", "consumer.text")

    with pytest.raises(PipelineRuntimeError, match="Cannot unwrap a list of 3 items"):
        await pipe.run_async({})


@pytest.mark.asyncio
async def test_run_async_does_not_double_wrap_a_nested_pipeline_runtime_error():
    """
    A component that internally runs a pipeline and lets a PipelineRuntimeError escape (e.g. an Agent
    carrying a snapshot) should have that error propagate unchanged, not wrapped in another
    PipelineRuntimeError. This matches the synchronous _run_component.
    """
    from haystack.core.errors import PipelineRuntimeError

    inner_error = PipelineRuntimeError(component_name="inner", component_type=None, message="inner failure")

    @component
    class NestedPipelineComponent:
        @component.output_types(value=str)
        def run(self, text: str) -> dict[str, str]:  # pragma: no cover - async path is under test
            raise inner_error

        @component.output_types(value=str)
        async def run_async(self, text: str) -> dict[str, str]:
            raise inner_error

    pp = Pipeline()
    pp.add_component("nested", NestedPipelineComponent())

    with pytest.raises(PipelineRuntimeError) as exc_info:
        await pp.run_async({"nested": {"text": "x"}})

    assert exc_info.value is inner_error
    assert not isinstance(exc_info.value.__cause__, PipelineRuntimeError)


@pytest.mark.asyncio
async def test_run_async_lets_a_nested_breakpoint_exception_bubble_up():
    """A BreakpointException raised by a nested component must reach the caller as-is, not wrapped."""
    from haystack.core.errors import BreakpointException
    from haystack.dataclasses.breakpoints import Breakpoint

    break_point = Breakpoint(component_name="inner", visit_count=0)
    breakpoint_error = BreakpointException.from_triggered_breakpoint(break_point)

    @component
    class BreakpointingComponent:
        @component.output_types(value=str)
        def run(self, text: str) -> dict[str, str]:  # pragma: no cover - async path is under test
            raise breakpoint_error

        @component.output_types(value=str)
        async def run_async(self, text: str) -> dict[str, str]:
            raise breakpoint_error

    pp = Pipeline()
    pp.add_component("bp", BreakpointingComponent())

    with pytest.raises(BreakpointException) as exc_info:
        await pp.run_async({"bp": {"text": "x"}})

    assert exc_info.value is breakpoint_error
