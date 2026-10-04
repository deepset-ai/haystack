# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import json
import logging
from dataclasses import replace

import pytest

from haystack import Document, Pipeline, component
from haystack.components.joiners import BranchJoiner
from haystack.core.errors import BreakpointException, PipelineInvalidPipelineSnapshotError, PipelineRuntimeError
from haystack.core.pipeline.breakpoint import HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, load_pipeline_snapshot
from haystack.dataclasses.breakpoints import INTERNAL_INPUTS_FORMAT, Breakpoint, PipelineSnapshot, PipelineState
from haystack.utils import _serialize_value_with_schema

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


@component
class _Doubler:
    """Minimal component used to exercise the isolation helper."""

    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, int]:
        return {"value": value * 2}


@component
class _AsyncBreakpointCounter:
    def __init__(self):
        self.values = []

    @component.output_types(retry=int, done=int)
    def run(self, value: int) -> dict[str, int]:
        self.values.append(value)
        return {"retry" if value < 3 else "done": value + 1}

    @component.output_types(retry=int, done=int)
    async def run_async(self, value: int) -> dict[str, int]:
        return self.run(value)


async def _run_async_api(pipeline, run_method, **kwargs):
    if run_method == "run_async":
        return await pipeline.run_async(**kwargs)
    outputs = [output async for output in pipeline.run_async_generator(**kwargs)]
    return outputs[-1]


@pytest.mark.parametrize("close_early", [False, True])
async def test_breakpoint_drain_yields_fast_sibling_before_slow_finishes(close_early):
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    slow_finished = asyncio.Event()
    slow_cancelled = asyncio.Event()
    snapshots: list[PipelineSnapshot] = []

    @component
    class Fast:
        @component.output_types(value=int)
        def run(self, value: int) -> dict[str, int]:
            return {"value": value}

        @component.output_types(value=int)
        async def run_async(self, value: int) -> dict[str, int]:
            await slow_started.wait()
            return self.run(value)

    @component
    class Slow:
        @component.output_types(value=int)
        def run(self, value: int) -> dict[str, int]:
            return {"value": value}

        @component.output_types(value=int)
        async def run_async(self, value: int) -> dict[str, int]:
            slow_started.set()
            try:
                await release_slow.wait()
            except asyncio.CancelledError:
                slow_cancelled.set()
                raise
            slow_finished.set()
            return self.run(value)

    pipeline = Pipeline()
    pipeline.add_component("a_fast", Fast())
    pipeline.add_component("b_slow", Slow())
    pipeline.add_component("c_target", _Doubler())
    generator = pipeline.run_async_generator(
        {"value": 1}, break_point=Breakpoint("c_target"), snapshot_callback=snapshots.append
    )
    try:
        assert await asyncio.wait_for(anext(generator), timeout=1) == {"a_fast": {"value": 1}}
        assert not slow_finished.is_set()
        assert snapshots == []
        if close_early:
            await generator.aclose()
            assert slow_cancelled.is_set()
            assert snapshots == []
            return
        release_slow.set()
        assert await asyncio.wait_for(anext(generator), timeout=1) == {"b_slow": {"value": 1}}
        with pytest.raises(BreakpointException) as caught:
            await anext(generator)
        assert slow_finished.is_set()
        assert not slow_cancelled.is_set()
        snapshot = caught.value.pipeline_snapshot
        assert snapshot is not None
        assert snapshots == [snapshot]
        assert snapshot.pipeline_state.component_visits == {"a_fast": 1, "b_slow": 1, "c_target": 0}
    finally:
        await generator.aclose()


@pytest.mark.parametrize("run_method", ["run_async", "run_async_generator"])
class TestAsyncBreakpoints:
    @pytest.mark.parametrize("target", ["first", "second", "third"])
    async def test_snapshot_roundtrip_and_reuse(self, run_method, target, tmp_path, monkeypatch, spying_tracer):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "true")
        pipeline = Pipeline()
        for name in ("first", "second", "third"):
            pipeline.add_component(name, _Doubler())
        pipeline.connect("first.value", "second.value")
        pipeline.connect("second.value", "third.value")
        with pytest.raises(BreakpointException) as caught:
            await _run_async_api(
                pipeline,
                run_method,
                data={"first": {"value": 1}},
                include_outputs_from={"first", "second"},
                break_point=Breakpoint(target, snapshot_file_path=str(tmp_path)),
            )

        assert caught.value.pipeline_snapshot_file_path is not None
        snapshot = load_pipeline_snapshot(caught.value.pipeline_snapshot_file_path)
        assert snapshot.pipeline_state.inputs_format == INTERNAL_INPUTS_FORMAT
        assert snapshot.pipeline_state.component_visits[target] == 0
        original_snapshot = snapshot.to_dict()
        for _ in range(2):
            result = await _run_async_api(
                pipeline, run_method, data={"invalid": {}}, pipeline_snapshot=snapshot, include_outputs_from=set()
            )
            assert result == {"first": {"value": 2}, "second": {"value": 4}, "third": {"value": 8}}
            assert snapshot.to_dict() == original_snapshot

        component_spans = [span for span in spying_tracer.spans if span.operation_name == "haystack.component.run"]
        before_target = {"first": 0, "second": 1, "third": 2}[target]
        assert len(component_spans) == before_target + 2 * (3 - before_target)

    @pytest.mark.parametrize("native_async", [True, False])
    async def test_drains_siblings_before_snapshot_and_does_not_run_them_again(self, run_method, native_async):
        calls = []
        snapshots = []

        @component
        class SyncSibling:
            @component.output_types(value=int)
            def run(self, value: int) -> dict[str, int]:
                calls.append(value)
                return {"value": value * 2}

        @component
        class AsyncSibling:
            @component.output_types(value=int)
            def run(self, value: int) -> dict[str, int]:
                calls.append(value)
                return {"value": value * 2}

            @component.output_types(value=int)
            async def run_async(self, value: int) -> dict[str, int]:
                await asyncio.sleep(0)
                return self.run(value)

        pipeline = Pipeline()
        pipeline.add_component("a_sibling", AsyncSibling() if native_async else SyncSibling())
        pipeline.add_component("b_target", _Doubler())
        pipeline.add_component("c_later", _Doubler())
        data = {"a_sibling": {"value": 1}, "b_target": {"value": 2}, "c_later": {"value": 3}}

        def save(snapshot):
            snapshots.append(snapshot)
            assert calls == [1]
            return "saved-snapshot"

        partials = []
        with pytest.raises(BreakpointException) as caught:
            if run_method == "run_async_generator":
                async for partial in pipeline.run_async_generator(
                    data, concurrency_limit=3, break_point=Breakpoint("b_target"), snapshot_callback=save
                ):
                    partials.append(partial)
            else:
                await pipeline.run_async(
                    data, concurrency_limit=3, break_point=Breakpoint("b_target"), snapshot_callback=save
                )

        snapshot = caught.value.pipeline_snapshot
        assert snapshot is not None
        assert snapshots == [snapshot]
        assert caught.value.pipeline_snapshot_file_path == "saved-snapshot"
        assert snapshot.pipeline_state.component_visits == {"a_sibling": 1, "b_target": 0, "c_later": 0}
        assert snapshot.pipeline_state.inputs["serialized_data"]["b_target"]["value"] == [{"sender": None, "value": 2}]
        if run_method == "run_async_generator":
            assert partials == [{"a_sibling": {"value": 2}}]
        snapshot = PipelineSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
        assert await _run_async_api(pipeline, run_method, data={}, pipeline_snapshot=snapshot) == {
            "a_sibling": {"value": 2},
            "b_target": {"value": 4},
            "c_later": {"value": 6},
        }
        assert calls == [1]

    async def test_snapshot_restores_document_inputs(self, run_method):
        @component
        class Echo:
            @component.output_types(document=Document)
            def run(self, document: Document) -> dict[str, Document]:
                return {"document": document}

        pipeline = Pipeline()
        pipeline.add_component("first", Echo())
        pipeline.add_component("second", Echo())
        pipeline.connect("first.document", "second.document")
        document = Document(content="snapshot content", meta={"tags": ["example"]})
        with pytest.raises(BreakpointException) as caught:
            await _run_async_api(
                pipeline, run_method, data={"first": {"document": document}}, break_point=Breakpoint("second")
            )
        assert caught.value.pipeline_snapshot is not None
        snapshot = PipelineSnapshot.from_dict(json.loads(json.dumps(caught.value.pipeline_snapshot.to_dict())))
        assert await _run_async_api(pipeline, run_method, data={}, pipeline_snapshot=snapshot) == {
            "second": {"document": document}
        }

    async def test_loop_visits_and_step_to_later_breakpoint(self, run_method):
        pipeline = Pipeline()
        counter = _AsyncBreakpointCounter()
        pipeline.add_component("joiner", BranchJoiner(int))
        pipeline.add_component("counter", counter)
        pipeline.connect("joiner.value", "counter.value")
        pipeline.connect("counter.retry", "joiner.value")
        with pytest.raises(BreakpointException) as caught:
            await _run_async_api(
                pipeline, run_method, data={"joiner": {"value": 0}}, break_point=Breakpoint("joiner", 2)
            )
        assert counter.values == [0, 1]
        assert caught.value.pipeline_snapshot is not None
        first_snapshot = PipelineSnapshot.from_dict(caught.value.pipeline_snapshot.to_dict())
        first_state = first_snapshot.to_dict()
        assert first_snapshot.pipeline_state.component_visits == {"joiner": 2, "counter": 2}
        with pytest.raises(BreakpointException) as caught:
            await _run_async_api(
                pipeline, run_method, data={}, pipeline_snapshot=first_snapshot, break_point=Breakpoint("counter", 3)
            )
        assert counter.values == [0, 1, 2]
        assert first_snapshot.to_dict() == first_state
        assert await _run_async_api(
            pipeline, run_method, data={}, pipeline_snapshot=caught.value.pipeline_snapshot
        ) == {"counter": {"done": 4}}
        assert counter.values == [0, 1, 2, 3]

    async def test_sync_snapshot_resumes_and_rejects_same_breakpoint(self, run_method):
        pipeline = Pipeline()
        pipeline.add_component("first", _Doubler())
        pipeline.add_component("second", _Doubler())
        pipeline.connect("first.value", "second.value")
        with pytest.raises(BreakpointException) as caught:
            pipeline.run({"first": {"value": 2}}, break_point=Breakpoint("second"))
        snapshot = caught.value.pipeline_snapshot
        with pytest.raises(PipelineInvalidPipelineSnapshotError, match="same component and visit count"):
            await _run_async_api(
                pipeline, run_method, data={}, pipeline_snapshot=snapshot, break_point=Breakpoint("second")
            )
        assert await _run_async_api(pipeline, run_method, data={}, pipeline_snapshot=snapshot) == {
            "second": {"value": 8}
        }

        pipeline.add_component("extra", _Doubler())
        with pytest.raises(PipelineInvalidPipelineSnapshotError, match="not present in 'ordered_component_names'"):
            await _run_async_api(pipeline, run_method, data={}, pipeline_snapshot=snapshot)

    async def test_invalid_and_unreached_breakpoints(self, run_method, caplog):
        pipeline = Pipeline()
        pipeline.add_component("doubler", _Doubler())
        with pytest.raises(ValueError, match="not a registered component"):
            await _run_async_api(pipeline, run_method, data={}, break_point=Breakpoint("missing"))
        with caplog.at_level(logging.WARNING):
            assert await _run_async_api(
                pipeline, run_method, data={"value": 1}, break_point=Breakpoint("doubler", 2)
            ) == {"doubler": {"value": 2}}
        assert "was never triggered" in caplog.text

    async def test_legacy_snapshot_input_handling_applies_only_to_first_visit(self, run_method):
        pipeline = Pipeline()
        counter = _AsyncBreakpointCounter()
        pipeline.add_component("joiner", BranchJoiner(int))
        pipeline.add_component("counter", counter)
        pipeline.connect("joiner.value", "counter.value")
        pipeline.connect("counter.retry", "joiner.value")
        snapshot = PipelineSnapshot(
            pipeline_state=PipelineState(
                inputs=_serialize_value_with_schema({"joiner": {"value": [0]}, "counter": {}}),
                component_visits={"joiner": 0, "counter": 0},
                pipeline_outputs=_serialize_value_with_schema({}),
            ),
            break_point=Breakpoint("joiner"),
            original_input_data=_serialize_value_with_schema({"joiner": {"value": 0}}),
            ordered_component_names=["counter", "joiner"],
            include_outputs_from=set(),
        )
        assert await _run_async_api(pipeline, run_method, data={}, pipeline_snapshot=snapshot) == {
            "counter": {"done": 4}
        }
        assert counter.values == [0, 1, 2, 3]

    async def test_error_snapshot_restores_failed_and_cancelled_inputs(self, run_method, spying_tracer):
        slow_started = asyncio.Event()
        cancelled = asyncio.Event()
        snapshots: list[PipelineSnapshot] = []
        fail_once = True

        @component
        class Failing:
            @component.output_types(value=int)
            def run(self, value: int) -> dict[str, int]:
                return {"value": value * 2}

            @component.output_types(value=int)
            async def run_async(self, value: int) -> dict[str, int]:
                nonlocal fail_once
                if fail_once:
                    await slow_started.wait()
                    fail_once = False
                    raise RuntimeError("boom")
                return self.run(value)

        @component
        class Slow:
            @component.output_types(value=int)
            def run(self, value: int) -> dict[str, int]:
                return {"value": value * 2}

            @component.output_types(value=int)
            async def run_async(self, value: int) -> dict[str, int]:
                if not slow_started.is_set():
                    slow_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                return self.run(value)

        pipeline = Pipeline()
        pipeline.add_component("source", _Doubler())
        pipeline.add_component("failing", Failing())
        pipeline.add_component("slow", Slow())
        pipeline.connect("source.value", "failing.value")
        pipeline.connect("source.value", "slow.value")
        with pytest.raises(PipelineRuntimeError, match="boom") as caught:
            await _run_async_api(
                pipeline,
                run_method,
                data={"source": {"value": 1}},
                include_outputs_from={"source"},
                snapshot_callback=snapshots.append,
            )
        assert cancelled.is_set()
        snapshot = caught.value.pipeline_snapshot
        assert snapshot is not None
        assert snapshots == [snapshot]
        assert snapshot.break_point.component_name == "failing"
        assert snapshot.pipeline_state.component_visits == {"source": 1, "failing": 0, "slow": 0}
        for name in ("failing", "slow"):
            assert snapshot.pipeline_state.inputs["serialized_data"][name]["value"] == [
                {"sender": "source", "value": 2}
            ]
        assert await _run_async_api(pipeline, run_method, data={}, pipeline_snapshot=snapshot) == {
            "source": {"value": 2},
            "failing": {"value": 4},
            "slow": {"value": 4},
        }
        source_spans = [span for span in spying_tracer.spans if span.tags.get("haystack.component.name") == "source"]
        assert len(source_spans) == 1

    async def test_component_error_during_breakpoint_drain_takes_precedence(self, run_method):
        error = PipelineRuntimeError(component_name="inner", component_type=None, message="inner failure")
        slow_started = asyncio.Event()
        slow_cancelled = asyncio.Event()

        @component
        class Failing:
            @component.output_types(value=int)
            def run(self, value: int) -> dict[str, int]:
                raise error

            @component.output_types(value=int)
            async def run_async(self, value: int) -> dict[str, int]:
                await slow_started.wait()
                raise error

        @component
        class Slow:
            @component.output_types(value=int)
            def run(self, value: int) -> dict[str, int]:
                return {"value": value}

            @component.output_types(value=int)
            async def run_async(self, value: int) -> dict[str, int]:
                slow_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    slow_cancelled.set()
                    raise
                return self.run(value)

        pipeline = Pipeline()
        pipeline.add_component("a_failing", Failing())
        pipeline.add_component("b_slow", Slow())
        pipeline.add_component("c_target", _Doubler())
        with pytest.raises(PipelineRuntimeError) as caught:
            await asyncio.wait_for(
                _run_async_api(pipeline, run_method, data={"value": 1}, break_point=Breakpoint("c_target")), timeout=1
            )
        assert caught.value is error
        assert slow_cancelled.is_set()
        assert error.pipeline_snapshot is not None
        assert error.pipeline_snapshot.break_point == Breakpoint(
            "a_failing", snapshot_file_path=error.pipeline_snapshot.break_point.snapshot_file_path
        )

    async def test_cancellation_while_draining_breakpoint_cleans_up(self, run_method):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        @component
        class Slow:
            @component.output_types(value=int)
            def run(self, value: int) -> dict[str, int]:
                return {"value": value}

            @component.output_types(value=int)
            async def run_async(self, value: int) -> dict[str, int]:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
                return {"value": value}

        pipeline = Pipeline()
        pipeline.add_component("a_slow", Slow())
        pipeline.add_component("b_target", _Doubler())
        snapshots: list[PipelineSnapshot] = []
        task = asyncio.create_task(
            _run_async_api(
                pipeline,
                run_method,
                data={"value": 1},
                break_point=Breakpoint("b_target"),
                snapshot_callback=snapshots.append,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert snapshots == []


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
    assert inner_error.pipeline_snapshot is not None
    assert inner_error.pipeline_snapshot.break_point.component_name == "nested"


@pytest.mark.asyncio
async def test_run_async_lets_a_nested_breakpoint_exception_bubble_up():
    """A BreakpointException raised by a nested component must reach the caller as-is, not wrapped."""
    from haystack.core.errors import BreakpointException
    from haystack.dataclasses.breakpoints import Breakpoint

    break_point = Breakpoint(component_name="inner", visit_count=1)
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
    # The nested breakpoint may happen to name another component in the outer graph.
    pp.add_component("inner", _Doubler())

    with pytest.raises(BreakpointException) as exc_info:
        await pp.run_async(
            {"bp": {"text": "x"}, "inner": {"value": 1}}, break_point=Breakpoint(component_name="inner", visit_count=1)
        )

    assert exc_info.value is breakpoint_error
    assert breakpoint_error.break_point == break_point
    assert breakpoint_error.pipeline_snapshot is not None
    assert breakpoint_error.pipeline_snapshot.break_point.component_name == "bp"
