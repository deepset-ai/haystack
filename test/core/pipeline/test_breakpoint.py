# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import replace
from typing import Any

import pytest

from haystack import component
from haystack.components.joiners import BranchJoiner, ListJoiner
from haystack.components.routers import ConditionalRouter
from haystack.components.routers.conditional_router import Route
from haystack.core.errors import BreakpointException, PipelineInvalidPipelineSnapshotError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.breakpoint import (
    HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED,
    _create_pipeline_snapshot,
    _is_snapshot_save_enabled,
    _save_pipeline_snapshot,
    _transform_json_structure,
    load_pipeline_snapshot,
)
from haystack.core.pipeline.component_checks import _NoOutputProduced
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import INTERNAL_INPUTS_FORMAT, Breakpoint, PipelineSnapshot, PipelineState
from haystack.utils import _deserialize_value_with_schema
from haystack.utils.base_serialization import _serialize_value_with_schema

_EMPTY_OBJECT_PAYLOAD = {"serialization_schema": {"type": "object", "properties": {}}, "serialized_data": {}}


def test_transform_json_structure_unwraps_sender_value():
    data = {
        "key1": [{"sender": None, "value": "some value"}],
        "key2": [{"sender": "comp1", "value": 42}],
        "key3": "direct value",
    }

    result = _transform_json_structure(data)

    assert result == {"key1": "some value", "key2": 42, "key3": "direct value"}


def test_transform_json_structure_handles_nested_structures():
    data = {
        "key1": [{"sender": None, "value": "value1"}],
        "key2": {"nested": [{"sender": "comp1", "value": "value2"}], "direct": "value3"},
        "key3": [[{"sender": None, "value": "value4"}], [{"sender": "comp2", "value": "value5"}]],
    }

    result = _transform_json_structure(data)

    assert result == {"key1": "value1", "key2": {"nested": "value2", "direct": "value3"}, "key3": ["value4", "value5"]}


def test_load_pipeline_snapshot_loads_valid_snapshot(tmp_path):
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {"inputs": {}, "component_visits": {"comp1": 0, "comp2": 0}, "pipeline_outputs": {}},
        "original_input_data": {},
        "ordered_component_names": ["comp1", "comp2"],
        "include_outputs_from": ["comp1", "comp2"],
    }
    pipeline_snapshot_file = tmp_path / "state.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    loaded_snapshot = load_pipeline_snapshot(pipeline_snapshot_file)
    assert loaded_snapshot == PipelineSnapshot.from_dict(pipeline_snapshot)


def test_load_state_handles_invalid_state(tmp_path):
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {"inputs": {}, "component_visits": {"comp1": 0, "comp2": 0}, "pipeline_outputs": {}},
        "original_input_data": {},
        "include_outputs_from": ["comp1", "comp2"],
        "ordered_component_names": ["comp1", "comp3"],  # inconsistent with component_visits
    }

    pipeline_snapshot_file = tmp_path / "invalid_pipeline_snapshot.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    with pytest.raises(ValueError, match="Invalid pipeline snapshot from"):
        load_pipeline_snapshot(pipeline_snapshot_file)


def test_breakpoint_saves_intermediate_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "true")

    @component
    class SimpleComponent:
        @component.output_types(result=str)
        def run(self, input_value: str) -> dict[str, str]:
            return {"result": f"processed_{input_value}"}

    pipeline = Pipeline()
    comp1 = SimpleComponent()
    comp2 = SimpleComponent()
    pipeline.add_component("comp1", comp1)
    pipeline.add_component("comp2", comp2)
    pipeline.connect("comp1", "comp2")

    # breakpoint on comp2
    break_point = Breakpoint(component_name="comp2", visit_count=0, snapshot_file_path=str(tmp_path))

    # run with include_outputs_from to capture intermediate outputs
    with pytest.raises(BreakpointException) as exc_info:
        pipeline.run(data={"comp1": {"input_value": "test"}}, include_outputs_from={"comp1"}, break_point=break_point)

    # breakpoint should be triggered
    assert exc_info.value.component == "comp2"

    # verify snapshot file contains the intermediate outputs
    snapshot_files = list(tmp_path.glob("comp2_*.json"))
    assert len(snapshot_files) == 1, f"Expected exactly one snapshot file, found {len(snapshot_files)}"

    snapshot_file = snapshot_files[0]
    loaded_snapshot = load_pipeline_snapshot(snapshot_file)

    # verify the snapshot contains the intermediate outputs from comp1
    assert loaded_snapshot.pipeline_state.pipeline_outputs == (
        {
            "serialization_schema": {
                "type": "object",
                "properties": {"comp1": {"type": "object", "properties": {"result": {"type": "string"}}}},
            },
            "serialized_data": {"comp1": {"result": "processed_test"}},
        }
    )

    # verify the saved inputs record which component sent each one, in the order it arrived.
    # The accompanying schema is asserted in TestCreatePipelineSnapshot.
    assert loaded_snapshot.pipeline_state.inputs_format == INTERNAL_INPUTS_FORMAT
    assert loaded_snapshot.pipeline_state.inputs["serialized_data"] == {
        # comp1 was given its input from outside the pipeline
        "comp1": {"input_value": [{"sender": None, "value": "test"}]},
        # comp2 was given its input by comp1, and had not consumed it yet when the breakpoint hit
        "comp2": {"input_value": [{"sender": "comp1", "value": "processed_test"}]},
    }

    # verify the whole pipeline state contains the expected data
    assert loaded_snapshot.pipeline_state.component_visits["comp1"] == 1
    assert loaded_snapshot.pipeline_state.component_visits["comp2"] == 0
    assert "comp1" in loaded_snapshot.include_outputs_from
    assert isinstance(loaded_snapshot.break_point, Breakpoint)
    assert loaded_snapshot.break_point.component_name == "comp2"
    assert loaded_snapshot.break_point.visit_count == 0


@component
class _AppendingComponent:
    @component.output_types(result=str)
    def run(self, input_value: str) -> dict[str, str]:
        return {"result": f"{input_value}_processed"}


@component
class _CountUpTo:
    def __init__(self, limit: int) -> None:
        self.limit = limit

    @component.output_types(retry=int, done=str)
    def run(self, value: int) -> dict[str, Any]:
        if value < self.limit:
            return {"retry": value + 1}
        return {"done": f"finished at {value}"}


def _three_component_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("comp1", _AppendingComponent())
    pipeline.add_component("comp2", _AppendingComponent())
    pipeline.add_component("comp3", _AppendingComponent())
    pipeline.connect("comp1", "comp2")
    pipeline.connect("comp2", "comp3")
    return pipeline


def _looping_pipeline() -> Pipeline:
    pipeline = Pipeline(max_runs_per_component=20)
    pipeline.add_component("joiner", BranchJoiner(int))
    pipeline.add_component("counter", _CountUpTo(limit=5))
    pipeline.connect("joiner.value", "counter.value")
    pipeline.connect("counter.retry", "joiner.value")
    return pipeline


class TestResumeFromPipelineSnapshot:
    def test_break_point_with_pipeline_snapshot_steps_through_pipeline(self):
        pipeline = _three_component_pipeline()

        # run until the breakpoint on comp2
        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run(data={"comp1": {"input_value": "test"}}, break_point=Breakpoint(component_name="comp2"))
        first_snapshot = exc_info.value.pipeline_snapshot
        assert first_snapshot is not None
        assert first_snapshot.pipeline_state.component_visits == {"comp1": 1, "comp2": 0, "comp3": 0}

        # step: resume from the snapshot and pause again at comp3
        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run(data={}, pipeline_snapshot=first_snapshot, break_point=Breakpoint(component_name="comp3"))
        second_snapshot = exc_info.value.pipeline_snapshot
        assert second_snapshot is not None
        assert second_snapshot.pipeline_state.component_visits == {"comp1": 1, "comp2": 1, "comp3": 0}

        # resume from the second snapshot and run to completion
        result = pipeline.run(data={}, pipeline_snapshot=second_snapshot)
        assert result["comp3"]["result"] == "test_processed_processed_processed"

    def test_break_point_on_earlier_component_than_pipeline_snapshot_never_triggers(self):
        pipeline = _three_component_pipeline()

        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run(data={"comp1": {"input_value": "test"}}, break_point=Breakpoint(component_name="comp2"))
        snapshot = exc_info.value.pipeline_snapshot

        # comp1 already ran before the snapshot was taken, so a breakpoint on it never triggers
        # and the resumed run completes normally
        result = pipeline.run(data={}, pipeline_snapshot=snapshot, break_point=Breakpoint(component_name="comp1"))
        assert result["comp3"]["result"] == "test_processed_processed_processed"

    def test_break_point_matching_pipeline_snapshot_break_point_raises(self):
        pipeline = _three_component_pipeline()

        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run(data={"comp1": {"input_value": "test"}}, break_point=Breakpoint(component_name="comp2"))
        snapshot = exc_info.value.pipeline_snapshot

        with pytest.raises(PipelineInvalidPipelineSnapshotError, match="different component or visit count"):
            pipeline.run(
                data={}, pipeline_snapshot=snapshot, break_point=Breakpoint(component_name="comp2", visit_count=0)
            )

    @pytest.mark.parametrize("visit_count", [0, 1, 2, 3])
    def test_break_point_in_loop_resumes_on_any_visit(self, visit_count):
        """A component paused on a later visit of a loop must still resume and finish the loop."""
        with pytest.raises(BreakpointException) as exc_info:
            _looping_pipeline().run(
                {"joiner": {"value": 0}}, break_point=Breakpoint(component_name="joiner", visit_count=visit_count)
            )
        snapshot = exc_info.value.pipeline_snapshot
        assert snapshot is not None
        assert snapshot.pipeline_state.component_visits["joiner"] == visit_count

        # The loop runs to `_CountUpTo(limit=5)` regardless of where it was paused.
        assert _looping_pipeline().run(data={}, pipeline_snapshot=snapshot) == {"counter": {"done": "finished at 5"}}

    def test_snapshot_preserves_sockets_whose_sender_produced_no_output(self):
        """A mixed socket queue containing a value and `_NoOutputProduced()` survives a snapshot and resume."""
        routes: list[Route] = [
            {"condition": "{{ n > 5 }}", "output": "{{ ['big'] }}", "output_name": "big", "output_type": list[str]},
            {
                "condition": "{{ n <= 5 }}",
                "output": "{{ ['small'] }}",
                "output_name": "small",
                "output_type": list[str],
            },
        ]
        pipeline = Pipeline()
        pipeline.add_component("router", ConditionalRouter(routes=routes))
        pipeline.add_component("collect", ListJoiner(list[str]))
        pipeline.connect("router.big", "collect.values")
        pipeline.connect("router.small", "collect.values")

        expected = pipeline.run({"router": {"n": 9}})

        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run({"router": {"n": 9}}, break_point=Breakpoint(component_name="collect"))
        snapshot = exc_info.value.pipeline_snapshot
        assert snapshot is not None

        socket_schema = snapshot.pipeline_state.inputs["serialization_schema"]["properties"]["collect"]["properties"][
            "values"
        ]
        assert "prefixItems" in socket_schema

        restored = _deserialize_value_with_schema(snapshot.pipeline_state.inputs)
        restored_values = [entry["value"] for entry in restored["collect"]["values"]]
        assert ["big"] in restored_values
        assert any(isinstance(value, _NoOutputProduced) for value in restored_values)
        assert pipeline.run(data={}, pipeline_snapshot=snapshot) == expected


class TestResumeFromLegacyPipelineSnapshot:
    def test_resume_from_legacy_snapshot_without_sender_information(self):
        """Snapshots taken before the sender was recorded store flattened values and must still resume."""
        pipeline = _three_component_pipeline()

        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run(data={"comp1": {"input_value": "test"}}, break_point=Breakpoint(component_name="comp2"))
        snapshot = exc_info.value.pipeline_snapshot
        assert snapshot is not None

        legacy_inputs = _serialize_value_with_schema(
            _transform_json_structure(_deserialize_value_with_schema(snapshot.pipeline_state.inputs))
        )
        assert legacy_inputs["serialized_data"]["comp2"] == {"input_value": "test_processed"}
        legacy_snapshot = replace(
            snapshot, pipeline_state=replace(snapshot.pipeline_state, inputs=legacy_inputs, inputs_format=None)
        )

        result = pipeline.run(data={}, pipeline_snapshot=legacy_snapshot)
        assert result["comp3"]["result"] == "test_processed_processed_processed"

    def test_resume_from_legacy_snapshot_into_a_loop(self):
        """
        A legacy snapshot needs its special input handling for the visit it was paused on, and only that visit.

        The loop brings the paused component ordinary inputs again afterwards, which it has to consume the ordinary
        way. Keeping the special handling re-reads the restored input on every visit, so the loop never advances.

        The snapshot is written out literally rather than derived from a current one, because that is what a snapshot
        left over from an older Haystack looks like: a greedy socket stored the value it had already consumed.
        """
        legacy_snapshot = PipelineSnapshot(
            pipeline_state=PipelineState(
                inputs=_serialize_value_with_schema({"joiner": {"value": [0]}, "counter": {}}),
                component_visits={"joiner": 0, "counter": 0},
                pipeline_outputs=_serialize_value_with_schema({}),
                inputs_format=None,
            ),
            break_point=Breakpoint(component_name="joiner", visit_count=0),
            original_input_data=_serialize_value_with_schema({"joiner": {"value": 0}}),
            ordered_component_names=["counter", "joiner"],
            include_outputs_from=set(),
        )

        # The joiner is visited five more times after the resume.
        result = _looping_pipeline().run(data={}, pipeline_snapshot=legacy_snapshot)
        assert result == {"counter": {"done": "finished at 5"}}


class TestCreatePipelineSnapshot:
    def test_create_pipeline_snapshot_all_fields(self):
        break_point = Breakpoint(component_name="comp2")
        ordered_component_names = ["comp1", "comp2"]
        include_outputs_from = {"comp1"}

        snapshot = _create_pipeline_snapshot(
            inputs={"comp1": {"input_value": [{"sender": None, "value": "test"}]}, "comp2": {}},
            component_inputs={"input_value": [{"sender": "comp1", "value": "processed_test"}]},
            break_point=break_point,
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={"comp1": {"input_value": "test"}},
            ordered_component_names=ordered_component_names,
            include_outputs_from=include_outputs_from,
            pipeline_outputs={"comp1": {"result": "processed_test"}},
        )

        assert snapshot.original_input_data == {
            "serialization_schema": {
                "type": "object",
                "properties": {"comp1": {"type": "object", "properties": {"input_value": {"type": "string"}}}},
            },
            "serialized_data": {"comp1": {"input_value": "test"}},
        }
        assert snapshot.ordered_component_names == ordered_component_names
        assert snapshot.break_point == break_point
        assert snapshot.include_outputs_from == include_outputs_from

        # Each input a socket received is stored in a list. Mixed-type lists carry one schema per position.
        def socket_schema(sender_type: str) -> dict[str, Any]:
            return {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sender": {"type": sender_type}, "value": {"type": "string"}},
                },
            }

        assert snapshot.pipeline_state == PipelineState(
            inputs={
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "comp1": {"type": "object", "properties": {"input_value": socket_schema("null")}},
                        "comp2": {"type": "object", "properties": {"input_value": socket_schema("string")}},
                    },
                },
                "serialized_data": {
                    "comp1": {"input_value": [{"sender": None, "value": "test"}]},
                    "comp2": {"input_value": [{"sender": "comp1", "value": "processed_test"}]},
                },
            },
            component_visits={"comp1": 1, "comp2": 0},
            pipeline_outputs={
                "serialization_schema": {
                    "type": "object",
                    "properties": {"comp1": {"type": "object", "properties": {"result": {"type": "string"}}}},
                },
                "serialized_data": {"comp1": {"result": "processed_test"}},
            },
            inputs_format=INTERNAL_INPUTS_FORMAT,
        )

    def test_create_pipeline_snapshot_with_dataclasses_in_pipeline_outputs(self):
        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2"),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from={"comp1"},
            pipeline_outputs={"comp1": {"result": ChatMessage.from_user("hello")}},
        )

        assert snapshot.pipeline_state == PipelineState(
            inputs={
                "serialization_schema": {
                    "type": "object",
                    "properties": {"comp2": {"type": "object", "properties": {}}},
                },
                "serialized_data": {"comp2": {}},
            },
            component_visits={"comp1": 1, "comp2": 0},
            pipeline_outputs={
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "comp1": {
                            "type": "object",
                            "properties": {"result": {"type": "haystack.dataclasses.chat_message.ChatMessage"}},
                        }
                    },
                },
                "serialized_data": {
                    "comp1": {"result": {"role": "user", "meta": {}, "name": None, "content": [{"text": "hello"}]}}
                },
            },
            inputs_format=INTERNAL_INPUTS_FORMAT,
        )

    def test_create_pipeline_snapshot_non_serializable_inputs(self, caplog):
        class NonSerializable:
            def to_dict(self):
                raise TypeError("Cannot serialize")

        with caplog.at_level(logging.WARNING):
            _create_pipeline_snapshot(
                inputs={"comp1": {"input_value": [{"sender": None, "value": NonSerializable()}]}, "comp2": {}},
                component_inputs={},
                break_point=Breakpoint(component_name="comp2"),
                component_visits={"comp1": 1, "comp2": 0},
                original_input_data={"comp1": {"input_value": NonSerializable()}},
                ordered_component_names=["comp1", "comp2"],
                include_outputs_from={"comp1"},
                pipeline_outputs={},
            )

        assert any("Failed to serialize the inputs of the current pipeline state" in msg for msg in caplog.messages)
        assert any("Failed to serialize original input data for `pipeline.run`." in msg for msg in caplog.messages)

    def test_create_pipeline_snapshot_non_serializable_inputs_snapshot_is_resumable(self, caplog):
        """
        Guards against the same non-resumable snapshot regression fixed at the agent level: when
        top-level pipeline inputs/outputs contain non-serializable values, the snapshot fields
        must still round-trip through ``_deserialize_value_with_schema`` instead of failing with
        ``DeserializationError: ... Got: {}``. Serializable sibling components must stay intact.
        """

        class NonSerializable:
            def to_dict(self):
                raise TypeError("Cannot serialize")

        with caplog.at_level(logging.WARNING):
            snapshot = _create_pipeline_snapshot(
                inputs={
                    "comp1": {"input_value": [{"sender": None, "value": NonSerializable()}]},
                    "comp2": {"input_value": [{"sender": None, "value": "keep me"}]},
                },
                component_inputs={},
                break_point=Breakpoint(component_name="comp3"),
                component_visits={"comp1": 1, "comp2": 1, "comp3": 0},
                original_input_data={"comp1": {"input_value": NonSerializable()}},
                ordered_component_names=["comp1", "comp2", "comp3"],
                include_outputs_from=set(),
                pipeline_outputs={"comp1": {"result": NonSerializable()}},
            )

        # No DeserializationError on any of the three pipeline-level payloads.
        deserialized_inputs = _deserialize_value_with_schema(snapshot.pipeline_state.inputs)
        deserialized_original_input_data = _deserialize_value_with_schema(snapshot.original_input_data)
        deserialized_outputs = _deserialize_value_with_schema(snapshot.pipeline_state.pipeline_outputs)

        # The non-serializable comp1 field is omitted while the serializable siblings are preserved.
        assert "comp1" not in deserialized_inputs
        assert deserialized_inputs["comp2"] == {"input_value": [{"sender": None, "value": "keep me"}]}
        assert deserialized_inputs["comp3"] == {}
        # original_input_data and pipeline_outputs degrade to empty-but-valid payloads.
        assert deserialized_original_input_data == {}
        assert deserialized_outputs == {}
        assert any("Failed to serialize the inputs of the current pipeline state" in msg for msg in caplog.messages)
        assert any("Failed to serialize outputs of the current pipeline state" in msg for msg in caplog.messages)


def test_save_pipeline_snapshot_raises_on_failure(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "true")

    # Point the snapshot directory below an existing file so creating it fails with a filesystem
    # error, exercising the raise_on_failure contract.
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("i am a file")
    snapshot_path = blocking_file / "snapshots"

    snapshot = _create_pipeline_snapshot(
        inputs={},
        component_inputs={},
        break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(snapshot_path)),
        component_visits={"comp1": 1, "comp2": 0},
        original_input_data={},
        ordered_component_names=["comp1", "comp2"],
        include_outputs_from={"comp1"},
        pipeline_outputs={"comp1": {"result": "test"}},
    )

    with pytest.raises(OSError):
        _save_pipeline_snapshot(snapshot)

    with caplog.at_level(logging.ERROR):
        _save_pipeline_snapshot(snapshot, raise_on_failure=False)
        assert any("Failed to save pipeline snapshot to" in msg for msg in caplog.messages)


class TestSnapshotCallback:
    def test_save_pipeline_snapshot_with_callback_no_file_created(self, tmp_path):
        captured_snapshots = []

        def custom_callback(snapshot: PipelineSnapshot) -> str:
            captured_snapshots.append(snapshot)
            return "custom_path_or_id"

        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from=set(),
            pipeline_outputs={},
        )

        result = _save_pipeline_snapshot(snapshot, snapshot_callback=custom_callback)

        # Verify callback was invoked and returned expected value
        assert result == "custom_path_or_id"
        assert len(captured_snapshots) == 1
        assert captured_snapshots[0] == snapshot

        # Verify NO file was created on disk (callback bypasses file saving)
        assert list(tmp_path.glob("*.json")) == []

    def test_save_pipeline_snapshot_callback_returns_none_no_file_created(self, tmp_path):
        captured_snapshots = []

        def custom_callback(snapshot: PipelineSnapshot) -> None:
            captured_snapshots.append(snapshot)

        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from=set(),
            pipeline_outputs={},
        )

        result = _save_pipeline_snapshot(snapshot, snapshot_callback=custom_callback)

        assert result is None
        assert len(captured_snapshots) == 1

        # Verify NO file was created on disk even when snapshot_file_path is set
        assert list(tmp_path.glob("*.json")) == []

    def test_save_pipeline_snapshot_without_callback_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "true")

        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from=set(),
            pipeline_outputs={},
        )

        result = _save_pipeline_snapshot(snapshot)

        # Verify file WAS created on disk
        snapshot_files = list(tmp_path.glob("comp2_*.json"))

        # A file should be created when no callback is provided
        assert len(snapshot_files) == 1
        assert result == str(snapshot_files[0])

        # Verify file contains valid snapshot data
        loaded = load_pipeline_snapshot(snapshot_files[0])
        assert isinstance(loaded.break_point, Breakpoint)
        assert loaded.break_point.component_name == "comp2"

    def test_save_pipeline_snapshot_callback_raises_exception_no_file_created(self, tmp_path, caplog):
        def failing_callback(snapshot: PipelineSnapshot) -> str:
            raise RuntimeError("Database connection failed")

        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from=set(),
            pipeline_outputs={},
        )

        # Test with raise_on_failure=True (default)
        with pytest.raises(RuntimeError, match="Database connection failed"):
            _save_pipeline_snapshot(snapshot, snapshot_callback=failing_callback)

        # Verify NO file was created even after exception
        assert list(tmp_path.glob("*.json")) == []

        # Test with raise_on_failure=False
        with caplog.at_level(logging.ERROR):
            result = _save_pipeline_snapshot(snapshot, raise_on_failure=False, snapshot_callback=failing_callback)
            assert result is None
            assert any("Failed to handle pipeline snapshot with custom callback" in msg for msg in caplog.messages)

        # Still no file should exist
        assert list(tmp_path.glob("*.json")) == []

    def test_pipeline_run_with_snapshot_callback(self, tmp_path):
        captured_snapshots = []

        def custom_callback(snapshot: PipelineSnapshot) -> str:
            captured_snapshots.append(snapshot)
            return "custom_snapshot_id"

        @component
        class SimpleComponent:
            @component.output_types(result=str)
            def run(self, input_value: str) -> dict[str, str]:
                return {"result": f"processed_{input_value}"}

        pipeline = Pipeline()
        comp1 = SimpleComponent()
        comp2 = SimpleComponent()
        pipeline.add_component("comp1", comp1)
        pipeline.add_component("comp2", comp2)
        pipeline.connect("comp1", "comp2")

        # breakpoint on comp2
        break_point = Breakpoint(component_name="comp2", visit_count=0, snapshot_file_path=str(tmp_path))

        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run(
                data={"comp1": {"input_value": "test"}}, break_point=break_point, snapshot_callback=custom_callback
            )

        # Verify callback was called
        assert len(captured_snapshots) == 1
        assert isinstance(captured_snapshots[0].break_point, Breakpoint)
        assert captured_snapshots[0].break_point.component_name == "comp2"
        # Verify the file path in exception is from callback
        assert exc_info.value.pipeline_snapshot_file_path == "custom_snapshot_id"
        # Verify no file was saved to disk
        assert list(tmp_path.glob("*.json")) == []

    def test_pipeline_run_without_snapshot_callback_saves_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "true")

        @component
        class SimpleComponent:
            @component.output_types(result=str)
            def run(self, input_value: str) -> dict[str, str]:
                return {"result": f"processed_{input_value}"}

        pipeline = Pipeline()
        comp1 = SimpleComponent()
        comp2 = SimpleComponent()
        pipeline.add_component("comp1", comp1)
        pipeline.add_component("comp2", comp2)
        pipeline.connect("comp1", "comp2")

        # breakpoint on comp2
        break_point = Breakpoint(component_name="comp2", visit_count=0, snapshot_file_path=str(tmp_path))

        with pytest.raises(BreakpointException):
            pipeline.run(data={"comp1": {"input_value": "test"}}, break_point=break_point)

        # Verify file was saved to disk
        snapshot_files = list(tmp_path.glob("comp2_*.json"))
        assert len(snapshot_files) == 1


class TestSnapshotSaveEnabled:
    def test_is_snapshot_save_enabled_default(self, monkeypatch):
        monkeypatch.delenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, raising=False)
        assert _is_snapshot_save_enabled() is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1"])
    def test_is_snapshot_save_enabled_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, value)
        assert _is_snapshot_save_enabled() is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "False", "0"])
    def test_is_snapshot_save_enabled_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, value)
        assert _is_snapshot_save_enabled() is False

    def test_save_pipeline_snapshot_disabled_via_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "false")

        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from=set(),
            pipeline_outputs={},
        )

        result = _save_pipeline_snapshot(snapshot)

        # Verify no file was created
        assert result is None
        assert list(tmp_path.glob("*.json")) == []

    def test_save_pipeline_snapshot_enabled_via_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "true")

        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from=set(),
            pipeline_outputs={},
        )

        result = _save_pipeline_snapshot(snapshot)

        # Verify file was created
        snapshot_files = list(tmp_path.glob("comp2_*.json"))
        assert len(snapshot_files) == 1
        assert result == str(snapshot_files[0])

    def test_callback_still_invoked_when_env_var_disables_saving(self, tmp_path, monkeypatch):
        """
        This is more a behaviour documentation test: we want to ensure that when the snapshot_callback is provided,
        the file-saving behaviour is always bypassed (the callback is invoked instead).
        """
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "false")

        captured_snapshots = []

        def custom_callback(snapshot: PipelineSnapshot) -> str:
            captured_snapshots.append(snapshot)
            return "custom_result"

        snapshot = _create_pipeline_snapshot(
            inputs={},
            component_inputs={},
            break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
            component_visits={"comp1": 1, "comp2": 0},
            original_input_data={},
            ordered_component_names=["comp1", "comp2"],
            include_outputs_from=set(),
            pipeline_outputs={},
        )

        result = _save_pipeline_snapshot(snapshot, snapshot_callback=custom_callback)

        # Callback should still be invoked
        assert result == "custom_result"
        assert len(captured_snapshots) == 1
        # No file should be created (callback handles it)
        assert list(tmp_path.glob("*.json")) == []

    def test_pipeline_run_with_env_var_disabled(self, tmp_path, monkeypatch):
        """Test that pipeline.run respects the env var when breakpoint is triggered."""
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "false")

        @component
        class SimpleComponent:
            @component.output_types(result=str)
            def run(self, input_value: str) -> dict[str, str]:
                return {"result": f"processed_{input_value}"}

        pipeline = Pipeline()
        pipeline.add_component("comp1", SimpleComponent())
        pipeline.add_component("comp2", SimpleComponent())
        pipeline.connect("comp1", "comp2")

        break_point = Breakpoint(component_name="comp2", visit_count=0, snapshot_file_path=str(tmp_path))

        with pytest.raises(BreakpointException) as exc_info:
            pipeline.run(data={"comp1": {"input_value": "test"}}, break_point=break_point)

        # Verify no file was saved
        assert exc_info.value.pipeline_snapshot_file_path is None
        assert list(tmp_path.glob("*.json")) == []

        # Verify snapshot object is still available for programmatic access
        assert exc_info.value.pipeline_snapshot is not None
        assert isinstance(exc_info.value.pipeline_snapshot.break_point, Breakpoint)
        assert exc_info.value.pipeline_snapshot.break_point.component_name == "comp2"
