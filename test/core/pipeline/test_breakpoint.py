# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import pytest

from haystack import component
from haystack.core.errors import BreakpointException
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.breakpoint import (
    HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED,
    _create_pipeline_snapshot,
    _is_snapshot_save_enabled,
    _save_pipeline_snapshot,
    _transform_json_structure,
    load_pipeline_snapshot,
)
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint, PipelineSnapshot, PipelineState


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


def test_breakpoint_saves_intermediate_outputs(tmp_path):
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

    try:
        # run with include_outputs_from to capture intermediate outputs
        pipeline.run(data={"comp1": {"input_value": "test"}}, include_outputs_from={"comp1"}, break_point=break_point)
    except BreakpointException as e:
        # breakpoint should be triggered
        assert e.component == "comp2"

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

        # verify the whole pipeline state contains the expected data
        assert loaded_snapshot.pipeline_state.component_visits["comp1"] == 1
        assert loaded_snapshot.pipeline_state.component_visits["comp2"] == 0
        assert "comp1" in loaded_snapshot.include_outputs_from
        assert loaded_snapshot.break_point.component_name == "comp2"
        assert loaded_snapshot.break_point.visit_count == 0


class TestCreatePipelineSnapshot:
    def test_create_pipeline_snapshot_all_fields(self):
        break_point = Breakpoint(component_name="comp2")
        ordered_component_names = ["comp1", "comp2"]
        include_outputs_from = {"comp1"}

        snapshot = _create_pipeline_snapshot(
            inputs={"comp1": {"input_value": [{"sender": None, "value": "test"}]}, "comp2": {}},
            component_inputs={"input_value": "processed_test"},
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
        assert snapshot.agent_snapshot is None
        assert snapshot.include_outputs_from == include_outputs_from
        assert snapshot.pipeline_state == PipelineState(
            inputs={
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "comp1": {"type": "object", "properties": {"input_value": {"type": "string"}}},
                        "comp2": {"type": "object", "properties": {"input_value": {"type": "string"}}},
                    },
                },
                "serialized_data": {"comp1": {"input_value": "test"}, "comp2": {"input_value": "processed_test"}},
            },
            component_visits={"comp1": 1, "comp2": 0},
            pipeline_outputs={
                "serialization_schema": {
                    "type": "object",
                    "properties": {"comp1": {"type": "object", "properties": {"result": {"type": "string"}}}},
                },
                "serialized_data": {"comp1": {"result": "processed_test"}},
            },
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


def test_save_pipeline_snapshot_raises_on_failure(tmp_path, caplog):
    snapshot = _create_pipeline_snapshot(
        inputs={},
        component_inputs={},
        break_point=Breakpoint(component_name="comp2", snapshot_file_path=str(tmp_path)),
        component_visits={"comp1": 1, "comp2": 0},
        original_input_data={},
        ordered_component_names=["comp1", "comp2"],
        include_outputs_from={"comp1"},
        # We use a non-serializable type (bytes) directly in pipeline outputs to trigger the error
        pipeline_outputs={"comp1": {"result": b"test"}},
    )

    with pytest.raises(TypeError):
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

    def test_save_pipeline_snapshot_without_callback_creates_file(self, tmp_path):
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
        assert captured_snapshots[0].break_point.component_name == "comp2"
        # Verify the file path in exception is from callback
        assert exc_info.value.pipeline_snapshot_file_path == "custom_snapshot_id"
        # Verify no file was saved to disk
        assert list(tmp_path.glob("*.json")) == []

    def test_pipeline_run_without_snapshot_callback_saves_file(self, tmp_path):
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
        assert _is_snapshot_save_enabled() is True

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1"])
    def test_is_snapshot_save_enabled_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, value)
        assert _is_snapshot_save_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", ""])
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
