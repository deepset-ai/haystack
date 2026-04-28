# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from networkx import MultiDiGraph

from haystack import logging
from haystack.core.errors import PipelineInvalidPipelineSnapshotError
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses.breakpoints import Breakpoint, PipelineSnapshot, PipelineState
from haystack.utils.base_serialization import _serialize_value_with_schema

logger = logging.getLogger(__name__)

# Environment variable to control pipeline snapshot file saving (enabled by default)
HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED = "HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED"

# Type alias for snapshot callback function
# The callback receives a PipelineSnapshot and optionally returns a file path string
SnapshotCallback = Callable[[PipelineSnapshot], str | None]


def _is_snapshot_save_enabled() -> bool:
    """
    Check if pipeline snapshot file saving is enabled via environment variable.

    The environment variable HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED controls whether
    pipeline snapshots are saved to files. By default (when the variable is not set),
    saving is disabled. Only "true" and "1" (case-insensitive) enable saving; any other value disables it.

    :returns: True if snapshot saving is enabled, False otherwise.
    """
    value = os.environ.get(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "false").lower()
    return value in ("true", "1")


def _validate_break_point_against_pipeline(break_point: Breakpoint, graph: MultiDiGraph) -> None:
    """
    Validates the breakpoints passed to the pipeline.

    Makes sure the breakpoint contains a valid components registered in the pipeline.

    :param break_point: a breakpoint to validate
    """
    if break_point.component_name not in graph.nodes:
        raise ValueError(f"break_point {break_point} is not a registered component in the pipeline")


def _validate_pipeline_snapshot_against_pipeline(pipeline_snapshot: PipelineSnapshot, graph: MultiDiGraph) -> None:
    """
    Validates that the pipeline_snapshot contains valid configuration for the current pipeline.

    Raises a PipelineInvalidPipelineSnapshotError if any component in pipeline_snapshot is not part of the
    target pipeline.

    :param pipeline_snapshot: The saved state to validate.
    """

    pipeline_state = pipeline_snapshot.pipeline_state
    valid_components = set(graph.nodes.keys())

    # Check if the ordered_component_names are valid components in the pipeline
    invalid_ordered_components = set(pipeline_snapshot.ordered_component_names) - valid_components
    if invalid_ordered_components:
        raise PipelineInvalidPipelineSnapshotError(
            f"Invalid pipeline snapshot: components {invalid_ordered_components} in 'ordered_component_names' "
            f"are not part of the current pipeline."
        )

    # Check if the original_input_data is valid components in the pipeline
    serialized_input_data = pipeline_snapshot.original_input_data["serialized_data"]
    invalid_input_data = set(serialized_input_data.keys()) - valid_components
    if invalid_input_data:
        raise PipelineInvalidPipelineSnapshotError(
            f"Invalid pipeline snapshot: components {invalid_input_data} in 'input_data' "
            f"are not part of the current pipeline."
        )

    # Validate 'component_visits'
    invalid_component_visits = set(pipeline_state.component_visits.keys()) - valid_components
    if invalid_component_visits:
        raise PipelineInvalidPipelineSnapshotError(
            f"Invalid pipeline snapshot: components {invalid_component_visits} in 'component_visits' "
            f"are not part of the current pipeline."
        )

    component_name = pipeline_snapshot.break_point.component_name
    visit_count = pipeline_snapshot.pipeline_state.component_visits[component_name]

    logger.info(
        "Resuming pipeline from {component} with visit count {visits}", component=component_name, visits=visit_count
    )


def load_pipeline_snapshot(file_path: str | Path) -> PipelineSnapshot:
    """
    Load a saved pipeline snapshot.

    :param file_path: Path to the pipeline_snapshot file.
    :returns:
        Dict containing the loaded pipeline_snapshot.
    """

    file_path = Path(file_path)

    try:
        with open(file_path, encoding="utf-8") as f:
            pipeline_snapshot_dict = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON file {file_path}: {str(e)}", e.doc, e.pos) from e
    except OSError as e:
        raise OSError(f"Error reading {file_path}: {str(e)}") from e

    try:
        pipeline_snapshot = PipelineSnapshot.from_dict(pipeline_snapshot_dict)
    except ValueError as e:
        raise ValueError(f"Invalid pipeline snapshot from {file_path}: {str(e)}") from e

    logger.info("Successfully loaded the pipeline snapshot from: {file_path}", file_path=file_path)
    return pipeline_snapshot


def _save_pipeline_snapshot(
    pipeline_snapshot: PipelineSnapshot,
    raise_on_failure: bool = True,
    snapshot_callback: SnapshotCallback | None = None,
) -> str | None:
    """
    Save the pipeline snapshot dictionary to a JSON file, or invoke a custom callback.

    If a `snapshot_callback` is provided, it will be called with the pipeline snapshot instead of saving to a file.
    This allows users to customize how snapshots are handled (e.g., saving to a database, sending to a remote service).

    When no callback is provided, the default behavior saves to a JSON file:
    - The filename is generated based on the component name, visit count, and timestamp.
        - The component name is taken from the break point's `component_name`.
        - The visit count is taken from the pipeline state's `component_visits` for the component name.
        - The timestamp is taken from the pipeline snapshot's `timestamp` or the current time if not available.
    - The file path is taken from the break point's `snapshot_file_path`.
    - If the `snapshot_file_path` is None, the function will return without saving.

    The default file saving behavior is disabled. To enable it, set the environment variable
    `HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED` to "true" or "1". When disabled,
    the function will return None without saving to a file (custom callbacks are still invoked).

    :param pipeline_snapshot: The pipeline snapshot to save.
    :param raise_on_failure: If True, raises an exception if saving fails. If False, logs the error and returns.
    :param snapshot_callback: Optional callback function that receives the PipelineSnapshot.
        If provided, the callback is invoked instead of the default file-saving behavior.
        The callback should return an optional string (e.g., a file path or identifier) or None.

    :returns:
        The full path to the saved JSON file (or the value returned by the callback), or None if
        `snapshot_file_path` is None, no callback is provided, or snapshot saving is disabled.
    :raises:
        Exception: If saving the JSON snapshot fails (when raise_on_failure is True).
    """
    # If a callback is provided, use it instead of the default file-saving behavior
    if snapshot_callback is not None:
        try:
            result = snapshot_callback(pipeline_snapshot)
            logger.info("Pipeline snapshot handled by custom callback.")
            return result
        except Exception as error:
            logger.exception("Failed to handle pipeline snapshot with custom callback. Error: {error}", error=error)
            if raise_on_failure:
                raise
            return None

    # Check if snapshot saving is enabled via environment variable (enabled by default)
    if not _is_snapshot_save_enabled():
        logger.debug("Pipeline snapshot file saving is disabled via HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED env var.")
        return None

    break_point = pipeline_snapshot.break_point
    snapshot_file_path = break_point.snapshot_file_path

    if snapshot_file_path is None:
        return None

    dt = pipeline_snapshot.timestamp or datetime.now()
    snapshot_dir = Path(snapshot_file_path)

    component_name = break_point.component_name
    visit_nr = pipeline_snapshot.pipeline_state.component_visits.get(component_name, 0)
    timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{component_name}_{visit_nr}_{timestamp}.json"
    full_path = snapshot_dir / file_name

    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f_out:
            json.dump(pipeline_snapshot.to_dict(), f_out, indent=2)
        logger.info(
            "Pipeline snapshot saved to '{full_path}'. You can use this file to debug or resume the pipeline.",
            full_path=full_path,
        )
    except Exception as error:
        logger.exception("Failed to save pipeline snapshot to '{full_path}'. Error: {e}", full_path=full_path, e=error)
        if raise_on_failure:
            raise

    return str(full_path)


def _create_pipeline_snapshot(
    *,
    inputs: dict[str, Any],
    component_inputs: dict[str, Any],
    break_point: Breakpoint,
    component_visits: dict[str, int],
    original_input_data: dict[str, Any],
    ordered_component_names: list[str],
    include_outputs_from: set[str],
    pipeline_outputs: dict[str, Any],
) -> PipelineSnapshot:
    """
    Create a snapshot of the pipeline at the point where the breakpoint was triggered.

    :param inputs: The current pipeline snapshot inputs.
    :param component_inputs: The inputs to the component that triggered the breakpoint.
    :param break_point: The breakpoint that triggered the snapshot.
    :param component_visits: The visit count of the component that triggered the breakpoint.
    :param original_input_data: The original input data.
    :param ordered_component_names: The ordered component names.
    :param include_outputs_from: Set of component names whose outputs should be included in the pipeline results.
    :param pipeline_outputs: The current outputs of the pipeline.
    :returns:
        A PipelineSnapshot containing the state of the pipeline at the point of the breakpoint.
    """
    component_name = break_point.component_name

    transformed_original_input_data = _transform_json_structure(original_input_data)
    transformed_inputs = _transform_json_structure({**inputs, component_name: component_inputs})

    serialized_inputs = _serialize_with_field_fallback(
        transformed_inputs, description="the inputs of the current pipeline state"
    )
    serialized_original_input_data = _serialize_with_field_fallback(
        transformed_original_input_data, description="original input data for `pipeline.run`"
    )
    serialized_pipeline_outputs = _serialize_with_field_fallback(
        pipeline_outputs, description="outputs of the current pipeline state"
    )

    return PipelineSnapshot(
        pipeline_state=PipelineState(
            inputs=serialized_inputs, component_visits=component_visits, pipeline_outputs=serialized_pipeline_outputs
        ),
        timestamp=datetime.now(),
        break_point=break_point,
        original_input_data=serialized_original_input_data,
        ordered_component_names=ordered_component_names,
        include_outputs_from=include_outputs_from,
    )


def _transform_json_structure(data: dict[str, Any] | list[Any] | Any) -> Any:
    """
    Transforms a JSON structure by removing the 'sender' key and moving the 'value' to the top level.

    For example:
    "key": [{"sender": null, "value": "some value"}] -> "key": "some value"

    :param data: The JSON structure to transform.
    :returns: The transformed structure.
    """
    if isinstance(data, dict):
        # If this dict has both 'sender' and 'value', return just the value
        if "value" in data and "sender" in data:
            return data["value"]
        # Otherwise, recursively process each key-value pair
        return {k: _transform_json_structure(v) for k, v in data.items()}

    if isinstance(data, list):
        # First, transform each item in the list.
        transformed = [_transform_json_structure(item) for item in data]
        # If the original list has exactly one element and that element was a dict
        # with 'sender' and 'value', then unwrap the list.
        if len(data) == 1 and isinstance(data[0], dict) and "value" in data[0] and "sender" in data[0]:
            return transformed[0]
        return transformed

    # For other data types, just return the value as is.
    return data


def _serialize_with_field_fallback(payload: Any, *, description: str) -> dict[str, Any]:
    """
    Serialize a payload and, on failure, retry field-by-field to preserve resumable fields.

    If the whole payload serializes, the result is returned as-is. Otherwise, and if the payload is a
    mapping, each top-level field is serialized individually and only the failing fields are omitted.
    When the payload is not a mapping, or when every field fails to serialize, the helper returns a
    structurally valid empty-object payload so that the downstream ``_deserialize_value_with_schema``
    can still load it back instead of raising ``DeserializationError`` on a bare ``{}``.

    :param payload: The value to serialize.
    :param description: Short human-readable label used in warning messages, for example
        ``"the agent's chat_generator inputs"`` or ``"the inputs of the current pipeline state"``.
    :returns: A dict of the form ``{"serialization_schema": ..., "serialized_data": ...}``.
    """
    try:
        return _serialize_value_with_schema(_deepcopy_with_exceptions(payload))
    except Exception as error:
        logger.warning(
            "Failed to serialize {description}. "
            "Haystack will omit only the non-serializable fields when possible. Error: {e}",
            description=description,
            e=error,
        )

    serialized_properties: dict[str, Any] = {}
    serialized_data: dict[str, Any] = {}

    if isinstance(payload, dict):
        for field_name, value in payload.items():
            try:
                serialized_value = _serialize_value_with_schema(_deepcopy_with_exceptions(value))
            except Exception as field_error:
                logger.warning(
                    "Failed to serialize the '{field_name}' field of {description}. "
                    "The field will be omitted from the snapshot. Error: {e}",
                    field_name=field_name,
                    description=description,
                    e=field_error,
                )
                continue

            serialized_properties[field_name] = serialized_value["serialization_schema"]
            serialized_data[field_name] = serialized_value["serialized_data"]

    return {
        "serialization_schema": {"type": "object", "properties": serialized_properties},
        "serialized_data": serialized_data,
    }
