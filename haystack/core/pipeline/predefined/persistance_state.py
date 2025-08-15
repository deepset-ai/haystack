from datetime import datetime
from typing import Any, Optional

from haystack.core.pipeline.breakpoint import _save_pipeline_snapshot_to_file, _transform_json_structure
from haystack.dataclasses.breakpoints import Breakpoint, PipelineSnapshot, PipelineState
from haystack.utils import _serialize_value_with_schema


def _create_automatic_pipeline_snapshot(
    *,
    inputs: dict[str, Any],
    component_name: str,
    component_visits: dict[str, int],
    original_input_data: Optional[dict[str, Any]] = None,
    ordered_component_names: Optional[list[str]] = None,
    include_outputs_from: Optional[set[str]] = None,
    pipeline_outputs: Optional[dict[str, Any]] = None,
    snapshot_file_path: Optional[str] = None,
) -> PipelineSnapshot:
    """
    Create an automatic snapshot of the pipeline after a component execution.

    :param inputs: The current pipeline snapshot inputs.
    :param component_name: The name of the component that just completed execution.
    :param component_visits: The visit count of all components.
    :param original_input_data: The original input data.
    :param ordered_component_names: The ordered component names.
    :param include_outputs_from: Set of component names whose outputs should be included.
    :param pipeline_outputs: Dictionary containing outputs from components.
    :param snapshot_file_path: Optional path to save the snapshot.
    """
    dt = datetime.now()

    # Create a dummy breakpoint for the automatic snapshot
    auto_breakpoint = Breakpoint(
        component_name=component_name,
        visit_count=component_visits[component_name],
        snapshot_file_path=snapshot_file_path,
    )

    transformed_original_input_data = _transform_json_structure(original_input_data)
    transformed_inputs = _transform_json_structure(inputs)

    pipeline_snapshot = PipelineSnapshot(
        pipeline_state=PipelineState(
            inputs=_serialize_value_with_schema(transformed_inputs),
            component_visits=component_visits,
            pipeline_outputs=pipeline_outputs or {},
        ),
        timestamp=dt,
        break_point=auto_breakpoint,
        original_input_data=_serialize_value_with_schema(transformed_original_input_data),
        ordered_component_names=ordered_component_names or [],
        include_outputs_from=include_outputs_from or set(),
    )

    # Save the snapshot if a path is provided
    if snapshot_file_path is not None:
        _save_pipeline_snapshot_to_file(
            pipeline_snapshot=pipeline_snapshot, snapshot_file_path=snapshot_file_path, dt=dt
        )

    return pipeline_snapshot
