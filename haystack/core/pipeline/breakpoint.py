# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from networkx import MultiDiGraph

from haystack import logging
from haystack.core.errors import BreakpointException, PipelineInvalidPipelineSnapshotError
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import (
    AgentBreakpoint,
    AgentSnapshot,
    Breakpoint,
    PipelineSnapshot,
    PipelineState,
    ToolBreakpoint,
)
from haystack.utils.base_serialization import _serialize_value_with_schema
from haystack.utils.misc import _get_output_dir

if TYPE_CHECKING:
    from haystack.components.agents.agent import _ExecutionContext
    from haystack.tools import Tool, Toolset, ToolsType

logger = logging.getLogger(__name__)


def _validate_break_point_against_pipeline(
    break_point: Union[Breakpoint, AgentBreakpoint], graph: MultiDiGraph
) -> None:
    """
    Validates the breakpoints passed to the pipeline.

    Makes sure the breakpoint contains a valid components registered in the pipeline.

    :param break_point: a breakpoint to validate, can be Breakpoint or AgentBreakpoint
    """

    # all Breakpoints must refer to a valid component in the pipeline
    if isinstance(break_point, Breakpoint) and break_point.component_name not in graph.nodes:
        raise ValueError(f"break_point {break_point} is not a registered component in the pipeline")

    if isinstance(break_point, AgentBreakpoint):
        breakpoint_agent_component = graph.nodes.get(break_point.agent_name)
        if not breakpoint_agent_component:
            raise ValueError(f"break_point {break_point} is not a registered Agent component in the pipeline")

        if isinstance(break_point.break_point, ToolBreakpoint):
            instance = breakpoint_agent_component["instance"]
            for tool in instance.tools:
                if break_point.break_point.tool_name == tool.name:
                    break
            else:
                raise ValueError(
                    f"break_point {break_point.break_point} is not a registered tool in the Agent component"
                )


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

    if isinstance(pipeline_snapshot.break_point, AgentBreakpoint):
        component_name = pipeline_snapshot.break_point.agent_name
    else:
        component_name = pipeline_snapshot.break_point.component_name

    visit_count = pipeline_snapshot.pipeline_state.component_visits[component_name]

    logger.info(
        "Resuming pipeline from {component} with visit count {visits}", component=component_name, visits=visit_count
    )


def load_pipeline_snapshot(file_path: Union[str, Path]) -> PipelineSnapshot:
    """
    Load a saved pipeline snapshot.

    :param file_path: Path to the pipeline_snapshot file.
    :returns:
        Dict containing the loaded pipeline_snapshot.
    """

    file_path = Path(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            pipeline_snapshot_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON file {file_path}: {str(e)}", e.doc, e.pos)
    except IOError as e:
        raise IOError(f"Error reading {file_path}: {str(e)}")

    try:
        pipeline_snapshot = PipelineSnapshot.from_dict(pipeline_snapshot_dict)
    except ValueError as e:
        raise ValueError(f"Invalid pipeline snapshot from {file_path}: {str(e)}")

    logger.info(f"Successfully loaded the pipeline snapshot from: {file_path}")
    return pipeline_snapshot


def _save_pipeline_snapshot(pipeline_snapshot: PipelineSnapshot, raise_on_failure: bool = True) -> Optional[str]:
    """
    Save the pipeline snapshot dictionary to a JSON file.

    - The filename is generated based on the component name, visit count, and timestamp.
        - The component name is taken from the break point's `component_name`.
        - The visit count is taken from the pipeline state's `component_visits` for the component name.
        - The timestamp is taken from the pipeline snapshot's `timestamp` or the current time if not available.
    - The file path is taken from the break point's `snapshot_file_path`.
    - If the `snapshot_file_path` is None, the function will return without saving.

    :param pipeline_snapshot: The pipeline snapshot to save.
    :param raise_on_failure: If True, raises an exception if saving fails. If False, logs the error and returns.

    :returns:
        The full path to the saved JSON file, or None if `snapshot_file_path` is None.
    :raises:
        Exception: If saving the JSON snapshot fails.
    """
    break_point = pipeline_snapshot.break_point
    snapshot_file_path = (
        break_point.break_point.snapshot_file_path
        if isinstance(break_point, AgentBreakpoint)
        else break_point.snapshot_file_path
    )

    if snapshot_file_path is None:
        return None

    dt = pipeline_snapshot.timestamp or datetime.now()
    snapshot_dir = Path(snapshot_file_path)

    # Generate filename
    # We check if the agent_name is provided to differentiate between agent and non-agent breakpoints
    if isinstance(break_point, AgentBreakpoint):
        agent_name = break_point.agent_name
        component_name = break_point.break_point.component_name
    else:
        component_name = break_point.component_name
        agent_name = None

    visit_nr = pipeline_snapshot.pipeline_state.component_visits.get(component_name, 0)
    timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{agent_name + '_' if agent_name else ''}{component_name}_{visit_nr}_{timestamp}.json"
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
        logger.error("Failed to save pipeline snapshot to '{full_path}'. Error: {e}", full_path=full_path, e=error)
        if raise_on_failure:
            raise

    return str(full_path)


def _create_pipeline_snapshot(
    *,
    inputs: dict[str, Any],
    component_inputs: dict[str, Any],
    break_point: Union[AgentBreakpoint, Breakpoint],
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
    :param break_point: The breakpoint that triggered the snapshot, can be AgentBreakpoint or Breakpoint.
    :param component_visits: The visit count of the component that triggered the breakpoint.
    :param original_input_data: The original input data.
    :param ordered_component_names: The ordered component names.
    :param include_outputs_from: Set of component names whose outputs should be included in the pipeline results.
    :param pipeline_outputs: The current outputs of the pipeline.
    :returns:
        A PipelineSnapshot containing the state of the pipeline at the point of the breakpoint.
    """
    if isinstance(break_point, AgentBreakpoint):
        component_name = break_point.agent_name
    else:
        component_name = break_point.component_name

    transformed_original_input_data = _transform_json_structure(original_input_data)
    transformed_inputs = _transform_json_structure({**inputs, component_name: component_inputs})

    try:
        serialized_inputs = _serialize_value_with_schema(transformed_inputs)
    except Exception as error:
        logger.warning(
            "Failed to serialize the inputs of the current pipeline state. "
            "The inputs in the snapshot will be replaced with an empty dictionary. Error: {e}",
            e=error,
        )
        serialized_inputs = {}

    try:
        serialized_original_input_data = _serialize_value_with_schema(transformed_original_input_data)
    except Exception as error:
        logger.warning(
            "Failed to serialize original input data for `pipeline.run`. "
            "This likely occurred due to non-serializable object types. "
            "The snapshot will store an empty dictionary instead. Error: {e}",
            e=error,
        )
        serialized_original_input_data = {}

    pipeline_snapshot = PipelineSnapshot(
        pipeline_state=PipelineState(
            inputs=serialized_inputs, component_visits=component_visits, pipeline_outputs=pipeline_outputs
        ),
        timestamp=datetime.now(),
        break_point=break_point,
        original_input_data=serialized_original_input_data,
        ordered_component_names=ordered_component_names,
        include_outputs_from=include_outputs_from,
    )
    return pipeline_snapshot


def _transform_json_structure(data: Union[dict[str, Any], list[Any], Any]) -> Any:
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


def _trigger_break_point(*, pipeline_snapshot: PipelineSnapshot) -> None:
    """
    Trigger a breakpoint by saving a snapshot and raising exception.

    :param pipeline_snapshot: The current pipeline snapshot containing the state and break point
    :raises BreakpointException: When breakpoint is triggered
    """
    full_file_path = _save_pipeline_snapshot(pipeline_snapshot=pipeline_snapshot)

    if isinstance(pipeline_snapshot.break_point, Breakpoint):
        component_name = pipeline_snapshot.break_point.component_name
    else:
        component_name = pipeline_snapshot.break_point.agent_name

    component_visits = pipeline_snapshot.pipeline_state.component_visits
    msg = f"Breaking at component {component_name} at visit count {component_visits[component_name]}"
    raise BreakpointException(
        message=msg,
        component=component_name,
        pipeline_snapshot=pipeline_snapshot,
        pipeline_snapshot_file_path=full_file_path,
    )


def _create_agent_snapshot(
    *, component_visits: dict[str, int], agent_breakpoint: AgentBreakpoint, component_inputs: dict[str, Any]
) -> AgentSnapshot:
    """
    Create a snapshot of the agent's state.

    :param component_visits: The visit counts for the agent's components.
    :param agent_breakpoint: AgentBreakpoint object containing breakpoints
    :return: An AgentSnapshot containing the agent's state and component visits.
    """
    return AgentSnapshot(
        component_inputs={
            "chat_generator": _serialize_value_with_schema(
                _deepcopy_with_exceptions(component_inputs["chat_generator"])
            ),
            "tool_invoker": _serialize_value_with_schema(_deepcopy_with_exceptions(component_inputs["tool_invoker"])),
        },
        component_visits=component_visits,
        break_point=agent_breakpoint,
        timestamp=datetime.now(),
    )


def _validate_tool_breakpoint_is_valid(agent_breakpoint: AgentBreakpoint, tools: "ToolsType") -> None:
    """
    Validates the AgentBreakpoint passed to the agent.

    Validates that the tool name in ToolBreakpoints correspond to a tool available in the agent.

    :param agent_breakpoint: AgentBreakpoint object containing breakpoints for the agent components.
    :param tools: A list of Tool and/or Toolset objects, or a Toolset that the agent can use.
    :raises ValueError: If any tool name in ToolBreakpoints is not available in the agent's tools.
    """
    from haystack.tools.utils import flatten_tools_or_toolsets  # avoid circular import

    available_tool_names = {tool.name for tool in flatten_tools_or_toolsets(tools)}
    tool_breakpoint = agent_breakpoint.break_point
    # Assert added for mypy to pass, but this is already checked before this function is called
    assert isinstance(tool_breakpoint, ToolBreakpoint)
    if tool_breakpoint.tool_name and tool_breakpoint.tool_name not in available_tool_names:
        raise ValueError(f"Tool '{tool_breakpoint.tool_name}' is not available in the agent's tools")


def _create_pipeline_snapshot_from_chat_generator(
    *,
    execution_context: "_ExecutionContext",
    agent_name: Optional[str] = None,
    break_point: Optional[AgentBreakpoint] = None,
    parent_snapshot: Optional[PipelineSnapshot] = None,
) -> PipelineSnapshot:
    """
    Create a pipeline snapshot when a chat generator breakpoint is raised or an exception during execution occurs.

    :param execution_context: The current execution context of the agent.
    :param agent_name: The name of the agent component if present in a pipeline.
    :param break_point: An optional AgentBreakpoint object. If provided, it will be used instead of creating a new one.
        A scenario where a new breakpoint is created is when an exception occurs during chat generation and we want to
        capture the state at that point.
    :param parent_snapshot: An optional parent PipelineSnapshot to build upon.
    :returns:
        A PipelineSnapshot containing the state of the pipeline and agent at the point of the breakpoint or exception.
    """
    if break_point is None:
        agent_breakpoint = AgentBreakpoint(
            agent_name=agent_name or "agent",
            break_point=Breakpoint(
                component_name="chat_generator",
                visit_count=execution_context.component_visits["chat_generator"],
                snapshot_file_path=_get_output_dir("pipeline_snapshot"),
            ),
        )
    else:
        agent_breakpoint = break_point

    agent_snapshot = _create_agent_snapshot(
        component_visits=execution_context.component_visits,
        agent_breakpoint=agent_breakpoint,
        component_inputs={
            "chat_generator": {
                "messages": execution_context.state.data["messages"],
                **execution_context.chat_generator_inputs,
            },
            "tool_invoker": {"messages": [], "state": execution_context.state, **execution_context.tool_invoker_inputs},
        },
    )
    if parent_snapshot is None:
        # Create an empty pipeline snapshot if no parent snapshot is provided
        final_snapshot = PipelineSnapshot(
            pipeline_state=PipelineState(inputs={}, component_visits={}, pipeline_outputs={}),
            timestamp=agent_snapshot.timestamp,
            break_point=agent_snapshot.break_point,
            agent_snapshot=agent_snapshot,
            original_input_data={},
            ordered_component_names=[],
            include_outputs_from=set(),
        )
    else:
        final_snapshot = replace(parent_snapshot, agent_snapshot=agent_snapshot)

    return final_snapshot


def _create_pipeline_snapshot_from_tool_invoker(
    *,
    execution_context: "_ExecutionContext",
    tool_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    break_point: Optional[AgentBreakpoint] = None,
    parent_snapshot: Optional[PipelineSnapshot] = None,
) -> PipelineSnapshot:
    """
    Create a pipeline snapshot when a tool invoker breakpoint is raised or an exception during execution occurs.

    :param execution_context: The current execution context of the agent.
    :param tool_name: The name of the tool that triggered the breakpoint, if available.
    :param agent_name: The name of the agent component if present in a pipeline.
    :param break_point: An optional AgentBreakpoint object. If provided, it will be used instead of creating a new one.
        A scenario where a new breakpoint is created is when an exception occurs during tool execution and we want to
        capture the state at that point.
    :param parent_snapshot: An optional parent PipelineSnapshot to build upon.
    :returns:
        A PipelineSnapshot containing the state of the pipeline and agent at the point of the breakpoint or exception.
    """
    if break_point is None:
        agent_breakpoint = AgentBreakpoint(
            agent_name=agent_name or "agent",
            break_point=ToolBreakpoint(
                component_name="tool_invoker",
                visit_count=execution_context.component_visits["tool_invoker"],
                tool_name=tool_name,
                snapshot_file_path=_get_output_dir("pipeline_snapshot"),
            ),
        )
    else:
        agent_breakpoint = break_point

    messages = execution_context.state.data["messages"]
    agent_snapshot = _create_agent_snapshot(
        component_visits=execution_context.component_visits,
        agent_breakpoint=agent_breakpoint,
        component_inputs={
            "chat_generator": {"messages": messages[:-1], **execution_context.chat_generator_inputs},
            "tool_invoker": {
                "messages": messages[-1:],  # tool invoker consumes last msg from the chat_generator, contains tool call
                "state": execution_context.state,
                **execution_context.tool_invoker_inputs,
            },
        },
    )
    if parent_snapshot is None:
        # Create an empty pipeline snapshot if no parent snapshot is provided
        final_snapshot = PipelineSnapshot(
            pipeline_state=PipelineState(inputs={}, component_visits={}, pipeline_outputs={}),
            timestamp=agent_snapshot.timestamp,
            break_point=agent_snapshot.break_point,
            agent_snapshot=agent_snapshot,
            original_input_data={},
            ordered_component_names=[],
            include_outputs_from=set(),
        )
    else:
        final_snapshot = replace(parent_snapshot, agent_snapshot=agent_snapshot)

    return final_snapshot


def _trigger_chat_generator_breakpoint(*, pipeline_snapshot: PipelineSnapshot) -> None:
    """
    Trigger a breakpoint before ChatGenerator execution in Agent.

    :param pipeline_snapshot: PipelineSnapshot object containing the state of the pipeline and Agent snapshot.
    :raises BreakpointException: Always raised when this function is called, indicating a breakpoint has been triggered.
    """
    if not isinstance(pipeline_snapshot.break_point, AgentBreakpoint):
        raise ValueError("PipelineSnapshot must contain an AgentBreakpoint to trigger a chat generator breakpoint.")

    if not isinstance(pipeline_snapshot.agent_snapshot, AgentSnapshot):
        raise ValueError("PipelineSnapshot must contain an AgentSnapshot to trigger a chat generator breakpoint.")

    break_point = pipeline_snapshot.break_point.break_point
    full_file_path = _save_pipeline_snapshot(pipeline_snapshot=pipeline_snapshot)
    msg = (
        f"Breaking at {break_point.component_name} visit count "
        f"{pipeline_snapshot.agent_snapshot.component_visits[break_point.component_name]}"
    )
    raise BreakpointException(
        message=msg,
        component=break_point.component_name,
        pipeline_snapshot=pipeline_snapshot,
        pipeline_snapshot_file_path=full_file_path,
    )


def _trigger_tool_invoker_breakpoint(*, llm_messages: list[ChatMessage], pipeline_snapshot: PipelineSnapshot) -> None:
    """
    Check if a tool call breakpoint should be triggered before executing the tool invoker.

    :param llm_messages: List of ChatMessage objects containing potential tool calls.
    :param pipeline_snapshot: PipelineSnapshot object containing the state of the pipeline and Agent snapshot.
    :raises BreakpointException: If the breakpoint is triggered, indicating a breakpoint has been reached for a tool
        call.
    """
    if not pipeline_snapshot.agent_snapshot:
        raise ValueError("PipelineSnapshot must contain an AgentSnapshot to trigger a tool call breakpoint.")

    if not isinstance(pipeline_snapshot.agent_snapshot.break_point.break_point, ToolBreakpoint):
        return

    tool_breakpoint = pipeline_snapshot.agent_snapshot.break_point.break_point

    # Check if we should break for this specific tool or all tools
    if tool_breakpoint.tool_name is None:
        # Break for any tool call
        should_break = any(msg.tool_call for msg in llm_messages)
    else:
        # Break only for the specific tool
        should_break = any(
            tc.tool_name == tool_breakpoint.tool_name for msg in llm_messages for tc in msg.tool_calls or []
        )

    if not should_break:
        return  # No breakpoint triggered

    full_file_path = _save_pipeline_snapshot(pipeline_snapshot=pipeline_snapshot)

    msg = (
        f"Breaking at {tool_breakpoint.component_name} visit count "
        f"{pipeline_snapshot.agent_snapshot.component_visits[tool_breakpoint.component_name]}"
    )
    if tool_breakpoint.tool_name:
        msg += f" for tool {tool_breakpoint.tool_name}"
    logger.info(msg)

    raise BreakpointException(
        message=msg,
        component=tool_breakpoint.component_name,
        pipeline_snapshot=pipeline_snapshot,
        pipeline_snapshot_file_path=full_file_path,
    )
