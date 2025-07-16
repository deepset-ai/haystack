# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-return-statements

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from networkx import MultiDiGraph

from haystack import logging
from haystack.core.errors import BreakpointException, PipelineInvalidResumeStateError
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
from haystack.utils.base_serialization import _serialize_value_with_schema

logger = logging.getLogger(__name__)


def _validate_break_point(break_point: Union[Breakpoint, AgentBreakpoint], graph: MultiDiGraph) -> None:
    """
    Validates the breakpoints passed to the pipeline.

    Makes sure the breakpoint contains a valid components registered in the pipeline.

    :param break_point: a breakpoint to validate, can be Breakpoint or AgentBreakpoint
    """

    # all Breakpoints must refer to a valid component in the pipeline
    if isinstance(break_point, Breakpoint) and break_point.component_name not in graph.nodes:
        raise ValueError(f"pipeline_breakpoint {break_point} is not a registered component in the pipeline")

    if isinstance(break_point, AgentBreakpoint):
        breakpoint_agent_component = graph.nodes.get(break_point.agent_name)
        if not breakpoint_agent_component:
            raise ValueError(f"pipeline_breakpoint {break_point} is not a registered Agent component in the pipeline")

        if isinstance(break_point.break_point, ToolBreakpoint):
            instance = breakpoint_agent_component["instance"]
            for tool in instance.tools:
                if break_point.break_point.tool_name == tool.name:
                    break
            else:
                raise ValueError(
                    f"pipeline_breakpoint {break_point.break_point} is not a registered tool in the Agent component"
                )


def _validate_components_against_pipeline(resume_state: Dict[str, Any], graph: MultiDiGraph) -> None:
    """
    Validates that the resume_state contains valid configuration for the current pipeline.

    Raises a PipelineInvalidResumeStateError if any component in resume_state is not part of the target pipeline.

    :param resume_state: The saved state to validate.
    """

    pipeline_state = resume_state["pipeline_state"]
    valid_components = set(graph.nodes.keys())

    # Check if the ordered_component_names are valid components in the pipeline
    invalid_ordered_components = set(pipeline_state["ordered_component_names"]) - valid_components
    if invalid_ordered_components:
        raise PipelineInvalidResumeStateError(
            f"Invalid resume state: components {invalid_ordered_components} in 'ordered_component_names' "
            f"are not part of the current pipeline."
        )

    # Check if the input_data is valid components in the pipeline
    serialized_input_data = resume_state["input_data"]["serialized_data"]
    invalid_input_data = set(serialized_input_data.keys()) - valid_components
    if invalid_input_data:
        raise PipelineInvalidResumeStateError(
            f"Invalid resume state: components {invalid_input_data} in 'input_data' "
            f"are not part of the current pipeline."
        )

    # Validate 'component_visits'
    invalid_component_visits = set(pipeline_state["component_visits"].keys()) - valid_components
    if invalid_component_visits:
        raise PipelineInvalidResumeStateError(
            f"Invalid resume state: components {invalid_component_visits} in 'component_visits' "
            f"are not part of the current pipeline."
        )

    logger.info(
        f"Resuming pipeline from component: {resume_state['pipeline_breakpoint']['component']} "
        f"(visit {resume_state['pipeline_breakpoint']['visits']})"
    )


def _validate_resume_state(resume_state: Dict[str, Any]) -> None:
    """
    Validates the loaded pipeline resume_state.

    Ensures that the resume_state contains required keys: "input_data", "pipeline_breakpoint", and "pipeline_state".

    Raises:
        ValueError: If required keys are missing or the component sets are inconsistent.
    """

    # top-level state has all required keys
    required_top_keys = {"input_data", "pipeline_breakpoint", "pipeline_state"}
    missing_top = required_top_keys - resume_state.keys()
    if missing_top:
        raise ValueError(f"Invalid state file: missing required keys {missing_top}")

    # pipeline_state has the necessary keys
    pipeline_state = resume_state["pipeline_state"]

    required_pipeline_keys = {"inputs", "component_visits", "ordered_component_names"}
    missing_pipeline = required_pipeline_keys - pipeline_state.keys()
    if missing_pipeline:
        raise ValueError(f"Invalid pipeline_state: missing required keys {missing_pipeline}")

    # component_visits and ordered_component_names must be consistent
    components_in_state = set(pipeline_state["component_visits"].keys())
    components_in_order = set(pipeline_state["ordered_component_names"])

    if components_in_state != components_in_order:
        raise ValueError(
            f"Inconsistent state: components in pipeline_state['component_visits'] {components_in_state} "
            f"do not match components in ordered_component_names {components_in_order}"
        )

    logger.info("Passed resume state validated successfully.")


def load_state(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a saved pipeline state.

    :param file_path: Path to the resume_state file.
    :returns:
        Dict containing the loaded resume_state.
    """

    file_path = Path(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON file {file_path}: {str(e)}", e.doc, e.pos)
    except IOError as e:
        raise IOError(f"Error reading {file_path}: {str(e)}")

    try:
        _validate_resume_state(resume_state=state)
    except ValueError as e:
        raise ValueError(f"Invalid pipeline state from {file_path}: {str(e)}")

    logger.info(f"Successfully loaded pipeline state from: {file_path}")
    return state


def _process_main_pipeline_state(main_pipeline_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Process and serialize main pipeline state for agent breakpoints.

    :param main_pipeline_state: Dictionary containing main pipeline state with keys: "component_visits",
                                "ordered_component_names", "original_input_data", and "inputs".
    :returns: Processed main pipeline state or None if not available or invalid.
    """
    if not main_pipeline_state:
        return None

    original_input_data = main_pipeline_state.get("original_input_data")
    inputs = main_pipeline_state.get("inputs")

    if not (original_input_data and inputs):
        return None

    return {
        "component_visits": main_pipeline_state.get("component_visits"),
        "ordered_component_names": main_pipeline_state.get("ordered_component_names"),
        "original_input_data": _serialize_value_with_schema(_transform_json_structure(original_input_data)),
        "inputs": _serialize_value_with_schema(_transform_json_structure(inputs)),
    }


def _save_state_to_file(
    *,
    state: Dict[str, Any],
    debug_path: Union[str, Path],
    dt: datetime,
    is_agent: bool,
    agent_name: Optional[str],
    component_name: str,
) -> None:
    """
    Save state dictionary to a JSON file.

    :param state: The state dictionary to save.
    :param debug_path: The path where to save the file.
    :param dt: The datetime object for timestamping.
    :param is_agent: Whether this is an agent pipeline.
    :param agent_name: Name of the agent (if applicable).
    :param component_name: Name of the component that triggered the breakpoint.
    :raises:
        ValueError: If the debug_path is not a string or a Path object.
        Exception: If saving the JSON state fails.
    """
    debug_path = Path(debug_path) if isinstance(debug_path, str) else debug_path
    if not isinstance(debug_path, Path):
        raise ValueError("Debug path must be a string or a Path object.")

    debug_path.mkdir(exist_ok=True)

    # Generate filename
    if is_agent:
        file_name = f"{agent_name}_{component_name}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}.json"
    else:
        file_name = f"{component_name}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}.json"

    try:
        with open(debug_path / file_name, "w") as f_out:
            json.dump(state, f_out, indent=2)
        logger.info(f"Pipeline state saved at: {file_name}")
    except Exception as e:
        logger.error(f"Failed to save pipeline state: {str(e)}")
        raise


def _save_state(
    *,
    inputs: Dict[str, Any],
    component_name: str,
    component_visits: Dict[str, int],
    debug_path: Optional[Union[str, Path]] = None,
    original_input_data: Optional[Dict[str, Any]] = None,
    ordered_component_names: Optional[List[str]] = None,
    agent_name: Optional[str] = None,
    main_pipeline_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save the pipeline state to a file.

    :param inputs: The current pipeline state inputs.
    :param component_name: The name of the component that triggered the breakpoint.
    :param component_visits: The visit count of the component that triggered the breakpoint.
    :param debug_path: The path to save the state to.
    :param original_input_data: The original input data.
    :param ordered_component_names: The ordered component names.
    :param main_pipeline_state: Dictionary containing main pipeline state with keys: "component_visits",
                                "ordered_component_names", "original_input_data", and "inputs".

    :returns:
        The dictionary containing the state of the pipeline containing the following keys:
        - input_data: The original input data passed to the pipeline.
        - timestamp: The timestamp of the breakpoint.
        - pipeline_breakpoint: The component name and visit count that triggered the breakpoint.
        - pipeline_state: The state of the pipeline when the breakpoint was triggered containing the following keys:
            - inputs: The current state of inputs for pipeline components.
            - component_visits: The visit count of the components when the breakpoint was triggered.
            - ordered_component_names: The order of components in the pipeline.
    """
    dt = datetime.now()

    # remove duplicated information
    if original_input_data:
        original_input_data.pop("main_pipeline_state", None)

    transformed_original_input_data = _transform_json_structure(original_input_data)
    transformed_inputs = _transform_json_structure(inputs)

    state = {
        # related to the main pipeline where the agent running as a breakpoint - only used with AgentBreakpoint
        "agent_name": agent_name if agent_name else None,
        "main_pipeline_state": _process_main_pipeline_state(main_pipeline_state) if agent_name else None,
        # breakpoint - information for the component that triggered the breakpoint, can also be an Agent
        "component_name": component_name,
        "input_data": _serialize_value_with_schema(transformed_original_input_data),  # original input data
        "timestamp": dt.isoformat(),
        "pipeline_breakpoint": {"component": component_name, "visits": component_visits[component_name]},
        "pipeline_state": {
            "inputs": _serialize_value_with_schema(transformed_inputs),  # current pipeline state inputs
            "component_visits": component_visits,
            "ordered_component_names": ordered_component_names,
        },
    }

    if not debug_path:
        return state

    is_agent = agent_name is not None
    _save_state_to_file(
        state=state,
        debug_path=debug_path,
        dt=dt,
        is_agent=is_agent,
        agent_name=agent_name,
        component_name=component_name,
    )

    return state


def _transform_json_structure(data: Union[Dict[str, Any], List[Any], Any]) -> Any:
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


def _handle_agent_break_point(
    *,
    break_point: AgentBreakpoint,
    component_name: str,
    component_inputs: Dict[str, Any],
    inputs: Dict[str, Any],
    component_visits: Dict[str, int],
    ordered_component_names: list,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle agent-specific breakpoint logic.

    :param break_point: The agent breakpoint to handle
    :param component_name: Name of the current component
    :param component_inputs: Inputs for the current component
    :param inputs: Global pipeline inputs
    :param component_visits: Component visit counts
    :param ordered_component_names: Ordered list of component names
    :param data: Original pipeline data
    :return: Updated component inputs
    """
    component_inputs["break_point"] = break_point

    # Store pipeline state for agent resume
    state_inputs_serialised = deepcopy(inputs)
    state_inputs_serialised[component_name] = deepcopy(component_inputs)
    component_inputs["main_pipeline_state"] = {
        "inputs": state_inputs_serialised,
        "component_visits": component_visits,
        "ordered_component_names": ordered_component_names,
        "original_input_data": data,
    }

    return component_inputs


def _check_regular_break_point(break_point: Breakpoint, component_name: str, component_visits: Dict[str, int]) -> bool:
    """
    Check if a regular breakpoint should be triggered.

    :param break_point: The breakpoint to check
    :param component_name: Name of the current component
    :param component_visits: Component visit counts
    :return: True if breakpoint should be triggered
    """
    return break_point.component_name == component_name and break_point.visit_count == component_visits[component_name]


def _trigger_break_point(
    *,
    component_name: str,
    component_inputs: Dict[str, Any],
    inputs: Dict[str, Any],
    component_visits: Dict[str, int],
    debug_path: Optional[Union[str, Path]],
    data: Dict[str, Any],
    ordered_component_names: list,
    pipeline_outputs: Dict[str, Any],
) -> None:
    """
    Trigger a breakpoint by saving state and raising exception.

    :param component_name: Name of the component where breakpoint is triggered
    :param component_inputs: Inputs for the current component
    :param inputs: Global pipeline inputs
    :param component_visits: Component visit counts
    :param debug_path: Path for debug files
    :param data: Original pipeline data
    :param ordered_component_names: Ordered list of component names
    :param pipeline_outputs: Current pipeline outputs
    :raises PipelineBreakpointException: When breakpoint is triggered
    """
    state_inputs_serialised = deepcopy(inputs)
    state_inputs_serialised[component_name] = deepcopy(component_inputs)
    _save_state(
        inputs=state_inputs_serialised,
        component_name=str(component_name),
        component_visits=component_visits,
        debug_path=debug_path,
        original_input_data=data,
        ordered_component_names=ordered_component_names,
    )

    msg = f"Breaking at component {component_name} at visit count {component_visits[component_name]}"
    raise BreakpointException(
        message=msg, component=component_name, state=state_inputs_serialised, results=pipeline_outputs
    )
