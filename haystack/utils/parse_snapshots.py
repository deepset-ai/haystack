import json
from pathlib import Path


def get_component_input(json_input, component_name):
    """
    Get the input for a specific component from pipeline_state.inputs

    Args:
        json_input: The JSON data (string, dict, or Path to JSON file)
        component_name: Name of the component to get input for

    Returns:
        The input data for the component, or None if not found
    """
    # Handle different input types
    if isinstance(json_input, (str, Path)):
        # Check if it's a file path
        path = Path(json_input)
        if path.exists() and path.is_file():
            # Read from file
            with open(path, "r") as f:
                data = json.load(f)
        else:
            # Treat as JSON string
            data = json.loads(json_input)
    else:
        # Already a dict
        data = json_input

    # Navigate to pipeline_state.inputs.serialized_data
    pipeline_state = data.get("pipeline_state", {})
    inputs = pipeline_state.get("inputs", {})
    serialized_data = inputs.get("serialized_data", {})

    # Return the component's input data
    return serialized_data.get(component_name)


# Example usage:
# From file path:
# component_input = get_component_input("path/to/pipeline.json", "prompt_builder")
#
# From JSON string:
# component_input = get_component_input(json_string, "prompt_builder")
#
# From dict:
# component_input = get_component_input(json_dict, "prompt_builder")
