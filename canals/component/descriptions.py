from typing import Dict, Any


def find_component_inputs(component: Any) -> Dict[str, Dict[str, Any]]:
    """
    Returns a mapping of input names to their expected types and optionality for a given component.

    :param component: The target component to introspect.
    :return: A dictionary where keys are input names, with each key's value being another dictionary
    containing 'type' (the data type expected) and 'is_optional' (a boolean indicating if the input is optional).

    :raise: Throws a ValueError if the class of component instance is not appropriately decorated with @component.
    """
    if not hasattr(component, "__canals_input__"):
        raise ValueError(
            f"Component {component} does not have defined inputs or is improperly decorated. "
            "Ensure it is a valid @component with declared inputs."
        )

    return {
        name: {"type": socket.type, "is_optional": socket.is_optional}
        for name, socket in component.__canals_input__.items()
    }


def find_component_outputs(component: Any) -> Dict[str, Dict[str, Any]]:
    """
    Returns a mapping of component output names to their expected types.

    :param component: The component being examined for its outputs.
    :return: A dictionary where each key is an output name and the value is a dictionary with a 'type' key
    indicating the data type of the output.

    :raise: Throws a ValueError if the class of component instance is not appropriately decorated with @component.
    """
    if not hasattr(component, "__canals_output__"):
        raise ValueError(
            f"The specified component {component} does not have defined outputs or is not properly decorated. "
            "Check that it is a valid @component with outputs specified."
        )

    return {name: {"type": socket.type} for name, socket in component.__canals_output__.items()}
