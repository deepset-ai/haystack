from typing import Any, Dict, List

from haystack.dataclasses import Answer, Document, ExtractedAnswer, GeneratedAnswer, ByteStream, ChatMessage


def deserialize_dataclass(data_type: str, data_value: Any) -> Any:
    """
    Deserialize a dataclass from dictionary to dataclass using the from_dict() method.

    :param: data_type: Dataclass type
    :param: data_value: Dictionary to deserialize
    :return: Deserialized Dataclass with values
    """
    type_mapping = {
        "Document": Document,
        "Answer": Answer,
        "ExtractedAnswer": ExtractedAnswer,
        "GeneratedAnswer": GeneratedAnswer,
        "ChatMessage": ChatMessage,
        "ByteStream": ByteStream,
    }

    if data_type in type_mapping:
        dataclass = type_mapping[data_type]
        return dataclass.from_dict(data_value)  # type: ignore[attr-defined]
    else:
        return data_value  # Fallback if the type isn't found


def convert_component_output_to_dict(component_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts the output of a single component to a serializable dictionary format using the to_dict() method of the dataclass.
    The type of the dataclass is also stored.

    :param output: A dictionary containing the output of the component.
    :return: A dictionary with component's output converted into a serializable format.
    """
    type_mapping = {
        Document: "Document",
        Answer: "Answer",
        ExtractedAnswer: "ExtractedAnswer",
        GeneratedAnswer: "GeneratedAnswer",
        ChatMessage: "ChatMessage",
        ByteStream: "ByteStream",
    }

    for key, value in list(component_output.items()):
        if isinstance(value, list) and any(isinstance(item, tuple(type_mapping.keys())) for item in value):
            serialized_values = []
            for file in value:
                file_type = next((type_str for cls, type_str in type_mapping.items() if isinstance(file, cls)), None)
                serialized_values.append((file_type, file.to_dict()))
            component_output[key] = serialized_values

    return component_output


def convert_component_output_from_dict(component_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts the dataclasses in the output of a single component back to corresponding dataclasses from the serialized dictionary format.

    :param output: A dictionary containing the serialized output of the component.
    :return: A dictionary with deserialized component output.
    """

    for key, value in component_output.items():
        if isinstance(value, list):
            deserialized_values = []
            if any(isinstance(item, tuple) and len(item) == 2 for item in value):
                for item in value:
                    if isinstance(item, tuple) and len(item) == 2:
                        data_type, data_value = item
                        deserialized_values.append(deserialize_dataclass(data_type, data_value))
                    else:
                        deserialized_values.append(item)
                component_output[key] = deserialized_values
    return component_output


def convert_component_outputs_to_dict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterates over multiple outputs of a component and converts the outputs to serialized dictionaries.

    :param data: A list of dictionaries containing the outputs of the components.
    :return: A list of dictionaries with component outputs converted into serialized dictionaries.
    """
    converted_data = []
    for output in data:
        converted_data.append(convert_component_output_to_dict(output))
    return converted_data


def convert_component_outputs_from_dict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterates over multiple outputs of a component and deserializes the outputs from dictionaries.

    :param data: A list of dictionaries containing the outputs of the components.
    :return: A list of dictionaries with component outputs converted back to deserialized dataclasses.
    """
    converted_data = []
    for output in data:
        converted_data.append(convert_component_output_from_dict(output))
    return converted_data


def convert_pipeline_outputs_to_dict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterates over the outputs of a Pipeline and converts the outputs to serialized dictionaries.

    :param data: A list of dictionaries containing the outputs of the pipeline.
    :return: A list of dictionaries with pipeline outputs converted into serialized dictionaries.
    """
    for output in data:
        for component_name, component_output in output.items():
            output[component_name] = convert_component_output_to_dict(component_output)
    return data


def convert_pipeline_outputs_from_dict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterates over the outputs of a Pipeline and deserializes the outputs from dictionaries.

    :param data: A list of dictionaries containing the outputs of the pipeline.
    :return: A list of dictionaries with pipeline outputs converted back to deserialized dataclasses.
    """
    for output in data:
        for component_name, component_output in output.items():
            output[component_name] = convert_component_output_from_dict(component_output)
    return data
