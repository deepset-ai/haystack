from copy import deepcopy
from typing import Any, Dict, Generator, List, Union

from haystack import Answer, Document, ExtractedAnswer, GeneratedAnswer
from haystack.dataclasses.byte_stream import ByteStream


def convert_objects_to_dict(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterates over the outputs of a component/Pipeline and converts the dataclasses to python types using the to_dict() method of each dataclass.

    :param data: A list of dictionaries containing the outputs of the components.
    :return: A list of dictionaries with component outputs converted into a serializable format.
    """

    unpacked_data = []

    # Iterating through each dictionary in the list
    for item in data:
        grouped_data: Dict[Any, Any] = {}
        # Iterating through values in the dictionary
        for value in item.values():
            # Iterating through key-value pairs in the nested dictionary
            for sub_key, sub_value in value.items():
                if sub_key in grouped_data:
                    if isinstance(grouped_data[sub_key], list):
                        # Extend the existing list with new values
                        grouped_data[sub_key].extend((type(sub_value), sub_value))
                    else:
                        # Convert the Haystack objects to dictionary
                        grouped_data[sub_key].extend((type(sub_value), sub_value.to_dict()))
                else:
                    # Create a new list with values if the key is not in the grouped data
                    grouped_data[sub_key] = sub_value[:]
        unpacked_data.append(grouped_data)
    return unpacked_data


def convert_dict_to_objects(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterates over the outputs of a component/Pipeline and converts the python dictionaries to dataclasses using the from_dict() method of each dataclass.

    :param data: A list of dictionaries containing the outputs of the components.
    :return: A list of dictionaries with component outputs converted back to dataclasses from the serialized format.
    """
    unpacked_data = []

    for item in data:
        grouped_data: Dict[Any, Any] = {}
        for sub_key, sub_value in item.items():
            data_type, data_value = sub_value
            if sub_key in grouped_data:
                if isinstance(data_type, list):
                    grouped_data[sub_key].extend(data_value)
                # Convert the dictionaries to Haystack objects
                elif issubclass(data_type, (Document, Answer, GeneratedAnswer, ExtractedAnswer, ByteStream)):
                    grouped_data[sub_key].extend(data_type.from_dict(data_value))
            else:
                # Create a new list with values if the key is not in the grouped data
                grouped_data[sub_key] = sub_value[:]
        unpacked_data.append(grouped_data)
    return unpacked_data


def group_values(data: Dict[str, Any]):
    """
    Groups values from a list of dictionaries containing nested dictionaries.
    Used to unpack the results from the ouputs generated from a Component or Pipeline.

    :param data: A list of dictionaries
    :return: A dictionary with keys from nested dictionaries and values as a
            flattened list of corresponding values.
    """
    grouped_data = {}

    # Iterating through each dictionary in the list
    for item in data:
        # Iterating through values in the dictionary
        for value in item.values():
            # Iterating through key-value pairs in the nested dictionary
            for sub_key, sub_value in value.items():
                if sub_key in grouped_data:
                    if isinstance(grouped_data[sub_key], list):
                        # Extend the existing list with new values
                        grouped_data[sub_key].extend(sub_value)
                else:
                    # Create a new list with values if the key is not in the grouped data
                    grouped_data[sub_key] = sub_value[:]
    return grouped_data


def get_grouped_values(grouped_data: Dict[str, Any], key_to_find: str) -> Union[List[Any], None]:
    """
    Retrieves the values associated with a given key from grouped data.

    :param grouped_data (dict): Dictionary containing grouped data
    :param key_to_find (str): Key to search within the grouped data

    :return: List of values corresponding to the provided key.
                    Returns None if the key is not found.
    """
    if key_to_find in grouped_data:
        return grouped_data[key_to_find]
    return None


def flatten_list(nested_list: List[Any]) -> Generator[List[Any], List[Any], None]:
    """
    Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, leaving the original list unchanged.

    :param nested_list: A nested list that needs to be flattened

    :return: Yields the elements of the flattened list
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist
