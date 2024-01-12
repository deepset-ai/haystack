from typing import Union, Any, Dict

FILTER_OPERATORS = ["$and", "$or", "$not", "$eq", "$in", "$gt", "$gte", "$lt", "$lte"]
EXCLUDE_FROM_METADATA_PREPEND = ["id", "embedding"]

METADATA_FIELD_NAME = "meta"


def mongo_filter_converter(filter) -> Dict[str, Any]:
    if not filter:
        return {}
    else:
        filter = _target_filter_to_metadata(filter, METADATA_FIELD_NAME)
        filter = _and_or_to_list(filter)
        return filter


def _target_filter_to_metadata(filter, metadata_field_name) -> Union[Dict[str, Any], list]:
    """
    Returns a new filter with any non-operator, non-excluded keys renamed so that the metadata
    field name is prepended. Does not mutate input filter.

    Example:

    {
        "$and": {
            "url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes",
            "_split_id": 0
        }
    }

    will be replaced with:

    {
        "$and": {
            "meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes",
            "meta._split_id": 0
        }
    }

    """
    if isinstance(filter, dict):
        updated_dict = {}
        for key, value in filter.items():
            if key not in FILTER_OPERATORS + EXCLUDE_FROM_METADATA_PREPEND:
                key = f"{metadata_field_name}.{key}"

            if isinstance(value, (dict, list)):
                updated_dict[key] = _target_filter_to_metadata(value, metadata_field_name)
            else:
                updated_dict[key] = value
        return updated_dict
    elif isinstance(filter, list):
        return [_target_filter_to_metadata(item, metadata_field_name) for item in filter]
    return filter


def _and_or_to_list(filter) -> Union[Dict[str, Any], list]:
    """
    Returns a new filter replacing any dict values associated with "$and" or "$or" keys
    with a list. Does not mutate input filter.

    Example:

    {
        "$and": {
            "url": {"$eq": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            "_split_id": {"$eq": 0},
        },
    }

    will be replaced with:

    {
        "$and": [
            {"url": {"$eq": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}},
            {"_split_id": {"$eq": 0}},
        ]
    }
    """
    if isinstance(filter, dict):
        updated_dict = filter.copy()
        if "$and" in updated_dict and isinstance(filter["$and"], dict):
            updated_dict["$and"] = [{key: value} for key, value in filter["$and"].items()]
        if "$or" in updated_dict and isinstance(filter["$or"], dict):
            updated_dict["$or"] = [{key: value} for key, value in filter["$or"].items()]
        return {key: _and_or_to_list(value) for key, value in updated_dict.items()}
    elif isinstance(filter, list):
        return [_and_or_to_list(item) for item in filter]
    else:
        return filter
