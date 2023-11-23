from typing import List, Any, Union, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from haystack.preview.dataclasses import Document
from haystack.preview.errors import FilterError


GT_TYPES = (int, float, np.number)
IN_TYPES = (list, set, tuple)


def not_operation(conditions: List[Any], document: Document, _current_key: str):
    """
    Applies a NOT to all the nested conditions.

    :param conditions: the filters dictionary.
    :param document: the document to test.
    :param _current_key: internal, don't use.
    :return: True if the document matches the negated filters, False otherwise
    """
    return not and_operation(conditions=conditions, document=document, _current_key=_current_key)


def and_operation(conditions: List[Any], document: Document, _current_key: str):
    """
    Applies an AND to all the nested conditions.

    :param conditions: the filters dictionary.
    :param document: the document to test.
    :param _current_key: internal, don't use.
    :return: True if the document matches all the filters, False otherwise
    """
    return all(
        document_matches_filter(conditions=condition, document=document, _current_key=_current_key)
        for condition in conditions
    )


def or_operation(conditions: List[Any], document: Document, _current_key: str):
    """
    Applies an OR to all the nested conditions.

    :param conditions: the filters dictionary.
    :param document: the document to test.
    :param _current_key: internal, don't use.
    :return: True if the document matches any of the filters, False otherwise
    """
    return any(
        document_matches_filter(conditions=condition, document=document, _current_key=_current_key)
        for condition in conditions
    )


def _safe_eq(first: Any, second: Any) -> bool:
    """
    Compares objects for equality, even np.ndarrays and pandas DataFrames.
    """

    if isinstance(first, pd.DataFrame):
        first = first.to_json()

    if isinstance(second, pd.DataFrame):
        second = second.to_json()

    if isinstance(first, np.ndarray):
        first = first.tolist()

    if isinstance(second, np.ndarray):
        second = second.tolist()

    return first == second


def _safe_gt(first: Any, second: Any) -> bool:
    """
    Checks if first is bigger than second.

    Works only for numerical values and dates in ISO format (YYYY-MM-DD). Strings, lists, tables and tensors all raise exceptions.
    """
    if not isinstance(first, GT_TYPES) or not isinstance(second, GT_TYPES):
        try:
            first = datetime.fromisoformat(first)
            second = datetime.fromisoformat(second)
        except (ValueError, TypeError):
            raise FilterError(
                f"Can't evaluate '{type(first).__name__} > {type(second).__name__}'. "
                f"Convert these values into one of the following types: {[type_.__name__ for type_ in GT_TYPES]} "
                f"or a datetime string in ISO 8601 format."
            )
    return bool(first > second)


def eq_operation(fields, field_name, value):
    """
    Checks for equality between the document's field value value and a fixed value.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value: the fixed value to compare against
    :return: True if the values are equal, False otherwise
    """
    if not field_name in fields:
        return False

    return _safe_eq(fields[field_name], value)


def in_operation(fields, field_name, value):
    """
    Checks for whether the document's field value value is present into the given list.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is included in the given list, False otherwise
    """
    if not field_name in fields:
        return False

    if not isinstance(value, IN_TYPES):
        raise FilterError("$in accepts only iterable values like lists, sets and tuples.")

    return any(_safe_eq(fields[field_name], v) for v in value)


def ne_operation(fields, field_name, value):
    """
    Checks for inequality between the document's field value value and a fixed value.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the values are different, False otherwise
    """
    return not eq_operation(fields, field_name, value)


def nin_operation(fields, field_name, value):
    """
    Checks whether the document's field value value is absent from the given list.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is not included in the given list, False otherwise
    """
    return not in_operation(fields, field_name, value)


def gt_operation(fields, field_name, value):
    """
    Checks whether the document's field value value is (strictly) larger than the given value.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is strictly larger than the fixed value, False otherwise
    """
    if not field_name in fields:
        return False
    return _safe_gt(fields[field_name], value)


def gte_operation(fields, field_name, value):
    """
    Checks whether the document's field value value is larger than or equal to the given value.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is larger than or equal to the fixed value, False otherwise
    """
    return gt_operation(fields, field_name, value) or eq_operation(fields, field_name, value)


def lt_operation(fields, field_name, value):
    """
    Checks whether the document's field value value is (strictly) smaller than the given value.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is strictly smaller than the fixed value, False otherwise
    """
    if not field_name in fields:
        return False
    return not _safe_gt(fields[field_name], value) and not _safe_eq(fields[field_name], value)


def lte_operation(fields, field_name, value):
    """
    Checks whether the document's field value value is smaller than or equal to the given value.

    :param fields: all the document's field value
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is smaller than or equal to the fixed value, False otherwise
    """
    if not field_name in fields:
        return False
    return not _safe_gt(fields[field_name], value)


LOGICAL_STATEMENTS = {"$not": not_operation, "$and": and_operation, "$or": or_operation}
OPERATORS = {
    "$eq": eq_operation,
    "$in": in_operation,
    "$ne": ne_operation,
    "$nin": nin_operation,
    "$gt": gt_operation,
    "$gte": gte_operation,
    "$lt": lt_operation,
    "$lte": lte_operation,
}
RESERVED_KEYS = [*LOGICAL_STATEMENTS.keys(), *OPERATORS.keys()]


def document_matches_filter(conditions: Union[Dict, List], document: Document, _current_key=None):
    """
    Check if a document's metadata matches the provided filter conditions.

    This function evaluates the specified conditions against the metadata of the given document
    and returns True if the conditions are met, otherwise it returns False.

    :param conditions: A dictionary or list containing filter conditions to be applied to the document's metadata.
    :param document: The document whose metadata will be evaluated against the conditions.
    :param _current_key: internal parameter, don't use.
    :return: True if the document's metadata matches the filter conditions, False otherwise.
    """
    if isinstance(conditions, dict):
        # Check for malformed filters, like {"name": {"year": "2020"}}
        if _current_key and any(key not in RESERVED_KEYS for key in conditions.keys()):
            raise FilterError(
                f"This filter ({{{_current_key}: {conditions}}}) seems to be malformed. "
                "Comparisons between dictionaries are not currently supported. "
                "Check the documentation to learn more about filters syntax."
            )

        if len(conditions.keys()) > 1:
            # The default operation for a list of sibling conditions is $and
            return and_operation(conditions=_list_conditions(conditions), document=document, _current_key=_current_key)

        field_key, field_value = list(conditions.items())[0]

        # Nested logical statement ($and, $or, $not)
        if field_key in LOGICAL_STATEMENTS.keys():
            return LOGICAL_STATEMENTS[field_key](
                conditions=_list_conditions(field_value), document=document, _current_key=_current_key
            )

        # A comparison operator ($eq, $in, $gte, ...)
        if field_key in OPERATORS.keys():
            if not _current_key:
                raise FilterError(
                    "Filters can't start with an operator like $eq and $in. You have to specify the field name first. "
                    "See the examples in the documentation."
                )
            return OPERATORS[field_key](fields=document.to_dict(), field_name=_current_key, value=field_value)

        # Otherwise fall back to the defaults
        conditions = _list_conditions(field_value)
        _current_key = field_key

    # Defaults for implicit filters
    if isinstance(conditions, list):
        if all(isinstance(cond, dict) for cond in conditions):
            # The default operation for a list of sibling conditions is $and
            return and_operation(conditions=_list_conditions(conditions), document=document, _current_key=_current_key)
        else:
            # The default operator for a {key: [value1, value2]} filter is $in
            return in_operation(fields=document.to_dict(), field_name=_current_key, value=conditions)

    if _current_key:
        # The default operator for a {key: value} filter is $eq
        return eq_operation(fields=document.to_dict(), field_name=_current_key, value=conditions)

    raise FilterError("Filters must be dictionaries or lists. See the examples in the documentation.")


def _list_conditions(conditions: Any) -> List[Any]:
    """
    Make sure all nested conditions are not dictionaries or single values, but always lists.

    :param conditions: the conditions to transform into a list
    :returns: a list of filters
    """
    if isinstance(conditions, list):
        return conditions
    if isinstance(conditions, dict):
        return [{key: value} for key, value in conditions.items()]
    return [conditions]


def convert(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a filter declared using the legacy style into the new style.
    This is mostly meant to ease migration from Haystack 1.x to 2.x for developers
    of Document Stores and Components that use filters.

    This function doesn't verify if `filters` are declared using the legacy style.

    Example usage:
    ```python
    legacy_filter = {
        "$and": {
            "type": {"$eq": "article"},
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {"genre": {"$in": ["economy", "politics"]}, "publisher": {"$eq": "nytimes"}},
        }
    }
    assert convert(legacy_filter) == {
        "operator": "AND",
        "conditions": [
            {"field": "type", "operator": "==", "value": "article"},
            {"field": "date", "operator": ">=", "value": "2015-01-01"},
            {"field": "date", "operator": "<", "value": "2021-01-01"},
            {"field": "rating", "operator": ">=", "value": 3},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "genre", "operator": "in", "value": ["economy", "politics"]},
                    {"field": "publisher", "operator": "==", "value": "nytimes"},
                ],
            },
        ],
    }
    ```
    """
    if not isinstance(filters, dict):
        msg = f"Can't convert filters from type '{type(filters)}'"
        raise ValueError(msg)

    converted = _internal_convert(filters)
    if "conditions" not in converted:
        # This is done to handle a corner case when filter is really simple like so:
        #   {"text": "A Foo Document 1"}
        # The root '$and' operator is implicit so the conversion doesn't handle
        # it and it must be added explicitly like so.
        # This only happens for simple filters like the one above.
        return {"operator": "AND", "conditions": [converted]}
    return converted


def _internal_convert(filters: Union[List[Any], Dict[str, Any]], previous_key=None) -> Any:
    """
    Recursively convert filters from legacy to new style.
    """
    conditions = []

    if isinstance(filters, list) and (result := _handle_list(filters, previous_key)) is not None:
        return result

    if not isinstance(filters, dict):
        return _handle_non_dict(filters, previous_key)

    for key, value in filters.items():
        if (
            previous_key is not None
            and previous_key not in ALL_LEGACY_OPERATORS_MAPPING
            and key not in ALL_LEGACY_OPERATORS_MAPPING
        ):
            msg = f"This filter ({filters}) seems to be malformed."
            raise FilterError(msg)
        if key not in ALL_LEGACY_OPERATORS_MAPPING:
            converted = _internal_convert(value, previous_key=key)
            if isinstance(converted, list):
                conditions.extend(converted)
            else:
                conditions.append(converted)
        elif key in LEGACY_LOGICAL_OPERATORS_MAPPING:
            if previous_key not in ALL_LEGACY_OPERATORS_MAPPING and isinstance(value, list):
                converted = [_internal_convert({previous_key: v}) for v in value]
                conditions.append({"operator": ALL_LEGACY_OPERATORS_MAPPING[key], "conditions": converted})
            else:
                converted = _internal_convert(value, previous_key=key)
                if key == "$not" and type(converted) not in [dict, list]:
                    # This handles a corner when '$not' is used like this:
                    # '{"page": {"$not": 102}}'
                    # Without this check we would miss the implicit '$eq'
                    converted = {"field": previous_key, "operator": "==", "value": value}
                if not isinstance(converted, list):
                    converted = [converted]
                conditions.append({"operator": ALL_LEGACY_OPERATORS_MAPPING[key], "conditions": converted})
        elif key in LEGACY_COMPARISON_OPERATORS_MAPPING:
            conditions.append({"field": previous_key, "operator": ALL_LEGACY_OPERATORS_MAPPING[key], "value": value})

    if len(conditions) == 1:
        return conditions[0]

    if previous_key is None:
        return {"operator": "AND", "conditions": conditions}

    return conditions


def _handle_list(filters, previous_key):
    if previous_key in LEGACY_LOGICAL_OPERATORS_MAPPING:
        return [_internal_convert(f) for f in filters]
    elif previous_key not in LEGACY_COMPARISON_OPERATORS_MAPPING:
        return {"field": previous_key, "operator": "in", "value": filters}
    return None


def _handle_non_dict(filters, previous_key):
    if previous_key not in ALL_LEGACY_OPERATORS_MAPPING:
        return {"field": previous_key, "operator": "==", "value": filters}
    return filters


# Operator mappings from legacy style to new one
LEGACY_LOGICAL_OPERATORS_MAPPING = {"$and": "AND", "$or": "OR", "$not": "NOT"}

LEGACY_COMPARISON_OPERATORS_MAPPING = {
    "$eq": "==",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$in": "in",
    "$nin": "not in",
}

ALL_LEGACY_OPERATORS_MAPPING = {**LEGACY_LOGICAL_OPERATORS_MAPPING, **LEGACY_COMPARISON_OPERATORS_MAPPING}
