from typing import List, Any

import numpy as np
import pandas as pd

from haystack.preview.document_stores.errors import StoreError
from haystack.preview.dataclasses import Document


GT_TYPES = (int, float, np.number)
IN_TYPES = (list, set, tuple)


class MemoryDocumentStoreFilterError(StoreError):
    pass


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
    for condition in conditions:
        if not match(conditions=condition, document=document, _current_key=_current_key):
            return False
    return True


def or_operation(conditions: List[Any], document: Document, _current_key: str):
    """
    Applies an OR to all the nested conditions.

    :param conditions: the filters dictionary.
    :param document: the document to test.
    :param _current_key: internal, don't use.
    :return: True if the document matches ano of the filters, False otherwise
    """
    for condition in conditions:
        if match(conditions=condition, document=document, _current_key=_current_key):
            return True
    return False


def _safe_eq(first: Any, second: Any) -> bool:
    """
    Compares objects for equality, even np.ndarrays and pandas DataFrames.
    """
    if type(first) != type(second):
        return False

    if isinstance(first, pd.DataFrame):
        return first.equals(second)

    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)

    return first == second


def _safe_gt(first: Any, second: Any) -> bool:
    """
    Checks if first is bigger than second.

    Works only for numerical values and dates. Strings, lists, tables and tensors all raise exceptions.
    """
    if not isinstance(first, GT_TYPES) or not isinstance(second, GT_TYPES):
        raise MemoryDocumentStoreFilterError(
            f"Can't evaluate '{type(first).__name__} > {type(second).__name__}'. "
            f"Convert these values into one of the following types: {[type_.__name__ for type_ in GT_TYPES]}"
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
        raise MemoryDocumentStoreFilterError("$in accepts only iterable values like lists, sets and tuples.")

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


def match(conditions: Any, document: Document, _current_key=None):
    """
    This method applies the filters to any given document and returns True when the documents
    metadata matches the filters, False otherwise.

    :param conditions: the filters dictionary.
    :param document: the document to test.
    :return: True if the document matches the filters, False otherwise
    """
    if isinstance(conditions, dict):
        # Check for malformed filters, like {"name": {"year": "2020"}}
        if _current_key and any(key not in RESERVED_KEYS for key in conditions.keys()):
            raise ValueError(
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
                raise ValueError(
                    "Filters can't start with an operator like $eq and $in. You have to specify the field name first. "
                    "See the examples in the documentation."
                )
            return OPERATORS[field_key](fields=document.flatten(), field_name=_current_key, value=field_value)

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
            return in_operation(fields=document.flatten(), field_name=_current_key, value=conditions)

    if _current_key:
        # The default operator for a {key: value} filter is $eq
        return eq_operation(fields=document.flatten(), field_name=_current_key, value=conditions)

    raise ValueError("Filters must be dictionaries or lists. See the examples in the documentation.")


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
