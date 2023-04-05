from typing import List, Any, Optional

from haystack.preview.dataclasses import Document


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


def eq_operation(fields, field_name, value):
    """
    Checks for equality between the document's metadata value and a fixed value.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the values are equal, False otherwise
    """
    if not field_name in fields:
        return False
    return fields[field_name] == value


def in_operation(fields, field_name, value):
    """
    Checks for whether the document's metadata value is present into the given list.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is included in the given list, False otherwise
    """
    if not field_name in fields:
        return False
    return fields[field_name] in value


def ne_operation(fields, field_name, value):
    """
    Checks for inequality between the document's metadata value and a fixed value.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the values are different, False otherwise
    """
    if not field_name in fields:
        return True
    return fields[field_name] != value


def nin_operation(fields, field_name, value):
    """
    Checks whether the document's metadata value is absent from the given list.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is not included in the given list, False otherwise
    """
    if not field_name in fields:
        return True
    return fields[field_name] not in value


def gt_operation(fields, field_name, value):
    """
    Checks whether the document's metadata value is (strictly) larger than the given value.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is strictly larger than the fixed value, False otherwise
    """
    if not field_name in fields:
        return False
    return fields[field_name] > value


def gte_operation(fields, field_name, value):
    """
    Checks whether the document's metadata value is larger than or equal to the given value.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is larger than or equal to the fixed value, False otherwise
    """
    if not field_name in fields:
        return False
    return fields[field_name] >= value


def lt_operation(fields, field_name, value):
    """
    Checks whether the document's metadata value is (strictly) smaller than the given value.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is strictly smaller than the fixed value, False otherwise
    """
    if not field_name in fields:
        return False
    return fields[field_name] < value


def lte_operation(fields, field_name, value):
    """
    Checks whether the document's metadata value is smaller than or equal to the given value.

    :param fields: all the document's metadata
    :param field_name: the field to test
    :param value; the fixed value to compare against
    :return: True if the document's value is smaller than or equal to the fixed value, False otherwise
    """
    if not field_name in fields:
        return False
    return fields[field_name] <= value


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


def match(conditions: Any, document: Document, _current_key: Optional[str] = None):
    """
    This method applies the filters to any given document and returns True when the documents
    metadata matches the filters, False otherwise.

    :param conditions: the filters dictionary.
    :param document: the document to test.
    :param _current_key: internal, don't use.
    :return: True if the document matches the filters, False otherwise
    """
    if isinstance(conditions, list):
        # The default operation for a list of sibling conditions is AND
        return and_operation(conditions=conditions, document=document, _current_key=_current_key)

    if isinstance(conditions, dict):
        # The default operation for a list of sibling conditions is AND
        if len(conditions.keys()) > 1:
            return and_operation(
                conditions=_conditions_as_list(conditions), document=document, _current_key=_current_key
            )

        field_key, field_value = list(conditions.items())[0]

        if field_key in LOGICAL_STATEMENTS.keys():
            # It's a nested logical statement (AND, OR, NOT)
            # Resolve the nested operator
            return LOGICAL_STATEMENTS[field_key](
                conditions=_conditions_as_list(field_value), document=document, _current_key=_current_key
            )
        if field_key in OPERATORS.keys():
            # It's a comparison operator (EQ, IN, GTE, ...)
            # Make sure the field to apply this operation on was specified.
            if not _current_key:
                raise ValueError(
                    "Filters can't start with an operator like $eq and $in, you have to specify which field to use first."
                )
            # Evaluate the operator
            return OPERATORS[field_key](fields=document.metadata, field_name=_current_key, value=field_value)

        if isinstance(field_value, list):
            # The default operator for a {key: [value]} filter is $in
            return in_operation(fields=document.metadata, field_name=field_key, value=field_value)
        else:
            # The default operator for a {key: value} filter is $eq
            return eq_operation(fields=document.metadata, field_name=field_key, value=field_value)

    if not _current_key:
        # If we reached here it means that conditions was neither a dict nor a list.
        raise ValueError("Filters must be dictionaries or lists.")
    return eq_operation(fields=document.metadata, field_name=_current_key, value=field_value)


def _conditions_as_list(conditions: Any) -> List[Any]:
    """
    Make sure all nested conditions are in a list.

    :param conditions: the conditions to transform into a list
    :returns: a list of filters
    """
    if isinstance(conditions, list):
        return conditions
    if isinstance(conditions, dict):
        return [{key: value} for key, value in conditions.items()]
    return [conditions]
