# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from datetime import datetime
from typing import Any, Dict, List, Union

import pandas as pd

from haystack.dataclasses import Document
from haystack.errors import FilterError


def document_matches_filter(filters: Dict[str, Any], document: Document) -> bool:
    """
    Return whether `filters` match the Document.

    For a detailed specification of the filters, refer to the
    `DocumentStore.filter_documents()` protocol documentation.
    """
    if "field" in filters:
        return _comparison_condition(filters, document)
    return _logic_condition(filters, document)


def _and(document: Document, conditions: List[Dict[str, Any]]) -> bool:
    return all(_comparison_condition(condition, document) for condition in conditions)


def _or(document: Document, conditions: List[Dict[str, Any]]) -> bool:
    return any(_comparison_condition(condition, document) for condition in conditions)


def _not(document: Document, conditions: List[Dict[str, Any]]) -> bool:
    return not _and(document, conditions)


LOGICAL_OPERATORS = {"NOT": _not, "OR": _or, "AND": _and}


def _equal(document_value: Any, filter_value: Any) -> bool:
    if isinstance(document_value, pd.DataFrame):
        document_value = document_value.to_json()

    if isinstance(filter_value, pd.DataFrame):
        filter_value = filter_value.to_json()

    return document_value == filter_value


def _not_equal(document_value: Any, filter_value: Any) -> bool:
    return not _equal(document_value=document_value, filter_value=filter_value)


def _greater_than(document_value: Any, filter_value: Any) -> bool:
    if document_value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    if isinstance(document_value, str) or isinstance(filter_value, str):
        try:
            document_value = datetime.fromisoformat(document_value)
            filter_value = datetime.fromisoformat(filter_value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(filter_value) in [list, pd.DataFrame]:
        msg = f"Filter value can't be of type {type(filter_value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return document_value > filter_value


def _greater_than_equal(document_value: Any, filter_value: Any) -> bool:
    if document_value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    return _equal(document_value=document_value, filter_value=filter_value) or _greater_than(
        document_value=document_value, filter_value=filter_value
    )


def _less_than(document_value: Any, filter_value: Any) -> bool:
    if document_value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    return not _greater_than_equal(document_value=document_value, filter_value=filter_value)


def _less_than_equal(document_value: Any, filter_value: Any) -> bool:
    if document_value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    return not _greater_than(document_value=document_value, filter_value=filter_value)


def _in(document_value: Any, filter_value: Any) -> bool:
    if not isinstance(filter_value, list):
        msg = (
            f"Filter value must be a `list` when using operator 'in' or 'not in', received type '{type(filter_value)}'"
        )
        raise FilterError(msg)
    return any(_equal(e, document_value) for e in filter_value)


def _not_in(document_value: Any, filter_value: Any) -> bool:
    return not _in(document_value=document_value, filter_value=filter_value)


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}


def _logic_condition(condition: Dict[str, Any], document: Document) -> bool:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    conditions: List[Dict[str, Any]] = condition["conditions"]
    return LOGICAL_OPERATORS[operator](document, conditions)


def _comparison_condition(condition: Dict[str, Any], document: Document) -> bool:
    if "field" not in condition:
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
        return _logic_condition(condition, document)
    field: str = condition["field"]

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)

    if "." in field:
        # Handles fields formatted like so:
        # 'meta.person.name'
        parts = field.split(".")
        document_value = getattr(document, parts[0])
        for part in parts[1:]:
            if part not in document_value:
                # If a field is not found we treat it as None
                document_value = None
                break
            document_value = document_value[part]
    elif field not in [f.name for f in fields(document)]:
        # Converted legacy filters don't add the `meta.` prefix, so we assume
        # that all filter fields that are not actual fields in Document are converted
        # filters.
        #
        # We handle this to avoid breaking compatibility with converted legacy filters.
        # This will be removed as soon as we stop supporting legacy filters.
        document_value = document.meta.get(field)
    else:
        document_value = getattr(document, field)
    operator: str = condition["operator"]
    filter_value: Any = condition["value"]
    return COMPARISON_OPERATORS[operator](filter_value=filter_value, document_value=document_value)


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
