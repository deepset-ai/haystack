# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from datetime import datetime
from typing import Any, Optional, Union

import dateutil.parser

from haystack.dataclasses import ByteStream, Document
from haystack.errors import FilterError


def raise_on_invalid_filter_syntax(filters: Optional[dict[str, Any]] = None) -> None:
    """
    Raise an error if the filter syntax is invalid.
    """
    if filters and ("operator" not in filters or "conditions" not in filters):
        msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
        raise FilterError(msg)


def document_matches_filter(filters: dict[str, Any], document: Union[Document, ByteStream]) -> bool:
    """
    Return whether `filters` match the Document or the ByteStream.

    For a detailed specification of the filters, refer to the
    `DocumentStore.filter_documents()` protocol documentation.
    """
    if "field" in filters:
        return _comparison_condition(condition=filters, document=document)
    return _logic_condition(condition=filters, document=document)


def _and(document: Union[Document, ByteStream], conditions: list[dict[str, Any]]) -> bool:
    return all(_comparison_condition(condition=condition, document=document) for condition in conditions)


def _or(document: Union[Document, ByteStream], conditions: list[dict[str, Any]]) -> bool:
    return any(_comparison_condition(condition=condition, document=document) for condition in conditions)


def _not(document: Union[Document, ByteStream], conditions: list[dict[str, Any]]) -> bool:
    return not _and(document=document, conditions=conditions)


LOGICAL_OPERATORS = {"NOT": _not, "OR": _or, "AND": _and}


def _equal(value: Any, filter_value: Any) -> bool:
    return value == filter_value


def _not_equal(value: Any, filter_value: Any) -> bool:
    return not _equal(value=value, filter_value=filter_value)


def _greater_than(value: Any, filter_value: Any) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    if isinstance(value, str) or isinstance(filter_value, str):
        try:
            value = _parse_date(value)
            filter_value = _parse_date(filter_value)
            value, filter_value = _ensure_both_dates_naive_or_aware(value, filter_value)
        except FilterError as exc:
            raise exc
    if isinstance(filter_value, list):
        msg = f"Filter value can't be of type {type(filter_value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return value > filter_value


def _parse_date(value):
    """Try parsing the value as an ISO format date, then fall back to dateutil.parser."""
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        try:
            return dateutil.parser.parse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc


def _ensure_both_dates_naive_or_aware(date1: datetime, date2: datetime) -> tuple[datetime, datetime]:
    """Ensure that both dates are either naive or aware."""
    # Both naive
    if date1.tzinfo is None and date2.tzinfo is None:
        return date1, date2

    # Both aware
    if date1.tzinfo is not None and date2.tzinfo is not None:
        return date1, date2

    # One naive, one aware
    if date1.tzinfo is None:
        date1 = date1.replace(tzinfo=date2.tzinfo)
    else:
        date2 = date2.replace(tzinfo=date1.tzinfo)
    return date1, date2


def _greater_than_equal(value: Any, filter_value: Any) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    return _equal(value=value, filter_value=filter_value) or _greater_than(value=value, filter_value=filter_value)


def _less_than(value: Any, filter_value: Any) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    return not _greater_than_equal(value=value, filter_value=filter_value)


def _less_than_equal(value: Any, filter_value: Any) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    return not _greater_than(value=value, filter_value=filter_value)


def _in(value: Any, filter_value: Any) -> bool:
    if not isinstance(filter_value, list):
        msg = (
            f"Filter value must be a `list` when using operator 'in' or 'not in', received type '{type(filter_value)}'"
        )
        raise FilterError(msg)
    return any(_equal(e, value) for e in filter_value)


def _not_in(value: Any, filter_value: Any) -> bool:
    return not _in(value=value, filter_value=filter_value)


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


def _logic_condition(condition: dict[str, Any], document: Union[Document, ByteStream]) -> bool:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    conditions: list[dict[str, Any]] = condition["conditions"]
    return LOGICAL_OPERATORS[operator](document=document, conditions=conditions)


def _comparison_condition(condition: dict[str, Any], document: Union[Document, ByteStream]) -> bool:
    if "field" not in condition:
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
        return _logic_condition(condition=condition, document=document)
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
    return COMPARISON_OPERATORS[operator](filter_value=filter_value, value=document_value)
