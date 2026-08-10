# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from datetime import datetime
from typing import Any

import dateutil.parser

from haystack.dataclasses import ByteStream, Document
from haystack.errors import FilterError


def document_matches_filter(
    filters: dict[str, Any], document: Document | ByteStream, *, strict_datetime_comparison: bool = False
) -> bool:
    """
    Return whether `filters` match the Document or the ByteStream.

    For a detailed specification of the filters, refer to the
    `DocumentStore.filter_documents()` protocol documentation.

    :param strict_datetime_comparison:
        If `True`, timezone-naive and timezone-aware datetimes never match each other.
        If `False` (the default), the timezone from the aware datetime is copied to the naive one before comparing.
    """
    if "field" in filters:
        return _comparison_condition(
            condition=filters, document=document, strict_datetime_comparison=strict_datetime_comparison
        )
    return _logic_condition(condition=filters, document=document, strict_datetime_comparison=strict_datetime_comparison)


def _and(document: Document | ByteStream, conditions: list[dict[str, Any]], strict_datetime_comparison: bool) -> bool:
    return all(
        _comparison_condition(
            condition=condition, document=document, strict_datetime_comparison=strict_datetime_comparison
        )
        for condition in conditions
    )


def _or(document: Document | ByteStream, conditions: list[dict[str, Any]], strict_datetime_comparison: bool) -> bool:
    return any(
        _comparison_condition(
            condition=condition, document=document, strict_datetime_comparison=strict_datetime_comparison
        )
        for condition in conditions
    )


def _not(document: Document | ByteStream, conditions: list[dict[str, Any]], strict_datetime_comparison: bool) -> bool:
    return not _and(document=document, conditions=conditions, strict_datetime_comparison=strict_datetime_comparison)


LOGICAL_OPERATORS = {"NOT": _not, "OR": _or, "AND": _and}


def _equal(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    if value == filter_value:
        return True
    return _dates_are_equal(value=value, filter_value=filter_value, strict=strict_datetime_comparison)


def _not_equal(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    return not _equal(value=value, filter_value=filter_value, strict_datetime_comparison=strict_datetime_comparison)


def _looks_like_iso_date(value: Any) -> bool:
    """
    Cheaply reject values that can't be a full ISO 8601 date, before paying for a real parse.

    Deliberately conservative: a complete `YYYY-MM-DD` prefix is required, so partial dates such as
    "2025" are not read as dates, and the ISO 8601 basic format ("20250203") is not recognised.
    """
    return (
        isinstance(value, str)
        and len(value) >= 10
        and value[:4].isdigit()
        and value[4] == "-"
        and value[5:7].isdigit()
        and value[7] == "-"
        and value[8:10].isdigit()
        and (len(value) == 10 or value[10] in {"T", "t", " "})
    )


def _parse_iso_date(value: str) -> datetime | None:
    """Parse a strict ISO 8601 string, returning None if it isn't one."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Python 3.10's fromisoformat rejects valid ISO 8601 spellings that later versions accept,
        # most notably a trailing "Z".
        try:
            return dateutil.parser.isoparse(value)
        except (ValueError, OverflowError):
            return None


def _dates_are_equal(value: Any, filter_value: Any, strict: bool) -> bool:
    """
    Return whether both values are ISO 8601 datetimes denoting the same point in time.

    Only strict ISO 8601 strings and `datetime` objects are considered. Mixed-awareness datetimes are
    reconciled unless strict mode is enabled. Returns False for anything that isn't a pair of comparable
    datetimes, so that `==` stays total.
    """
    parsed: list[datetime] = []
    for candidate in (value, filter_value):
        if isinstance(candidate, datetime):
            parsed.append(candidate)
            continue
        if not _looks_like_iso_date(candidate):
            return False
        date = _parse_iso_date(candidate)
        if date is None:
            return False
        parsed.append(date)

    first, second = parsed
    if strict and (first.tzinfo is None) != (second.tzinfo is None):
        return False
    first, second = _ensure_both_dates_naive_or_aware(first, second)
    return first == second


def _prepare_ordering_comparison(
    value: Any, filter_value: Any, strict_datetime_comparison: bool
) -> tuple[Any, Any, bool]:
    """
    Normalize both values for ordering comparisons, parsing strings as dates.

    :returns:
        A tuple containing the normalized value, normalized filter value, and whether the values are comparable.
        The boolean is `False` when strict datetime comparison is enabled and one datetime is timezone-naive while
        the other is timezone-aware; otherwise it is `True`.
    """
    if isinstance(value, str) or isinstance(filter_value, str):
        if not isinstance(value, datetime):
            value = _parse_date(value)
        if not isinstance(filter_value, datetime):
            filter_value = _parse_date(filter_value)

    if isinstance(value, datetime) and isinstance(filter_value, datetime):
        if strict_datetime_comparison and (value.tzinfo is None) != (filter_value.tzinfo is None):
            return value, filter_value, False
        value, filter_value = _ensure_both_dates_naive_or_aware(value, filter_value)

    if isinstance(filter_value, list):
        msg = f"Filter value can't be of type {type(filter_value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return value, filter_value, True


def _greater_than(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    value, filter_value, comparable = _prepare_ordering_comparison(
        value=value, filter_value=filter_value, strict_datetime_comparison=strict_datetime_comparison
    )
    if not comparable:
        return False
    return value > filter_value


def _parse_date(value: str) -> datetime:
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


def _greater_than_equal(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    value, filter_value, comparable = _prepare_ordering_comparison(
        value=value, filter_value=filter_value, strict_datetime_comparison=strict_datetime_comparison
    )
    if not comparable:
        return False
    return value >= filter_value


def _less_than(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    value, filter_value, comparable = _prepare_ordering_comparison(
        value=value, filter_value=filter_value, strict_datetime_comparison=strict_datetime_comparison
    )
    if not comparable:
        return False
    return value < filter_value


def _less_than_equal(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    if value is None or filter_value is None:
        # We can't compare None values reliably using operators '>', '>=', '<', '<='
        return False

    value, filter_value, comparable = _prepare_ordering_comparison(
        value=value, filter_value=filter_value, strict_datetime_comparison=strict_datetime_comparison
    )
    if not comparable:
        return False
    return value <= filter_value


def _in(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    if not isinstance(filter_value, list):
        msg = (
            f"Filter value must be a `list` when using operator 'in' or 'not in', received type '{type(filter_value)}'"
        )
        raise FilterError(msg)
    return any(_equal(e, value, strict_datetime_comparison) for e in filter_value)


def _not_in(value: Any, filter_value: Any, strict_datetime_comparison: bool) -> bool:
    return not _in(value=value, filter_value=filter_value, strict_datetime_comparison=strict_datetime_comparison)


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


def _logic_condition(
    condition: dict[str, Any], document: Document | ByteStream, strict_datetime_comparison: bool
) -> bool:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    if operator not in LOGICAL_OPERATORS:
        msg = f"Unknown logical operator '{operator}'. Valid operators are: {sorted(LOGICAL_OPERATORS)}"
        raise FilterError(msg)
    conditions: list[dict[str, Any]] = condition["conditions"]
    return LOGICAL_OPERATORS[operator](
        document=document, conditions=conditions, strict_datetime_comparison=strict_datetime_comparison
    )


def _comparison_condition(
    condition: dict[str, Any], document: Document | ByteStream, strict_datetime_comparison: bool
) -> bool:
    if "field" not in condition:
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
        return _logic_condition(
            condition=condition, document=document, strict_datetime_comparison=strict_datetime_comparison
        )
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
            if not isinstance(document_value, dict) or part not in document_value:
                # If a field is not found (or an intermediate value is not a dict,
                # e.g. None) we treat it as None
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
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {sorted(COMPARISON_OPERATORS)}"
        raise FilterError(msg)
    filter_value: Any = condition["value"]
    return COMPARISON_OPERATORS[operator](
        filter_value=filter_value, value=document_value, strict_datetime_comparison=strict_datetime_comparison
    )
