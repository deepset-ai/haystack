from typing import Union, Dict, List, Any


def convert(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a filter declared using the legacy style into the new style.
    This is mostly meant to ease migration from Haystack 1.x to 2.x for developers
    of Document Stores and Components that uses filters.

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
    if isinstance(filters, list):
        if previous_key in LOGIC_OPERATORS:
            return [_internal_convert(f) for f in filters]
        elif previous_key not in COMPARISON_OPERATORS:
            return {"field": previous_key, "operator": "in", "value": filters}

    if not isinstance(filters, dict):
        if previous_key not in ALL_OPERATORS:
            return {"field": previous_key, "operator": "==", "value": filters}
        else:
            return filters

    for key, value in filters.items():
        if key not in ALL_OPERATORS:
            converted = _internal_convert(value, previous_key=key)
            if isinstance(converted, list):
                conditions.extend(converted)
            else:
                conditions.append(converted)
        elif key in LOGIC_OPERATORS:
            if previous_key not in ALL_OPERATORS and isinstance(value, list):
                converted = [_internal_convert({previous_key: v}) for v in value]
                conditions.append({"operator": ALL_OPERATORS[key], "conditions": converted})
            else:
                converted = _internal_convert(value, previous_key=key)
                if not isinstance(converted, list):
                    converted = [converted]
                conditions.append({"operator": ALL_OPERATORS[key], "conditions": converted})
        elif key in COMPARISON_OPERATORS:
            conditions.append({"field": previous_key, "operator": ALL_OPERATORS[key], "value": value})

    if len(conditions) == 1:
        return conditions[0]

    if previous_key is None:
        return {"operator": "AND", "conditions": conditions}

    return conditions


# Operator mappings from legacy style to new one
LOGIC_OPERATORS = {"$and": "AND", "$or": "OR", "$not": "NOT"}

COMPARISON_OPERATORS = {
    "$eq": "==",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$in": "in",
    "$nin": "not in",
}

ALL_OPERATORS = {**LOGIC_OPERATORS, **COMPARISON_OPERATORS}
