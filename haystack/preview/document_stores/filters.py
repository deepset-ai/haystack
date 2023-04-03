from typing import Union, List, Dict
from abc import ABC, abstractmethod
from collections import defaultdict


def nested_defaultdict() -> defaultdict:
    """
    Data structure that recursively adds a dictionary as value if a key does not exist. Advantage: In nested dictionary
    structures, we don't need to check if a key already exists (which can become hard to maintain in nested
    dictionaries with many levels) but access the existing value if a key exists and create an empty dictionary if a
    key does not exist.
    """
    return defaultdict(nested_defaultdict)


class LogicalFilterClause(ABC):
    """
    Class that is able to parse a filter and convert it to the format that the underlying databases of our
    DocumentStores require.

    Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
    operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`, `"$gte"`, `"$lt"`,
    `"$lte"`) or a metadata field name.
    Logical operator keys take a dictionary of metadata field names and/or logical operators as
    value. Metadata field names take a dictionary of comparison operators as value. Comparison
    operator keys take a single value or (in case of `"$in"`) a list of values as value.
    If no logical operator is provided, `"$and"` is used as default operation. If no comparison
    operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
    operation.
    Example:
        ```python
        filters = {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {
                    "genre": {"$in": ["economy", "politics"]},
                    "publisher": {"$eq": "nytimes"}
                }
            }
        }
        # or simpler using default operators
        filters = {
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": ["economy", "politics"],
                "publisher": "nytimes"
            }
        }
        ```

    To use the same logical operator multiple times on the same level, logical operators take optionally a list of
    dictionaries as value.

    Example:
        ```python
        filters = {
            "$or": [
                {
                    "$and": {
                        "Type": "News Paper",
                        "Date": {
                            "$lt": "2019-01-01"
                        }
                    }
                },
                {
                    "$and": {
                        "Type": "Blog Post",
                        "Date": {
                            "$gte": "2019-01-01"
                        }
                    }
                }
            ]
        }
        ```

    """

    def __init__(self, conditions: List[Union["LogicalFilterClause", "ComparisonOperation"]]):
        self.conditions = conditions

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

    @classmethod
    def parse(cls, filter_term: Union[dict, List[dict]]) -> Union["LogicalFilterClause", "ComparisonOperation"]:
        """
        Parses a filter dictionary/list and returns a LogicalFilterClause instance.

        :param filter_term: Dictionary or list that contains the filter definition.
        """
        conditions: List[Union[LogicalFilterClause, ComparisonOperation]] = []

        if isinstance(filter_term, dict):
            filter_term = [filter_term]
        for item in filter_term:
            for key, value in item.items():
                if key == "$not":
                    conditions.append(NotOperation.parse(value))
                elif key == "$and":
                    conditions.append(AndOperation.parse(value))
                elif key == "$or":
                    conditions.append(OrOperation.parse(value))
                # Key needs to be a metadata field
                else:
                    conditions.extend(ComparisonOperation.parse(key, value))

        if cls == LogicalFilterClause:
            if len(conditions) == 1:
                return conditions[0]
            else:
                return AndOperation(conditions)
        else:
            return cls(conditions)


class ComparisonOperation(ABC):
    def __init__(self, field_name: str, comparison_value: Union[str, int, float, bool, List]):
        self.field_name = field_name
        self.comparison_value = comparison_value

    @classmethod
    def parse(cls, field_name, comparison_clause: Union[Dict, List, str, float]) -> List["ComparisonOperation"]:
        comparison_operations: List[ComparisonOperation] = []

        if isinstance(comparison_clause, dict):
            for comparison_operation, comparison_value in comparison_clause.items():
                if comparison_operation == "$eq":
                    comparison_operations.append(EqOperation(field_name, comparison_value))
                elif comparison_operation == "$in":
                    comparison_operations.append(InOperation(field_name, comparison_value))
                elif comparison_operation == "$ne":
                    comparison_operations.append(NeOperation(field_name, comparison_value))
                elif comparison_operation == "$nin":
                    comparison_operations.append(NinOperation(field_name, comparison_value))
                elif comparison_operation == "$gt":
                    comparison_operations.append(GtOperation(field_name, comparison_value))
                elif comparison_operation == "$gte":
                    comparison_operations.append(GteOperation(field_name, comparison_value))
                elif comparison_operation == "$lt":
                    comparison_operations.append(LtOperation(field_name, comparison_value))
                elif comparison_operation == "$lte":
                    comparison_operations.append(LteOperation(field_name, comparison_value))

        # No comparison operator is given, so we use the default operators "$in" if the comparison value is a list and
        # "$eq" in every other case
        elif isinstance(comparison_clause, list):
            comparison_operations.append(InOperation(field_name, comparison_clause))
        else:
            comparison_operations.append((EqOperation(field_name, comparison_clause)))

        return comparison_operations


class NotOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'NOT' operations.
    """

    def evaluate(self, fields) -> bool:
        return not any(condition.evaluate(fields) for condition in self.conditions)


class AndOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'AND' operations.
    """

    def evaluate(self, fields) -> bool:
        return all(condition.evaluate(fields) for condition in self.conditions)


class OrOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'OR' operations.
    """

    def evaluate(self, fields) -> bool:
        return any(condition.evaluate(fields) for condition in self.conditions)


class EqOperation(ComparisonOperation):
    """
    Handles conversion of the '$eq' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] == self.comparison_value


class InOperation(ComparisonOperation):
    """
    Handles conversion of the '$in' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] in self.comparison_value  # type: ignore
        # is only initialized with lists, but changing the type annotation would mean duplicating __init__


class NeOperation(ComparisonOperation):
    """
    Handles conversion of the '$ne' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return True
        return fields[self.field_name] != self.comparison_value


class NinOperation(ComparisonOperation):
    """
    Handles conversion of the '$nin' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return True
        return fields[self.field_name] not in self.comparison_value  # type: ignore
        # is only initialized with lists, but changing the type annotation would mean duplicating __init__


class GtOperation(ComparisonOperation):
    """
    Handles conversion of the '$gt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] > self.comparison_value


class GteOperation(ComparisonOperation):
    """
    Handles conversion of the '$gte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] >= self.comparison_value


class LtOperation(ComparisonOperation):
    """
    Handles conversion of the '$lt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] < self.comparison_value


class LteOperation(ComparisonOperation):
    """
    Handles conversion of the '$lte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] <= self.comparison_value
