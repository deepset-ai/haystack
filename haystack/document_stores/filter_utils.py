from typing import Union, List, Dict
from abc import ABC, abstractmethod
from collections import defaultdict


def nested_defaultdict():
    """
    Data structure that recursively adds a dictionary as value if a key does not exist. Advantage: In nested dictionary
    structures, we don't need to check if a key already exists (which can become hard to maintain in nested dictionaries
    with many levels) but access the existing value if a key exists and create an empty dictionary if a key does not
    exist.
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

    def __init__(self, conditions: List["LogicalFilterClause"]):
        self.conditions = conditions

    @classmethod
    def parse(cls, filter_term: Union[dict, List[dict]]):
        """
        Parses a filter dictionary/list and returns a LogicalFilterClause instance.

        :param filter_term: Dictionary or list that contains the filter definition.
        """
        conditions = []

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

    @abstractmethod
    def convert_to_elasticsearch(self):
        """
        Converts the LogicalFilterClause instance to an Elasticsearch filter.
        """
        pass

    def _merge_es_range_queries(self, conditions: List[Dict]) -> List[Dict]:
        """
        Merges Elasticsearch range queries that perform on the same metadata field.
        """

        range_conditions = [cond["range"] for cond in filter(lambda condition: "range" in condition, conditions)]
        if range_conditions:
            conditions = [condition for condition in conditions if "range" not in condition]
            range_conditions_dict = nested_defaultdict()
            for condition in range_conditions:
                field_name = list(condition.keys())[0]
                operation = list(condition[field_name].keys())[0]
                comparison_value = condition[field_name][operation]
                range_conditions_dict[field_name][operation] = comparison_value

            for field_name, comparison_operations in range_conditions_dict.items():
                conditions.append({"range": {field_name: comparison_operations}})

        return conditions


class ComparisonOperation(ABC):
    def __init__(self, field_name: str, comparison_value: Union[str, float, List]):
        self.field_name = field_name
        self.comparison_value = comparison_value

    @classmethod
    def parse(cls, field_name, comparison_clause: Union[Dict, List, str, float]):
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

    @abstractmethod
    def convert_to_elasticsearch(self):
        """
        Converts the ComparisonOperation instance to an Elasticsearch query.
        """
        pass


class NotOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'NOT' operations.
    """

    def convert_to_elasticsearch(self):
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {"bool": {"must_not": conditions}}


class AndOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'AND' operations.
    """

    def convert_to_elasticsearch(self):
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {"bool": {"must": conditions}}


class OrOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'OR' operations.
    """

    def convert_to_elasticsearch(self):
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {"bool": {"should": conditions}}


class EqOperation(ComparisonOperation):
    """
    Handles conversion of the '$eq' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"term": {self.field_name: self.comparison_value}}


class InOperation(ComparisonOperation):
    """
    Handles conversion of the '$in' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"terms": {self.field_name: self.comparison_value}}


class NeOperation(ComparisonOperation):
    """
    Handles conversion of the '$ne' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"bool": {"must_not": {"term": {self.field_name: self.comparison_value}}}}


class NinOperation(ComparisonOperation):
    """
    Handles conversion of the '$nin' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"bool": {"must_not": {"terms": {self.field_name: self.comparison_value}}}}


class GtOperation(ComparisonOperation):
    """
    Handles conversion of the '$gt' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"range": {self.field_name: {"gt": self.comparison_value}}}


class GteOperation(ComparisonOperation):
    """
    Handles conversion of the '$gte' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"range": {self.field_name: {"gte": self.comparison_value}}}


class LtOperation(ComparisonOperation):
    """
    Handles conversion of the '$lt' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"range": {self.field_name: {"lt": self.comparison_value}}}


class LteOperation(ComparisonOperation):
    """
    Handles conversion of the '$lte' comparison operation.
    """

    def convert_to_elasticsearch(self):
        return {"range": {self.field_name: {"lte": self.comparison_value}}}
