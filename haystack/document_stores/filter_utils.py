from typing import Union, List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

from sqlalchemy.sql import select
from sqlalchemy import and_, or_

from haystack.document_stores.utils import convert_date_to_rfc3339
from haystack.errors import FilterError


def nested_defaultdict() -> defaultdict:
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

    @abstractmethod
    def convert_to_elasticsearch(self):
        """
        Converts the LogicalFilterClause instance to an Elasticsearch filter.
        """
        pass

    @abstractmethod
    def convert_to_sql(self, meta_document_orm):
        """
        Converts the LogicalFilterClause instance to an SQL filter.
        """
        pass

    def convert_to_weaviate(self):
        """
        Converts the LogicalFilterClause instance to a Weaviate filter.
        """
        pass

    def convert_to_pinecone(self):
        """
        Converts the LogicalFilterClause instance to a Pinecone filter.
        """
        pass

    def _merge_es_range_queries(self, conditions: List[Dict]) -> List[Dict[str, Dict]]:
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

    @abstractmethod
    def invert(self) -> Union["LogicalFilterClause", "ComparisonOperation"]:
        """
        Inverts the LogicalOperation instance.
        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.
        (https://github.com/semi-technologies/weaviate/issues/1717)
        """
        pass


class ComparisonOperation(ABC):
    def __init__(self, field_name: str, comparison_value: Union[str, int, float, bool, List]):
        self.field_name = field_name
        self.comparison_value = comparison_value

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

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

    @abstractmethod
    def convert_to_elasticsearch(self):
        """
        Converts the ComparisonOperation instance to an Elasticsearch query.
        """
        pass

    @abstractmethod
    def convert_to_sql(self, meta_document_orm):
        """
        Converts the ComparisonOperation instance to an SQL filter.
        """
        pass

    @abstractmethod
    def convert_to_weaviate(self):
        """
        Converts the ComparisonOperation instance to a Weaviate comparison operator.
        """
        pass

    def convert_to_pinecone(self):
        """
        Converts the ComparisonOperation instance to a Pinecone comparison operator.
        """
        pass

    @abstractmethod
    def invert(self) -> "ComparisonOperation":
        """
        Inverts the ComparisonOperation.
        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.
        (https://github.com/semi-technologies/weaviate/issues/1717)
        """
        pass

    def _get_weaviate_datatype(
        self, value: Optional[Union[str, int, float, bool]] = None
    ) -> Tuple[str, Union[str, int, float, bool]]:
        """
        Determines the type of the comparison value and converts it to RFC3339 format if it is as date,
        as Weaviate requires dates to be in RFC3339 format including the time and timezone.

        """
        if value is None:
            assert not isinstance(self.comparison_value, list)  # Necessary for mypy
            value = self.comparison_value

        if isinstance(value, str):
            # Check if comparison value is a date
            try:
                value = convert_date_to_rfc3339(value)
                data_type = "valueDate"
            # Comparison value is a plain string
            except ValueError:
                if self.field_name == "content":
                    data_type = "valueText"
                else:
                    data_type = "valueString"
        elif isinstance(value, int):
            data_type = "valueInt"
        elif isinstance(value, float):
            data_type = "valueNumber"
        elif isinstance(value, bool):
            data_type = "valueBoolean"
        else:
            raise ValueError(
                f"Unsupported data type of comparison value for {self.__class__.__name__}."
                f"Value needs to be of type str, int, float, or bool."
            )

        return data_type, value


class NotOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'NOT' operations.
    """

    def evaluate(self, fields) -> bool:
        return not any(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {"bool": {"must_not": conditions}}

    def convert_to_sql(self, meta_document_orm):
        conditions = [
            meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm))
            for condition in self.conditions
        ]
        return select(meta_document_orm.document_id).filter(~or_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        conditions = [condition.invert().convert_to_weaviate() for condition in self.conditions]
        if len(conditions) > 1:
            # Conditions in self.conditions are by default combined with AND which becomes OR according to DeMorgan
            return {"operator": "Or", "operands": conditions}
        else:
            return conditions[0]

    def convert_to_pinecone(self) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        conditions = [condition.invert().convert_to_pinecone() for condition in self.conditions]
        if len(conditions) > 1:
            # Conditions in self.conditions are by default combined with AND which becomes OR according to DeMorgan
            return {"$or": conditions}
        else:
            return conditions[0]

    def invert(self) -> Union[LogicalFilterClause, ComparisonOperation]:
        # This method is called when a "$not" operation is embedded in another "$not" operation. Therefore, we don't
        # invert the operations here, as two "$not" operation annihilate each other.
        # (If we have more than one condition, we return an AndOperation, the default logical operation for combining
        # multiple conditions.)
        if len(self.conditions) > 1:
            return AndOperation(self.conditions)
        else:
            return self.conditions[0]


class AndOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'AND' operations.
    """

    def evaluate(self, fields) -> bool:
        return all(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {"bool": {"must": conditions}}

    def convert_to_sql(self, meta_document_orm):
        conditions = [
            meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm))
            for condition in self.conditions
        ]
        return select(meta_document_orm.document_id).filter(and_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_weaviate() for condition in self.conditions]
        return {"operator": "And", "operands": conditions}

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {"$and": conditions}

    def invert(self) -> "OrOperation":
        return OrOperation([condition.invert() for condition in self.conditions])


class OrOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'OR' operations.
    """

    def evaluate(self, fields) -> bool:
        return any(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {"bool": {"should": conditions}}

    def convert_to_sql(self, meta_document_orm):
        conditions = [
            meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm))
            for condition in self.conditions
        ]
        return select(meta_document_orm.document_id).filter(or_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_weaviate() for condition in self.conditions]
        return {"operator": "Or", "operands": conditions}

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {"$or": conditions}

    def invert(self) -> AndOperation:
        return AndOperation([condition.invert() for condition in self.conditions])


class EqOperation(ComparisonOperation):
    """
    Handles conversion of the '$eq' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] == self.comparison_value

    def convert_to_elasticsearch(
        self,
    ) -> Dict[str, Dict[str, Union[str, int, float, bool, Dict[str, Union[list, Dict[str, str]]]]]]:
        if isinstance(self.comparison_value, list):
            return {
                "terms_set": {
                    self.field_name: {
                        "terms": self.comparison_value,
                        "minimum_should_match_script": {
                            "source": f"Math.max(params.num_terms, doc['{self.field_name}'].size())"
                        },
                    }
                }
            }
        return {"term": {self.field_name: self.comparison_value}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value == self.comparison_value
        )

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, int, float, bool]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        return {"path": [self.field_name], "operator": "Equal", comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        return {self.field_name: {"$eq": self.comparison_value}}

    def invert(self) -> "NeOperation":
        return NeOperation(self.field_name, self.comparison_value)


class InOperation(ComparisonOperation):
    """
    Handles conversion of the '$in' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] in self.comparison_value  # type: ignore
        # is only initialized with lists, but changing the type annotation would mean duplicating __init__

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, List]]:
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        return {"terms": {self.field_name: self.comparison_value}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value.in_(self.comparison_value)
        )

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        filter_dict: Dict[str, Union[str, List[Dict]]] = {"operator": "Or", "operands": []}
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        for value in self.comparison_value:
            comp_value_type, comp_value = self._get_weaviate_datatype(value)
            assert isinstance(filter_dict["operands"], list)  # Necessary for mypy
            filter_dict["operands"].append(
                {"path": [self.field_name], "operator": "Equal", comp_value_type: comp_value}
            )

        return filter_dict

    def convert_to_pinecone(self) -> Dict[str, Dict[str, List]]:
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        return {self.field_name: {"$in": self.comparison_value}}

    def invert(self) -> "NinOperation":
        return NinOperation(self.field_name, self.comparison_value)


class NeOperation(ComparisonOperation):
    """
    Handles conversion of the '$ne' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] != self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Union[str, int, float, bool]]]]]:
        if isinstance(self.comparison_value, list):
            raise FilterError("Use '$nin' operation for lists as comparison values.")
        return {"bool": {"must_not": {"term": {self.field_name: self.comparison_value}}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value != self.comparison_value
        )

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, int, float, bool]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        return {"path": [self.field_name], "operator": "NotEqual", comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        return {self.field_name: {"$ne": self.comparison_value}}

    def invert(self) -> "EqOperation":
        return EqOperation(self.field_name, self.comparison_value)


class NinOperation(ComparisonOperation):
    """
    Handles conversion of the '$nin' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] not in self.comparison_value  # type: ignore
        # is only initialized with lists, but changing the type annotation would mean duplicating __init__

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Dict[str, List]]]]:
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$nin' operation requires comparison value to be a list.")
        return {"bool": {"must_not": {"terms": {self.field_name: self.comparison_value}}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value.notin_(self.comparison_value)
        )

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        filter_dict: Dict[str, Union[str, List[Dict]]] = {"operator": "And", "operands": []}
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$nin' operation requires comparison value to be a list.")
        for value in self.comparison_value:
            comp_value_type, comp_value = self._get_weaviate_datatype(value)
            assert isinstance(filter_dict["operands"], list)  # Necessary for mypy
            filter_dict["operands"].append(
                {"path": [self.field_name], "operator": "NotEqual", comp_value_type: comp_value}
            )

        return filter_dict

    def convert_to_pinecone(self) -> Dict[str, Dict[str, List]]:
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        return {self.field_name: {"$nin": self.comparison_value}}

    def invert(self) -> "InOperation":
        return InOperation(self.field_name, self.comparison_value)


class GtOperation(ComparisonOperation):
    """
    Handles conversion of the '$gt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] > self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$gt' operation must not be a list.")
        return {"range": {self.field_name: {"gt": self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value > self.comparison_value
        )

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$gt' operation must not be a list.")
        return {"path": [self.field_name], "operator": "GreaterThan", comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$gt' operation must be a float or int.")
        return {self.field_name: {"$gt": self.comparison_value}}

    def invert(self) -> "LteOperation":
        return LteOperation(self.field_name, self.comparison_value)


class GteOperation(ComparisonOperation):
    """
    Handles conversion of the '$gte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] >= self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$gte' operation must not be a list.")
        return {"range": {self.field_name: {"gte": self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value >= self.comparison_value
        )

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$gte' operation must not be a list.")
        return {"path": [self.field_name], "operator": "GreaterThanEqual", comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$gte' operation must be a float or int.")
        return {self.field_name: {"$gte": self.comparison_value}}

    def invert(self) -> "LtOperation":
        return LtOperation(self.field_name, self.comparison_value)


class LtOperation(ComparisonOperation):
    """
    Handles conversion of the '$lt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] < self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$lt' operation must not be a list.")
        return {"range": {self.field_name: {"lt": self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value < self.comparison_value
        )

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$lt' operation must not be a list.")
        return {"path": [self.field_name], "operator": "LessThan", comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$lt' operation must be a float or int.")
        return {self.field_name: {"$lt": self.comparison_value}}

    def invert(self) -> "GteOperation":
        return GteOperation(self.field_name, self.comparison_value)


class LteOperation(ComparisonOperation):
    """
    Handles conversion of the '$lte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] <= self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$lte' operation must not be a list.")
        return {"range": {self.field_name: {"lte": self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(
            meta_document_orm.name == self.field_name, meta_document_orm.value <= self.comparison_value
        )

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$lte' operation must not be a list.")
        return {"path": [self.field_name], "operator": "LessThanEqual", comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$lte' operation must be a float or int.")
        return {self.field_name: {"$lte": self.comparison_value}}

    def invert(self) -> "GtOperation":
        return GtOperation(self.field_name, self.comparison_value)
