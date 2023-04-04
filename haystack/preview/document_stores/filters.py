from typing import List, Any, Optional

from haystack.preview.dataclasses import Document


class DocumentStoreFilters:
    """
    Class that is able to parse a filter and convert it to the format that the underlying document store.

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

    Note: this class provides some default implementation for all operators and logical statements.
    Override the ones required by your document store in order to apply the filters efficiently.
    """

    def __init__(self):
        self.logical_statements = {"$not": self.not_operation, "$and": self.and_operation, "$or": self.or_operation}
        self.operators = {
            "$eq": self.eq_operation,
            "$in": self.in_operation,
            "$ne": self.ne_operation,
            "$nin": self.nin_operation,
            "$gt": self.gt_operation,
            "$gte": self.gte_operation,
            "$lt": self.lt_operation,
            "$lte": self.lte_operation,
        }

    def match(self, conditions: Any, document: Document, _current_key: Optional[str] = None):
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
            return self.and_operation(conditions=conditions, document=document, _current_key=_current_key)

        if isinstance(conditions, dict):
            # The default operation for a list of sibling conditions is AND
            if len(conditions.keys()) > 1:
                conditions = [{key: value} for key, value in conditions.items()]
                return self.and_operation(conditions=conditions, document=document, _current_key=_current_key)

            field_key, field_value = list(conditions.items())[0]

            if field_key in self.logical_statements.keys():
                # It's a nested logical statement (AND, OR, NOT)
                # Make sure all nested conditions are in a list
                if isinstance(field_value, dict):
                    conditions = [{key: value} for key, value in field_value.items()]
                elif isinstance(field_value, list):
                    conditions = field_value
                else:
                    conditions = [field_value]
                # Resolve the nested operator
                return self.logical_statements[field_key](
                    conditions=conditions, document=document, _current_key=_current_key
                )

            if field_key in self.operators.keys():
                # It's a comparison operator (EQ, IN, GTE, ...)
                # Make sure the field to apply this operation on was specified.
                if not _current_key:
                    raise ValueError(
                        "Filters can't start with an operator like $eq and $in, you have to specify which field to use first."
                    )
                # Evaluate the operator
                return self.operators[field_key](fields=document.metadata, field_name=_current_key, value=field_value)

            # if isinstance(field_value, dict):
            # # Dictionaries are resolved recursively
            # return self.resolve(conditions=field_value, document=document, _current_key=field_key)
            if isinstance(field_value, list):
                # The default operator for a {key: [value]} filter is $in
                return self.in_operation(fields=document.metadata, field_name=field_key, value=field_value)
            else:
                # The default operator for a {key: value} filter is $eq
                return self.eq_operation(fields=document.metadata, field_name=field_key, value=field_value)

        if not _current_key:
            # If we reached here it means that conditions was neither a dict nor a list.
            raise ValueError("Filters must be dictionaries or lists.")
        return self.eq_operation(fields=document.metadata, field_name=_current_key, value=field_value)

    def not_operation(self, conditions: List[Any], document: Document, _current_key: str):
        """
        Applies a NOT to all the nested conditions.

        :param conditions: the filters dictionary.
        :param document: the document to test.
        :param _current_key: internal, don't use.
        :return: True if the document matches the negated filters, False otherwise
        """
        return not self.and_operation(conditions=conditions, document=document, _current_key=_current_key)

    def and_operation(self, conditions: List[Any], document: Document, _current_key: str):
        """
        Applies an AND to all the nested conditions.

        :param conditions: the filters dictionary.
        :param document: the document to test.
        :param _current_key: internal, don't use.
        :return: True if the document matches all the filters, False otherwise
        """
        for condition in conditions:
            if not self.match(conditions=condition, document=document, _current_key=_current_key):
                return False
        return True

    def or_operation(self, conditions: List[Any], document: Document, _current_key: str):
        """
        Applies an OR to all the nested conditions.

        :param conditions: the filters dictionary.
        :param document: the document to test.
        :param _current_key: internal, don't use.
        :return: True if the document matches ano of the filters, False otherwise
        """
        for condition in conditions:
            if self.match(conditions=condition, document=document, _current_key=_current_key):
                return True
        return False

    def eq_operation(self, fields, field_name, value):
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

    def in_operation(self, fields, field_name, value):
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

    def ne_operation(self, fields, field_name, value):
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

    def nin_operation(self, fields, field_name, value):
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

    def gt_operation(self, fields, field_name, value):
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

    def gte_operation(self, fields, field_name, value):
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

    def lt_operation(self, fields, field_name, value):
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

    def lte_operation(self, fields, field_name, value):
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
