from enum import Enum
from typing import Optional


class MetaRankerEnum(Enum):
    """
    Super-class for enum parameters of `MetaFieldRanker`.
    """

    def __str__(self):
        return self.value

    @staticmethod
    def get_param_name() -> Optional[str]:
        """
        Returns the name of the MetaFieldRanker parameter that the enum corresponds to.
        """
        return None

    @staticmethod
    def is_optional() -> bool:
        """
        Returns true if the MetaFieldRanker parameter that the enum corresponds to is optional.
        """
        return False

    @classmethod
    def from_str(cls, string: str) -> "MetaRankerEnum":
        """
        Convert a string to a `MetaRankerEnum` enum.

        :param string: The string to convert.
        :param param_name: The name of the parameter that the string is being converted for.
        :return: The corresponding `MetaRankerEnum` enum.
        """
        enum_map = {e.value: e for e in cls}
        mode = enum_map.get(string)
        if mode is None:
            supported_values = list(enum_map.keys())
            if cls.is_optional():
                supported_values.append(None)
            msg = f"Unknown <{cls.get_param_name()}> value. Supported values are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode

class MetaRankerMissingMeta(MetaRankerEnum):
    """
    Determines what a given `MetaFieldRanker` should do with documents that are missing the sorting metadata field.

    Possible values are:
    - 'drop' will drop the documents entirely.
    - 'top' will place the documents at the top of the metadata-sorted list
        (regardless of 'ascending' or 'descending').
    - 'bottom' will place the documents at the bottom of metadata-sorted list
        (regardless of 'ascending' or 'descending').
    """

    # To place the documents at the top of the metadata-sorted list
    # (regardless of 'ascending' or 'descending'):
    TOP = "top"

    # To place the documents at the bottom of metadata-sorted list
    # (regardless of 'ascending' or 'descending'):
    BOTTOM = "bottom"

    # To drop the documents entirely:
    DROP = "drop"

    @staticmethod
    def get_param_name() -> str:
        """
        Returns the name of the MetaFieldRanker parameter that the enum corresponds to.
        """
        return "missing_meta"

class MetaRankerRankingMode(MetaRankerEnum):
    """
    The mode used to combine the Retriever's and Ranker's scores.

    Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
    Use the 'linear_score' mode only with Retrievers or Rankers that return a score in range [0,1].
    """

    # To use reciprocal rank fusion for combining the original and sorted orders:
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"

    # To use linear score for combining the original and sorted orders:
    LINEAR_SCORE = "linear_score"

    @staticmethod
    def get_param_name() -> str:
        """
        Returns the name of the MetaFieldRanker parameter that the enum corresponds to.
        """
        return "ranking_mode"

class MetaRankerSortOrder(MetaRankerEnum):
    """
    The order in which the metadata field should be sorted.

    Possible values are 'ascending' and 'descending'.
    """

    # To sort the metadata field in ascending order:
    ASCENDING = "ascending"

    # To sort the metadata field in descending order:
    DESCENDING = "descending"

    @staticmethod
    def get_param_name() -> str:
        """
        Returns the name of the MetaFieldRanker parameter that the enum corresponds to.
        """
        return "sort_order"

class MetaRankerMetaValueType(MetaRankerEnum):
    """
    Parse the meta value into the data type specified before sorting.

    This will only work if all meta values stored under `meta_field` in the provided documents are strings.
    For example, if we specified the type as date then for the meta value `"date": "2015-02-01"`
    we would parse the string into a datetime object and then sort the documents by date.
    The available options are:
    - 'float' will parse the meta values into floats.
    - 'int' will parse the meta values into integers.
    - 'date' will parse the meta values into datetime objects.
    - None (default) will do no parsing.
    """

    # To parse the metadata field as a float:
    FLOAT = "float"

    # To parse the metadata field as an integer:
    INT = "int"

    # To parse the metadata field as a date:
    DATE = "date"

    @staticmethod
    def get_param_name() -> str:
        """
        Returns the name of the MetaFieldRanker parameter that the enum corresponds to.
        """
        return "meta_value_type"

    @staticmethod
    def is_optional() -> bool:
        """
        Returns true if the MetaFieldRanker parameter that the enum corresponds to is optional.
        """
        return True
