from enum import Enum


class MetaRankerMissingMeta(Enum):
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

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "MetaRankerMissingMeta":
        """
        Convert a string to a `MetaRankerEnum` enum.

        :param string: The string to convert.
        :return: The corresponding `MetaRankerEnum` enum.
        """
        enum_map = {e.value: e for e in MetaRankerMissingMeta}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown <missing_meta> value. Supported values are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode
