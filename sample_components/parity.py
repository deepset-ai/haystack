# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Any

from canals.serialization import default_to_dict, default_from_dict
from canals import component


@component
class Parity:  # pylint: disable=too-few-public-methods
    """
    Redirects the value, unchanged, along the 'even' connection if even, or along the 'odd' one if odd.
    """

    def to_dict(self) -> Dict[str, Any]:  # pylint: disable=missing-function-docstring
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Parity":  # pylint: disable=missing-function-docstring
        return default_from_dict(cls, data)

    @component.output_types(even=int, odd=int)
    def run(self, value: int):
        """
        :param value: The value to check for parity
        """
        remainder = value % 2
        if remainder:
            return {"even": None, "odd": value}
        return {"even": value, "odd": None}
