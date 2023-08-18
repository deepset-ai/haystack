# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Union, List, Any

from canals import component
from canals.serialization import default_to_dict, default_from_dict


@component
class Concatenate:
    """
    Concatenates two values
    """

    def to_dict(self) -> Dict[str, Any]:  # pylint: disable=missing-function-docstring
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Concatenate":  # pylint: disable=missing-function-docstring
        return default_from_dict(cls, data)

    @component.output_types(value=List[str])
    def run(self, first: Union[List[str], str], second: Union[List[str], str]):
        """
        Concatenates two values
        """
        if isinstance(first, str) and isinstance(second, str):
            res = [first, second]
        elif isinstance(first, list) and isinstance(second, list):
            res = first + second
        elif isinstance(first, list) and isinstance(second, str):
            res = first + [second]
        elif isinstance(first, str) and isinstance(second, list):
            res = [first] + second
        return {"value": res}
