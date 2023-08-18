# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any

from canals import component
from canals.serialization import default_to_dict, default_from_dict


@component
class AddFixedValue:
    """
    Adds two values together.
    """

    def __init__(self, add: int = 1):
        self.add = add

    def to_dict(self) -> Dict[str, Any]:  # pylint: disable=missing-function-docstring
        return default_to_dict(self, add=self.add)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AddFixedValue":  # pylint: disable=missing-function-docstring
        return default_from_dict(cls, data)

    @component.output_types(result=int)
    def run(self, value: int, add: Optional[int] = None):
        """
        Adds two values together.
        """
        if add is None:
            add = self.add
        return {"result": value + add}
