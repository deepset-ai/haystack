# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Dict, Any

from canals import component
from canals.serialization import default_to_dict, default_from_dict


@component
class Repeat:
    def __init__(self, outputs: List[str]):
        self.outputs = outputs
        component.set_output_types(self, **{k: int for k in outputs})

    def to_dict(self) -> Dict[str, Any]:  # pylint: disable=missing-function-docstring
        return default_to_dict(self, outputs=self.outputs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Repeat":  # pylint: disable=missing-function-docstring
        return default_from_dict(cls, data)

    def run(self, value: int):
        """
        :param value: the value to repeat.
        """
        return {val: value for val in self.outputs}
