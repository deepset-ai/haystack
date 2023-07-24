# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Any, Optional

from canals import component


@component
class MergeLoop:  # pylint: disable=too-few-public-methods
    def __init__(self, expected_type: Any, inputs: List[str]):
        component.set_input_types(self, **{input_name: Optional[expected_type] for input_name in inputs})
        component.set_output_types(self, value=expected_type)
        self.init_parameters = {"expected_type": str(expected_type), "inputs": inputs}

    def run(self, **kwargs):
        """
        :param kwargs: find the first non-None value and return it.
        """
        for value in kwargs.values():
            if value is not None:
                return {"value": value}
        return {"value": None}
