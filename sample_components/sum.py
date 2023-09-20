# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List

from canals import component


@component
class Sum:
    def __init__(self, inputs: List[str]):
        self.inputs = inputs
        component.set_input_types(self, **{input_name: Optional[int] for input_name in inputs})

    @component.output_types(total=int)
    def run(self, **kwargs):
        """
        :param value: the value to check the remainder of.
        """
        return {"total": sum(kwargs.values())}
