# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from canals import component


@component
class Sum:  # pylint: disable=too-few-public-methods
    def __init__(self, inputs):
        component.set_input_types(self, **{input_name: int for input_name in inputs})

    @component.return_types(total=int)
    def run(self, **kwargs: Any):
        """
        :param value: the value to check the remainder of.
        """
        return {"total": sum(kwargs.values())}
