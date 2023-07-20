# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import types
from typing import List, Any

from canals import component


class Sum:
    @staticmethod
    def create(inputs: List[str]):
        """
        Sums the values of all the input connections together.
        """

        @component
        class SumImpl:
            """
            Implementation of Sum()
            """

            @component.return_types(total=int)
            @component.run_method_types(**{input_name: int for input_name in inputs})
            def run(self, **kwargs: Any):
                """
                :param value: the value to check the remainder of.
                """
                return {"total": sum(kwargs.values())}

        return SumImpl()

    def __init__(self):
        raise NotImplementedError("use Sum.create()")
