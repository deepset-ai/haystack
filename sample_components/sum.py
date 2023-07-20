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


# @component
# class Sum:
# """
# Sums the values of all the input connections together.
# """

#     def __init__(self, inputs=["value_1"]) -> None:
#         # mypy complains that we can't Optional is not a type, so we ignore the error
#         # cause we consider this to be correct
#         self._input = make_dataclass("Input", fields=[(f, Optional[int]) for f in inputs])  # type: ignore

#     @component.input
#     def input(self):
#         return self._input

#     @component.output
#     def output(self):
#         class Output:
#             total: int

#         return Output

#     def run(self, data):
#         values = []
#         if is_dataclass(data):
#             values = [n for n in asdict(data).values() if n]
#         return self.output(total=sum(values))
