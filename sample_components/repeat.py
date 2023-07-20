# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from dataclasses import make_dataclass

from canals import component


class Repeat:
    ...


# @component
# class Repeat:
#     """
#     Repeats the input value on all outputs.
#     """

#     @component.input
#     def input(self):
#         class Input:
#             value: int

#         return Input

#     def __init__(self, outputs: List[str] = ["output_1", "output_2", "output_3"]):
#         self.outputs = outputs
#         self._output_type = make_dataclass("Output", fields=[(val, int, None) for val in outputs])

#     @component.output
#     def output(self):
#         return self._output_type

#     def run(self, data):
#         output_dataclass = self.output()
#         for output in self.outputs:
#             setattr(output_dataclass, output, data.value)
#         return output_dataclass
