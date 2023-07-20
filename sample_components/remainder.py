# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import make_dataclass

from canals import component


class Remainder:
    ...


# @component
# class Remainder:
#     """
#     Redirects the value, unchanged, along the connection corresponding to the remainder
#     of a division. For example, if `divisor=3`, the value `5` would be sent along
#     the second output connection.
#     """

#     def __new__(cls, **kwargs):
#         """
#         Factory method for Remainder
#         """
#         obj = super().__new__(cls)

#         @component.return_types(**{f"remainder_is_{val}": int for val in range(obj.divisor)})
#         def run(self, value: int):
#             """
#             :param value: the value to check the remainder of.
#             """
#             remainder = value % obj.divisor
#             output = {f"remainder_is_{val}": int for val in range(obj.divisor)}
#             output[f"remainder_is_{remainder}"] = value
#             return output

#         obj.run = run
#         return obj

#     def __init__(self, divisor: int = 2):
#         if divisor == 0:
#             raise ValueError("Can't divide by zero")
#         self.divisor = divisor
