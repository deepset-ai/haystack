# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import builtins
from typing import List, Union, Optional
from dataclasses import make_dataclass, is_dataclass, asdict

from canals import component


class MergeLoop:
    ...


# @component
# class MergeLoop:

#     """
#     Takes multiple inputs and returns the first one that is not None.
#     """

#     def __new__(cls, *args, **kwargs):
#         """
#         Factory method for MergeLoop
#         """

#         @component.return_types(value=str)
#         def run(self, values: List[Optional[str]]):
#             """
#             Takes some inputs and returns the first one that is not None.
#             """
#             for v in values:
#                 if v is not None:
#                     return {"value": v}
#             return {"value": None}

#         cls.run = run
#         return super().__new__(cls)

#     def __init__(self, expected_type: Union[type, str], inputs: List[str] = ["value_1", "value_2"]):
#         self.expected_type = expected_type


# def merge_loop_component(expected_type: Union[type, str], inputs: List[str] = ["value_1", "value_2"]):
#     """
#     Factory method for MergeLoop
#     """

#     @component
#     class MergeLoop:

#         """
#         Takes multiple inputs and returns the first one that is not None.
#         """

#         def __init__(self):
#             self.expected_type = expected_type

#         @component.run_types(**{input: expected_type for input in inputs})
#         @component.return_types(value=expected_type)
#         def run(self, values: List[Optional[str]]):
#             """
#             Takes some inputs and returns the first one that is not None.
#             """
#             for v in values:
#                 if v is not None:
#                     return {"value": v}
#             return {"value": None}

#     return MergeLoop
