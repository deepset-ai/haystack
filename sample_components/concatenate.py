# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, List

from canals import component


@component
class Concatenate:
    """
    Concatenates two values
    """

    @component.input
    def input(self):
        class Input:
            first: Union[List[str], str]
            second: Union[List[str], str]

        return Input

    @component.output
    def output(self):
        class Output:
            value: List[str]

        return Output

    def run(self, data):
        if type(data.first) is str and type(data.second) is str:
            res = [data.first, data.second]
        elif type(data.first) is list and type(data.second) is list:
            res = data.first + data.second
        elif type(data.first) is list and type(data.second) is str:
            res = data.first + [data.second]
        elif type(data.first) is str and type(data.second) is list:
            res = [data.first] + data.second

        return self.output(res)
