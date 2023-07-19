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

    @component.return_types(value=List[str])
    def run(self, first: Union[List[str], str], second: Union[List[str], str]):
        if type(first) is str and type(second) is str:
            res = [first, second]
        elif type(first) is list and type(second) is list:
            res = first + second
        elif type(first) is list and type(second) is str:
            res = first + [second]
        elif type(first) is str and type(second) is list:
            res = [first] + second
        return {"value": res}
