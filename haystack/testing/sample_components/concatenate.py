# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from haystack.core.component import component


@component
class Concatenate:
    """
    Concatenates two values
    """

    @component.output_types(value=list[str])
    def run(self, first: Union[list[str], str], second: Union[list[str], str]):
        """
        Concatenates two values
        """
        if isinstance(first, str) and isinstance(second, str):
            res = [first, second]
        elif isinstance(first, list) and isinstance(second, list):
            res = first + second
        elif isinstance(first, list) and isinstance(second, str):
            res = first + [second]
        elif isinstance(first, str) and isinstance(second, list):
            res = [first] + second
        else:
            res = None
        return {"value": res}
