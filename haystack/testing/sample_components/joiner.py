# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from haystack.core.component import component
from haystack.core.component.types import Variadic


@component
class StringJoiner:
    @component.output_types(output=str)
    def run(self, input_str: Variadic[str]):
        """
        Take strings from multiple input nodes and join them into a single one returned in output.

        Since `input_str` is Variadic, we know we'll receive a list[str].
        """
        return {"output": " ".join(input_str)}


@component
class StringListJoiner:
    @component.output_types(output=str)
    def run(self, inputs: Variadic[list[str]]):
        """
        Take list of strings from multiple input nodes and join them into a single one returned in output.

        Since `input_str` is Variadic, we know we'll receive a list[list[str]].
        """
        retval: list[str] = []
        for list_of_strings in inputs:
            retval += list_of_strings

        return {"output": retval}
