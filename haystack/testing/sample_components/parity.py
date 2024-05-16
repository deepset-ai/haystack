# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.core.component import component


@component
class Parity:  # pylint: disable=too-few-public-methods
    """
    Redirects the value, unchanged, along the 'even' connection if even, or along the 'odd' one if odd.
    """

    @component.output_types(even=int, odd=int)
    def run(self, value: int):
        """
        :param value: The value to check for parity
        """
        remainder = value % 2
        if remainder:
            return {"odd": value}
        return {"even": value}
