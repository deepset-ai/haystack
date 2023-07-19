# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from canals import component


@component
class Threshold:
    """
    Redirects the value, unchanged, along a different connection whether the value is above
    or below the given threshold.

    Single input, double output decision component.

    :param threshold: the number to compare the input value against. This is also a parameter.
    """

    def __init__(self, threshold: int = 10):
        """
        :param threshold: the number to compare the input value against.
        """
        self.threshold = threshold

    @component.return_types(above=int, below=int)
    def run(self, value: int, threshold: Optional[int] = None):
        if threshold is None:
            threshold = self.threshold

        if value < threshold:
            return {"above": None, "below": value}
        return {"above": value, "below": None}
