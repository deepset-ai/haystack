# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.core.component import component
from haystack.core.component.types import Variadic


@component
class Sum:
    @component.output_types(total=int)
    def run(self, values: Variadic[int]):
        """
        :param value: the values to sum.
        """
        return {"total": sum(values)}
