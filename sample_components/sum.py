# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals import component
from canals.component.types import Variadic


@component
class Sum:
    @component.output_types(total=int)
    def run(self, values: Variadic[int]):
        """
        :param value: the value to check the remainder of.
        """
        return {"total": sum(values)}  # type: ignore
