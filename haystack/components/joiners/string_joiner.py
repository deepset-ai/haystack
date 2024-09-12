# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from haystack import component, logging
from haystack.core.component.types import Variadic

logger = logging.getLogger(__name__)


@component
class StringJoiner:
    """
    Component to join strings from different components to a list of strings
    """

    @component.output_types(strings=List[str])
    def run(self, strings: Variadic[str]):
        """
        Joins strings into a list of strings

        :param strings: strings from different components
        """

        out_strings = list(strings)
        return {"strings": out_strings}
