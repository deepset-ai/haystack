# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from canals import component


class Repeat:
    @staticmethod
    def create(outputs: List[str]):
        """
        Repeats the input value on all outputs.
        """

        @component
        class RepeatImpl:
            """
            Implementation of Repeat()
            """

            def __init__(self, outputs: List[str]):
                self.outputs = outputs

            @component.return_types(**{val: int for val in outputs})
            def run(self, value: int):
                """
                :param value: the value to repeat.
                """
                return {val: value for val in self.outputs}

        return RepeatImpl(outputs=outputs)

    def __init__(self):
        raise NotImplementedError("use Repeat.create()")
