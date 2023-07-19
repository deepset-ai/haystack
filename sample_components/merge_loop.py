# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import builtins
from typing import List, Union, Optional
from dataclasses import make_dataclass, is_dataclass, asdict

from canals import component


@component
class MergeLoop:
    """
    Takes multiple inputs and returns the first one that is not None.
    """

    def __init__(self, expected_type: Union[type, str], inputs: List[str] = ["value_1", "value_2"]):
        if isinstance(expected_type, str):
            self.expected_type = getattr(builtins, expected_type)
        else:
            self.expected_type = expected_type
        self.init_parameters = {"expected_type": self.expected_type.__name__}
        # mypy complains that we can't Optional is not a type, so we ignore the error
        # cause we consider this to be correct
        self._input = make_dataclass("Input", fields=[(f, Optional[self.expected_type]) for f in inputs])  # type: ignore

    @component.input
    def input(self):
        return self._input

    @component.output
    def output(self):
        class Output:
            value: self.expected_type  # type: ignore

        return Output

    def run(self, data):
        """
        Takes some inputs and returns the first one that is not None.
        """
        values = []
        if is_dataclass(data):
            values = asdict(data).values()
        for v in values:
            if v is not None:
                return self.output(value=v)
        return self.output(value=None)
