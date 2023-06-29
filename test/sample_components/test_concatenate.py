# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, List

from canals.testing import BaseTestComponent
from canals.component import component


@component
class Concatenate:
    """
    Concatenates two values
    """

    @component.input
    def input(self):
        class Input:
            first: Union[List[str], str]
            second: Union[List[str], str]

        return Input

    @component.output
    def output(self):
        class Output:
            value: List[str]

        return Output

    def run(self, data):
        if type(data.first) is str and type(data.second) is str:
            res = [data.first, data.second]
        elif type(data.first) is list and type(data.second) is list:
            res = data.first + data.second
        elif type(data.first) is list and type(data.second) is str:
            res = data.first + [data.second]
        elif type(data.first) is str and type(data.second) is list:
            res = [data.first] + data.second

        return self.output(res)


class TestConcatenate(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Concatenate(), tmp_path)

    def test_input_lists(self):
        component = Concatenate()
        res = component.run(component.input(["This"], ["That"]))
        assert res == component.output(["This", "That"])

    def test_input_strings(self):
        component = Concatenate()
        res = component.run(component.input("This", "That"))
        assert res == component.output(["This", "That"])

    def test_input_first_list_second_string(self):
        component = Concatenate()
        res = component.run(component.input(["This"], "That"))
        assert res == component.output(["This", "That"])

    def test_input_first_string_second_list(self):
        component = Concatenate()
        res = component.run(component.input("This", ["That"]))
        assert res == component.output(["This", "That"])
