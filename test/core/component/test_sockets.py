# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.core.component.sockets import InputSocket, Sockets
from haystack.core.pipeline import Pipeline
from haystack.testing.factory import component_class


class TestSockets:
    def test_init(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        sockets = {"input_1": InputSocket("input_1", int), "input_2": InputSocket("input_2", int)}
        io = Sockets(component=comp, sockets_dict=sockets, sockets_io_type=InputSocket)
        assert io._component == comp
        assert "input_1" in io.__dict__
        assert io.__dict__["input_1"] == comp.__haystack_input__._sockets_dict["input_1"]
        assert "input_2" in io.__dict__
        assert io.__dict__["input_2"] == comp.__haystack_input__._sockets_dict["input_2"]

    def test_init_with_empty_sockets(self):
        comp = component_class("SomeComponent")()
        io = Sockets(component=comp, sockets_dict={}, sockets_io_type=InputSocket)

        assert io._component == comp
        assert io._sockets_dict == {}

    def test_getattribute(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = Sockets(component=comp, sockets_dict=comp.__haystack_input__._sockets_dict, sockets_io_type=InputSocket)

        assert io.input_1 == comp.__haystack_input__._sockets_dict["input_1"]
        assert io.input_2 == comp.__haystack_input__._sockets_dict["input_2"]

    def test_getattribute_non_existing_socket(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = Sockets(component=comp, sockets_dict=comp.__haystack_input__._sockets_dict, sockets_io_type=InputSocket)

        with pytest.raises(AttributeError):
            io.input_3

    def test_repr(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = Sockets(component=comp, sockets_dict=comp.__haystack_input__._sockets_dict, sockets_io_type=InputSocket)
        res = repr(io)
        assert res == "Inputs:\n  - input_1: int\n  - input_2: int"

    def test_get(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = Sockets(component=comp, sockets_dict=comp.__haystack_input__._sockets_dict, sockets_io_type=InputSocket)

        assert io.get("input_1") == comp.__haystack_input__._sockets_dict["input_1"]
        assert io.get("input_2") == comp.__haystack_input__._sockets_dict["input_2"]
        assert io.get("invalid") == None
        assert io.get("invalid", InputSocket("input_2", int)) == InputSocket("input_2", int)

    def test_contains(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = Sockets(component=comp, sockets_dict=comp.__haystack_input__._sockets_dict, sockets_io_type=InputSocket)

        assert "input_1" in io
        assert "input_2" in io
        assert "invalid" not in io
