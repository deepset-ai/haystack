from typing import List

import pytest

from haystack.preview.components.joiners.join import Join


class TestJoin:
    @pytest.mark.unit
    def test_join_to_dict(self):
        comp = Join(inputs_count=2, inputs_type=str)
        assert comp.to_dict() == {"type": "Join", "init_parameters": {"inputs_count": 2, "inputs_type": "str"}}

    @pytest.mark.unit
    def test_join_from_dict(self):
        data = {"type": "Join", "init_parameters": {"inputs_count": 2, "inputs_type": "str"}}
        comp = Join.from_dict(data)
        assert comp.inputs_count == 2
        assert comp.inputs_type == str

    @pytest.mark.unit
    def test_join_list(self):
        comp = Join(inputs_count=2, inputs_type=List[int])
        output = comp.run(input_0=[1, 2], input_1=[3, 4])
        assert output == {"output": [1, 2, 3, 4]}

    @pytest.mark.unit
    def test_join_str(self):
        comp = Join(inputs_count=2, inputs_type=str)
        output = comp.run(input_0="hello", input_1="test")
        assert output == {"output": "hellotest"}

    @pytest.mark.unit
    def test_join_one_input(self):
        comp = Join(inputs_count=1, inputs_type=str)
        output = comp.run(input_0="hello")
        assert output == {"output": "hello"}

    @pytest.mark.unit
    def test_join_zero_input(self):
        with pytest.raises(ValueError):
            Join(inputs_count=0, inputs_type=str)
