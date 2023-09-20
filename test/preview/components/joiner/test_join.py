from typing import List

import pytest

from haystack.preview.components.joiners.join import Join


class TestJoin:
    @pytest.mark.unit
    def test_join(self):
        comp = Join(inputs_count=2, inputs_type=List[int])
        output = comp.run(input_1=[1, 2], input_2=[3, 4])
        assert output == {"output": [1, 2, 3, 4]}
