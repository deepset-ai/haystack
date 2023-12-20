import pytest

from haystack.components.others import Multiplexer


class TestMultiplexer:
    def test_one_value(self):
        multiplexer = Multiplexer(int)
        output = multiplexer.run(value=[2])
        assert output == {"value": 2}

    def test_one_value_of_wrong_type(self):
        # Multiplexer does not type check the input
        multiplexer = Multiplexer(int)
        output = multiplexer.run(value=["hello"])
        assert output == {"value": "hello"}

    def test_one_value_of_none_type(self):
        # Multiplexer does not type check the input
        multiplexer = Multiplexer(int)
        output = multiplexer.run(value=[None])
        assert output == {"value": None}

    def test_more_values_of_expected_type(self):
        multiplexer = Multiplexer(int)
        with pytest.raises(ValueError, match="Multiplexer expects only one input, but 3 were received."):
            multiplexer.run(value=[2, 3, 4])

    def test_no_values(self):
        multiplexer = Multiplexer(int)
        with pytest.raises(ValueError, match="Multiplexer expects only one input, but 0 were received."):
            multiplexer.run(value=[])
