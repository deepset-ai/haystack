from typing import Any, Union

import pytest

from haystack.tracing import utils


class NonSerializableClass:
    def __str__(self) -> str:
        return "NonSerializableClass"


class TestTypeCoercion:
    @pytest.mark.parametrize(
        "raw_value,expected_tag_value",
        [
            (1, 1),
            (1.0, 1.0),
            (True, True),
            (None, ""),
            ("string", "string"),
            ([1, 2, 3], "[1, 2, 3]"),
            ({"key": "value"}, '{"key": "value"}'),
            (NonSerializableClass(), "NonSerializableClass"),
        ],
    )
    def test_type_coercion(self, raw_value: Any, expected_tag_value: Union[bool, str, int, float]) -> None:
        coerced_value = utils.coerce_tag_value(raw_value)

        assert coerced_value == expected_tag_value
