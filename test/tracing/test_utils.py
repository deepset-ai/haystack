# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Union

import pytest

from haystack import Document
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
            (
                Document(id="1", content="text"),
                '{"id": "1", "content": "text", "dataframe": null, "blob": null, "score": null, "embedding": null, "sparse_embedding": null}',
            ),
            (
                [Document(id="1", content="text")],
                '[{"id": "1", "content": "text", "dataframe": null, "blob": null, "score": null, "embedding": null, "sparse_embedding": null}]',
            ),
            (
                {"key": Document(id="1", content="text")},
                '{"key": {"id": "1", "content": "text", "dataframe": null, "blob": null, "score": null, "embedding": null, "sparse_embedding": null}}',
            ),
        ],
    )
    def test_type_coercion(self, raw_value: Any, expected_tag_value: Union[bool, str, int, float]) -> None:
        coerced_value = utils.coerce_tag_value(raw_value)

        assert coerced_value == expected_tag_value
