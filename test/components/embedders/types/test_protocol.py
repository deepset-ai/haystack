# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict

import pytest

from haystack import component
from haystack.components.embedders.types.protocol import TextEmbedder


@component
class MockTextEmbedder:
    def run(self, text: str, param_a: str = "default", param_b: str = "another_default") -> Dict[str, Any]:
        return {"embedding": [0.1, 0.2, 0.3], "metadata": {"text": text, "param_a": param_a, "param_b": param_b}}


@component
class MockInvalidTextEmbedder:
    def run(self, something_else: float) -> dict[str, bool]:
        return {"result": True}


def test_protocol_implementation():
    embedder: TextEmbedder = MockTextEmbedder()  # should not raise any type errors

    # check if the run method has the correct signature
    run_signature = inspect.signature(MockTextEmbedder.run)
    assert "text" in run_signature.parameters
    assert run_signature.parameters["text"].annotation == str
    assert run_signature.return_annotation == Dict[str, Any]

    result = embedder.run("test text")
    assert isinstance(result, dict)
    assert "embedding" in result
    assert isinstance(result["embedding"], list)
    assert all(isinstance(x, float) for x in result["embedding"])
    assert isinstance(result["metadata"], dict)


def test_protocol_optional_parameters():
    embedder = MockTextEmbedder()

    # default parameters
    result1 = embedder.run("test text")

    # with custom parameters
    result2 = embedder.run("test text", param_a="custom_a", param_b="custom_b")

    assert result1["metadata"]["param_a"] == "default"
    assert result2["metadata"]["param_a"] == "custom_a"
    assert result2["metadata"]["param_b"] == "custom_b"


def test_protocol_invalid_implementation():
    run_signature = inspect.signature(MockInvalidTextEmbedder.run)

    with pytest.raises(AssertionError):
        assert "text" in run_signature.parameters and run_signature.parameters["text"].annotation == str

    with pytest.raises(AssertionError):
        assert run_signature.return_annotation == Dict[str, Any]
