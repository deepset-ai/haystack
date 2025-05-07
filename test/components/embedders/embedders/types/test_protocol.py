import pytest
from typing import Any, Dict

from haystack.components.embedders.types.protocol import TextEmbedder


class MockTextEmbedder:
    def run(self, text: str, param_a: str = "default", param_b: str = "another_default") -> Dict[str, Any]:
        """
        Mock implementation that returns a simple embedding and metadata.
        """
        return {"embedding": [0.1, 0.2, 0.3], "metadata": {"text": text, "param_a": param_a, "param_b": param_b}}


def test_protocol_implementation():
    embedder: TextEmbedder = MockTextEmbedder()  # should not raise any type errors

    result = embedder.run("test text")
    assert isinstance(result, dict)
    assert "embedding" in result
    assert "metadata" in result
    assert isinstance(result["embedding"], list)
    assert isinstance(result["metadata"], dict)


def test_protocol_optional_parameters():
    embedder: TextEmbedder = MockTextEmbedder()  # should not raise any type errors

    # default parameters
    result1 = embedder.run("test text")

    # with custom parameters
    result2 = embedder.run("test text", param_a="custom_a", param_b="custom_b")

    assert result1["metadata"]["param_a"] == "default"
    assert result1["metadata"]["param_b"] == "another_default"
    assert result2["metadata"]["param_a"] == "custom_a"
    assert result2["metadata"]["param_b"] == "custom_b"
