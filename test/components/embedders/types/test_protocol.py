import inspect
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

    # check if the run method has the correct signature
    run_signature = inspect.signature(MockTextEmbedder.run)
    assert "text" in run_signature.parameters
    assert run_signature.parameters["text"].annotation == str
    assert run_signature.return_annotation == Dict[str, Any]

    result = embedder.run("test text")
    assert isinstance(result, dict)
    assert "embedding" in result
    assert "metadata" in result
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


"""
# Let's also add one or two InvalidTextEmbedder examples that don't have text as an input parameter, or don't have a key "embedding" in the output, or don't have a List[float] as value of the "embedding" key in the output.
class InvalidTextEmbedder:
    def run(self, text: str) -> Dict[str, Any]:

        # Invalid implementation that does not return the expected keys.

        return {"invalid_key": [0.1, 0.2, 0.3]}  # missing 'embedding' key

class InvalidTextEmbedder2:
    def run(self, text: str) -> Dict[str, Any]:

        # Invalid implementation that does not return the expected type for 'embedding'.

        return {"embedding": "not_a_list"}  # 'embedding' should be a List[float]

def test_invalid_protocol_implementation():
    embedder: TextEmbedder = InvalidTextEmbedder()  # should not raise any type errors

    result = embedder.run("test text")
    assert isinstance(result, dict)
    assert "embedding" not in result  # should not have 'embedding' key
    assert "invalid_key" in result
    assert isinstance(result["invalid_key"], list)  # 'invalid_key' is a list, but not the expected one
"""
