# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from haystack import Pipeline
from haystack.components.embedders import MockTextEmbedder


def _ones(text: str) -> list[float]:
    """Module-level embedding function used to test `embedding_fn` serialization."""
    return [1.0, 1.0, 1.0]


class TestMockTextEmbedder:
    def test_init_defaults(self):
        embedder = MockTextEmbedder()
        assert embedder.embedding is None
        assert embedder.embedding_fn is None
        assert embedder.dimension == 768
        assert embedder.model == "mock-model"
        assert embedder.meta == {}

    def test_init_rejects_embedding_and_embedding_fn(self):
        with pytest.raises(ValueError, match="either 'embedding' or 'embedding_fn'"):
            MockTextEmbedder([0.1, 0.2], embedding_fn=_ones)

    def test_init_rejects_non_positive_dimension(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            MockTextEmbedder(dimension=0)

    def test_init_rejects_empty_embedding(self):
        with pytest.raises(ValueError, match="must not be empty"):
            MockTextEmbedder([])

    def test_init_rejects_invalid_embedding(self):
        with pytest.raises(TypeError):
            MockTextEmbedder(["not", "numbers"])

    def test_deterministic_embedding_dimension(self):
        embedder = MockTextEmbedder(dimension=16)
        embedding = embedder.run("hello")["embedding"]
        assert len(embedding) == 16
        assert all(isinstance(value, float) for value in embedding)

    def test_deterministic_embedding_is_normalized(self):
        embedder = MockTextEmbedder(dimension=32)
        embedding = embedder.run("hello world")["embedding"]
        assert math.isclose(math.sqrt(sum(value * value for value in embedding)), 1.0, abs_tol=1e-9)

    def test_same_text_same_embedding(self):
        embedder = MockTextEmbedder(dimension=8)
        assert embedder.run("pizza")["embedding"] == embedder.run("pizza")["embedding"]

    def test_different_text_different_embedding(self):
        embedder = MockTextEmbedder(dimension=8)
        assert embedder.run("pizza")["embedding"] != embedder.run("pasta")["embedding"]

    def test_deterministic_across_instances(self):
        # determinism does not depend on instance or process state (uses a stable hash, not the salted built-in hash)
        assert (
            MockTextEmbedder(dimension=8).run("x")["embedding"] == MockTextEmbedder(dimension=8).run("x")["embedding"]
        )

    def test_fixed_embedding(self):
        embedder = MockTextEmbedder([0.1, 0.2, 0.3])
        assert embedder.run("anything")["embedding"] == [0.1, 0.2, 0.3]
        assert embedder.run("something else")["embedding"] == [0.1, 0.2, 0.3]

    def test_embedding_fn(self):
        embedder = MockTextEmbedder(embedding_fn=_ones)
        assert embedder.run("hello")["embedding"] == [1.0, 1.0, 1.0]

    def test_embedding_fn_invalid_return_raises(self):
        embedder = MockTextEmbedder(embedding_fn=lambda text: "not a vector")
        with pytest.raises(TypeError, match="must return a sequence of numbers"):
            embedder.run("hello")

    def test_prefix_suffix_affect_embedding(self):
        plain = MockTextEmbedder(dimension=8).run("hello")["embedding"]
        prefixed = MockTextEmbedder(dimension=8, prefix="search: ").run("hello")["embedding"]
        assert plain != prefixed

    def test_meta_defaults(self):
        embedder = MockTextEmbedder(dimension=4)
        meta = embedder.run("a b c")["meta"]
        assert meta["model"] == "mock-model"
        assert meta["usage"] == {"prompt_tokens": 3, "total_tokens": 3}

    def test_meta_merging(self):
        embedder = MockTextEmbedder(dimension=4, model="custom", meta={"extra": "value"})
        meta = embedder.run("hello")["meta"]
        assert meta["model"] == "custom"
        assert meta["extra"] == "value"

    def test_run_rejects_non_string(self):
        with pytest.raises(TypeError, match="expects a string"):
            MockTextEmbedder().run(["not", "a", "string"])

    async def test_run_async(self):
        embedder = MockTextEmbedder(dimension=8)
        sync_result = embedder.run("hello")
        async_result = await embedder.run_async("hello")
        assert async_result["embedding"] == sync_result["embedding"]

    def test_to_dict_from_dict_roundtrip(self):
        embedder = MockTextEmbedder(dimension=8, model="m", meta={"k": "v"}, prefix="p", suffix="s")
        data = embedder.to_dict()
        assert data["type"] == "haystack.components.embedders.mock_text_embedder.MockTextEmbedder"
        assert data["init_parameters"]["dimension"] == 8
        assert data["init_parameters"]["embedding_fn"] is None

        restored = MockTextEmbedder.from_dict(data)
        assert restored.run("hello")["embedding"] == embedder.run("hello")["embedding"]
        assert restored.meta == {"k": "v"}

    def test_to_dict_from_dict_with_embedding_fn(self):
        embedder = MockTextEmbedder(embedding_fn=_ones)
        data = embedder.to_dict()
        assert data["init_parameters"]["embedding_fn"].endswith("test_mock_text_embedder._ones")
        restored = MockTextEmbedder.from_dict(data)
        assert restored.run("hello")["embedding"] == [1.0, 1.0, 1.0]

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("embedder", MockTextEmbedder(dimension=8))
        result = pipeline.run({"embedder": {"text": "hello"}})
        assert len(result["embedder"]["embedding"]) == 8

        restored = Pipeline.from_dict(pipeline.to_dict())
        restored_result = restored.run({"embedder": {"text": "hello"}})
        assert restored_result["embedder"]["embedding"] == result["embedder"]["embedding"]
