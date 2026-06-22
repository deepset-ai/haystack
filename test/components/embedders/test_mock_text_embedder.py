# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from haystack import Pipeline
from haystack.components.embedders import MockTextEmbedder
from haystack.components.embedders.mock_utils import _l2_normalize


def _ones(text: str) -> list[float]:
    """Module-level embedding function used to test `embedding_fn` and its serialization."""
    return [1.0, 1.0, 1.0]


def test_l2_normalize_handles_zero_vector():
    # defensive guard in the shared deterministic-embedding helper: a zero vector is returned unchanged
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class TestMockTextEmbedder:
    @pytest.mark.parametrize(
        ("args", "kwargs", "exception", "match"),
        [
            (([0.1, 0.2],), {"embedding_fn": _ones}, ValueError, "either 'embedding' or 'embedding_fn'"),
            ((), {"dimension": 0}, ValueError, "must be a positive integer"),
            (([],), {}, ValueError, "must not be empty"),
            ((["not", "numbers"],), {}, TypeError, "must be a sequence of numbers"),
        ],
    )
    def test_init_rejects_invalid_config(self, args, kwargs, exception, match):
        with pytest.raises(exception, match=match):
            MockTextEmbedder(*args, **kwargs)

    def test_deterministic_embedding(self):
        embedding = MockTextEmbedder(dimension=16).run("hello")["embedding"]
        assert len(embedding) == 16
        assert all(isinstance(value, float) for value in embedding)
        # embeddings are L2-normalized, like real ones
        assert math.isclose(math.sqrt(sum(value * value for value in embedding)), 1.0, abs_tol=1e-9)

    def test_deterministic_distinguishes_texts(self):
        embedder = MockTextEmbedder(dimension=8)
        assert embedder.run("pizza")["embedding"] == embedder.run("pizza")["embedding"]
        assert embedder.run("pizza")["embedding"] != embedder.run("pasta")["embedding"]
        # determinism holds across instances and processes (stable hash, not the salted built-in hash)
        assert (
            MockTextEmbedder(dimension=8).run("x")["embedding"] == MockTextEmbedder(dimension=8).run("x")["embedding"]
        )

    def test_fixed_embedding(self):
        embedder = MockTextEmbedder([0.1, 0.2, 0.3])
        assert embedder.run("anything")["embedding"] == [0.1, 0.2, 0.3]
        assert embedder.run("something else")["embedding"] == [0.1, 0.2, 0.3]

    def test_embedding_fn(self):
        assert MockTextEmbedder(embedding_fn=_ones).run("hello")["embedding"] == [1.0, 1.0, 1.0]

    def test_embedding_fn_invalid_return_raises(self):
        embedder = MockTextEmbedder(embedding_fn=lambda text: "not a vector")
        with pytest.raises(TypeError, match="must return a sequence of numbers"):
            embedder.run("hello")

    def test_prefix_suffix_affect_embedding(self):
        plain = MockTextEmbedder(dimension=8).run("hello")["embedding"]
        prefixed = MockTextEmbedder(dimension=8, prefix="search: ").run("hello")["embedding"]
        assert plain != prefixed

    def test_meta(self):
        meta = MockTextEmbedder(dimension=4).run("a b c")["meta"]
        assert meta["model"] == "mock-model"
        assert meta["usage"] == {"prompt_tokens": 3, "total_tokens": 3}
        # init model and meta are reflected and merged
        custom = MockTextEmbedder(dimension=4, model="custom", meta={"extra": "value"}).run("hi")["meta"]
        assert custom["model"] == "custom"
        assert custom["extra"] == "value"

    def test_run_rejects_non_string(self):
        with pytest.raises(TypeError, match="expects a string"):
            MockTextEmbedder().run(["not", "a", "string"])

    async def test_run_async(self):
        embedder = MockTextEmbedder(dimension=8)
        assert (await embedder.run_async("hello"))["embedding"] == embedder.run("hello")["embedding"]

    @pytest.mark.parametrize(
        "embedder",
        [
            MockTextEmbedder(dimension=8, model="m", meta={"k": "v"}, prefix="p", suffix="s"),
            MockTextEmbedder(embedding_fn=_ones),
            MockTextEmbedder([0.1, 0.2, 0.3]),
        ],
        ids=["deterministic", "embedding_fn", "fixed"],
    )
    def test_serialization_roundtrip(self, embedder):
        restored = MockTextEmbedder.from_dict(embedder.to_dict())
        assert isinstance(restored, MockTextEmbedder)
        assert restored.run("hello")["embedding"] == embedder.run("hello")["embedding"]

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("embedder", MockTextEmbedder(dimension=8))
        restored = Pipeline.from_dict(pipeline.to_dict())
        result = restored.run({"embedder": {"text": "hello"}})
        assert len(result["embedder"]["embedding"]) == 8
