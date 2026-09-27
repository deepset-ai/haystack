# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, call

import pytest

from haystack import Document, Pipeline
from haystack.components.embedders import MockDocumentEmbedder, MockTextEmbedder


def _ones(text: str) -> list[float]:
    """Module-level embedding function used to test `embedding_fn` and its serialization."""
    return [1.0, 1.0, 1.0]


class TestMockDocumentEmbedder:
    @pytest.mark.parametrize("dimension", [768, 0, -1])
    @pytest.mark.parametrize(
        ("args", "kwargs", "exception", "match"),
        [
            (([0.1],), {"embedding_fn": _ones}, ValueError, "either 'embedding' or 'embedding_fn'"),
            (([],), {}, ValueError, "must not be empty"),
            ((["not", "numbers"],), {}, TypeError, "must be a sequence of numbers"),
        ],
    )
    def test_init_rejects_invalid_config(self, args, kwargs, exception, match, dimension):
        with pytest.raises(exception, match=match):
            MockDocumentEmbedder(*args, dimension=dimension, **kwargs)

    @pytest.mark.parametrize("dimension", [0, -1])
    def test_deterministic_embedding_rejects_non_positive_dimension(self, dimension):
        with pytest.raises(ValueError, match=r"^'dimension' must be a positive integer\.$"):
            MockDocumentEmbedder(dimension=dimension)

    def test_embeds_documents(self):
        embedder = MockDocumentEmbedder(dimension=16)
        result = embedder.run([Document(content="first"), Document(content="second")])
        embeddings = [doc.embedding for doc in result["documents"]]
        assert all(len(embedding) == 16 for embedding in embeddings)
        assert embeddings[0] != embeddings[1]

    def test_consistent_with_text_embedder(self):
        # the same prepared text yields the same embedding from both mock embedders (shared deterministic algorithm)
        text_embedding = MockTextEmbedder(dimension=8).run("pizza")["embedding"]
        doc_embedding = MockDocumentEmbedder(dimension=8).run([Document(content="pizza")])["documents"][0].embedding
        assert text_embedding == doc_embedding

    @pytest.mark.parametrize("dimension", [768, 0, -1])
    def test_fixed_embedding_ignores_dimension(self, dimension):
        result = MockDocumentEmbedder([0.5, 0.5], dimension=dimension).run(
            [Document(content="a"), Document(content="b")]
        )
        assert all(doc.embedding == [0.5, 0.5] for doc in result["documents"])

    @pytest.mark.parametrize("dimension", [768, 0, -1])
    def test_embedding_fn_ignores_dimension(self, dimension):
        embedding_fn = Mock(wraps=_ones)
        result = MockDocumentEmbedder(embedding_fn=embedding_fn, dimension=dimension).run(
            [Document(content="a"), Document(content="b")]
        )
        assert [doc.embedding for doc in result["documents"]] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        assert embedding_fn.call_args_list == [call("a"), call("b")]

    @pytest.mark.parametrize("dimension", [768, 0, -1])
    @pytest.mark.parametrize(
        ("embedding", "exception", "match"),
        [([], ValueError, "must not be empty"), ("not a vector", TypeError, "must be a sequence of numbers")],
    )
    def test_embedding_fn_invalid_return_raises(self, dimension, embedding, exception, match):
        embedding_fn = Mock(return_value=embedding)
        embedder = MockDocumentEmbedder(embedding_fn=embedding_fn, dimension=dimension)
        with pytest.raises(exception, match=match):
            embedder.run([Document(content="hello")])
        embedding_fn.assert_called_once_with("hello")

    def test_meta_fields_to_embed_affect_embedding(self):
        document = Document(content="hello", meta={"title": "Greetings"})
        without_meta = MockDocumentEmbedder(dimension=8).run([document])["documents"][0].embedding
        with_meta = (
            MockDocumentEmbedder(dimension=8, meta_fields_to_embed=["title"]).run([document])["documents"][0].embedding
        )
        assert without_meta != with_meta

    def test_preserves_document_fields(self):
        document = Document(content="hello", meta={"title": "Greetings"})
        embedded = MockDocumentEmbedder(dimension=8).run([document])["documents"][0]
        assert embedded.id == document.id
        assert embedded.content == "hello"
        assert embedded.meta == {"title": "Greetings"}
        assert embedded.embedding is not None
        # the original document is not mutated
        assert document.embedding is None

    def test_empty_documents(self):
        result = MockDocumentEmbedder().run([])
        assert result["documents"] == []
        assert result["meta"]["usage"] == {"prompt_tokens": 0, "total_tokens": 0}

    def test_meta(self):
        meta = MockDocumentEmbedder(dimension=4).run([Document(content="a b"), Document(content="c")])["meta"]
        assert meta["model"] == "mock-model"
        assert meta["usage"] == {"prompt_tokens": 3, "total_tokens": 3}

    @pytest.mark.parametrize("documents", ["not a list", [1, 2, 3]])
    def test_run_rejects_non_documents(self, documents):
        with pytest.raises(TypeError, match="expects a list of Documents"):
            MockDocumentEmbedder().run(documents)

    async def test_run_async(self):
        embedder = MockDocumentEmbedder(dimension=8)
        documents = [Document(content="hello")]
        async_embedding = (await embedder.run_async(documents))["documents"][0].embedding
        assert async_embedding == embedder.run(documents)["documents"][0].embedding

    def test_warm_up_sets_flag_and_run_auto_warms(self):
        embedder = MockDocumentEmbedder(dimension=8)
        assert embedder._is_warmed_up is False
        embedder.warm_up()
        assert embedder._is_warmed_up is True
        # run auto-warms a fresh instance
        fresh = MockDocumentEmbedder(dimension=8)
        assert len(fresh.run([Document(content="hello")])["documents"][0].embedding) == 8
        assert fresh._is_warmed_up is True

    @pytest.mark.parametrize(
        "embedder",
        [
            MockDocumentEmbedder(
                dimension=8, model="m", meta={"k": "v"}, meta_fields_to_embed=["title"], embedding_separator=" | "
            ),
            MockDocumentEmbedder(embedding_fn=_ones),
        ],
        ids=["deterministic", "embedding_fn"],
    )
    def test_serialization_roundtrip(self, embedder):
        restored = MockDocumentEmbedder.from_dict(embedder.to_dict())
        assert isinstance(restored, MockDocumentEmbedder)
        document = Document(content="hello", meta={"title": "t"})
        assert restored.run([document])["documents"][0].embedding == embedder.run([document])["documents"][0].embedding

    @pytest.mark.parametrize("dimension", [0, -1])
    @pytest.mark.parametrize(
        "kwargs", [{"embedding": [0.1, 0.2]}, {"embedding_fn": _ones}], ids=["fixed", "embedding_fn"]
    )
    def test_custom_embedding_serialization_roundtrip(self, dimension, kwargs):
        embedder = MockDocumentEmbedder(dimension=dimension, **kwargs)
        restored = MockDocumentEmbedder.from_dict(embedder.to_dict())
        assert restored.dimension == dimension
        assert restored.embedding == embedder.embedding
        assert restored.embedding_fn is embedder.embedding_fn
        document = Document(content="hello")
        assert restored.run([document])["documents"][0].embedding == embedder.run([document])["documents"][0].embedding

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("embedder", MockDocumentEmbedder(dimension=8))
        restored = Pipeline.from_dict(pipeline.to_dict())
        result = restored.run({"embedder": {"documents": [Document(content="hello")]}})
        assert len(result["embedder"]["documents"][0].embedding) == 8
