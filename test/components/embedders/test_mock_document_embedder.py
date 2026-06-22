# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.components.embedders import MockDocumentEmbedder, MockTextEmbedder


def _ones(text: str) -> list[float]:
    """Module-level embedding function used to test `embedding_fn` and its serialization."""
    return [1.0, 1.0, 1.0]


class TestMockDocumentEmbedder:
    @pytest.mark.parametrize(
        ("args", "kwargs", "match"),
        [
            (([0.1],), {"embedding_fn": _ones}, "either 'embedding' or 'embedding_fn'"),
            ((), {"dimension": -1}, "must be a positive integer"),
        ],
    )
    def test_init_rejects_invalid_config(self, args, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MockDocumentEmbedder(*args, **kwargs)

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

    def test_fixed_embedding(self):
        result = MockDocumentEmbedder([0.5, 0.5]).run([Document(content="a"), Document(content="b")])
        assert all(doc.embedding == [0.5, 0.5] for doc in result["documents"])

    def test_embedding_fn(self):
        result = MockDocumentEmbedder(embedding_fn=_ones).run([Document(content="a")])
        assert result["documents"][0].embedding == [1.0, 1.0, 1.0]

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

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("embedder", MockDocumentEmbedder(dimension=8))
        restored = Pipeline.from_dict(pipeline.to_dict())
        result = restored.run({"embedder": {"documents": [Document(content="hello")]}})
        assert len(result["embedder"]["documents"][0].embedding) == 8
