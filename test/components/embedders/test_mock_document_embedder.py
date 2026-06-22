# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.components.embedders import MockDocumentEmbedder, MockTextEmbedder


def _ones(text: str) -> list[float]:
    """Module-level embedding function used to test `embedding_fn` serialization."""
    return [1.0, 1.0, 1.0]


class TestMockDocumentEmbedder:
    def test_init_defaults(self):
        embedder = MockDocumentEmbedder()
        assert embedder.embedding is None
        assert embedder.embedding_fn is None
        assert embedder.dimension == 768
        assert embedder.model == "mock-model"
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_rejects_embedding_and_embedding_fn(self):
        with pytest.raises(ValueError, match="either 'embedding' or 'embedding_fn'"):
            MockDocumentEmbedder([0.1], embedding_fn=_ones)

    def test_init_rejects_non_positive_dimension(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            MockDocumentEmbedder(dimension=-1)

    def test_deterministic_embeddings(self):
        embedder = MockDocumentEmbedder(dimension=16)
        documents = [Document(content="first"), Document(content="second")]
        result = embedder.run(documents)
        embeddings = [doc.embedding for doc in result["documents"]]
        assert all(len(embedding) == 16 for embedding in embeddings)
        assert embeddings[0] != embeddings[1]

    def test_same_content_same_embedding(self):
        embedder = MockDocumentEmbedder(dimension=8)
        first = embedder.run([Document(content="pizza")])["documents"][0].embedding
        second = embedder.run([Document(content="pizza")])["documents"][0].embedding
        assert first == second

    def test_consistent_with_text_embedder(self):
        # the same prepared text must yield the same embedding from both mock embedders
        text_embedding = MockTextEmbedder(dimension=8).run("pizza")["embedding"]
        doc_embedding = MockDocumentEmbedder(dimension=8).run([Document(content="pizza")])["documents"][0].embedding
        assert text_embedding == doc_embedding

    def test_fixed_embedding(self):
        embedder = MockDocumentEmbedder([0.5, 0.5])
        result = embedder.run([Document(content="a"), Document(content="b")])
        assert all(doc.embedding == [0.5, 0.5] for doc in result["documents"])

    def test_embedding_fn(self):
        embedder = MockDocumentEmbedder(embedding_fn=_ones)
        result = embedder.run([Document(content="a")])
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
        result = MockDocumentEmbedder(dimension=8).run([document])
        embedded = result["documents"][0]
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

    def test_meta_defaults(self):
        embedder = MockDocumentEmbedder(dimension=4)
        meta = embedder.run([Document(content="a b"), Document(content="c")])["meta"]
        assert meta["model"] == "mock-model"
        assert meta["usage"] == {"prompt_tokens": 3, "total_tokens": 3}

    def test_run_rejects_non_documents(self):
        with pytest.raises(TypeError, match="expects a list of Documents"):
            MockDocumentEmbedder().run("not a list")

        with pytest.raises(TypeError, match="expects a list of Documents"):
            MockDocumentEmbedder().run([1, 2, 3])

    async def test_run_async(self):
        embedder = MockDocumentEmbedder(dimension=8)
        documents = [Document(content="hello")]
        sync_result = embedder.run(documents)
        async_result = await embedder.run_async(documents)
        assert async_result["documents"][0].embedding == sync_result["documents"][0].embedding

    def test_to_dict_from_dict_roundtrip(self):
        embedder = MockDocumentEmbedder(
            dimension=8, model="m", meta={"k": "v"}, meta_fields_to_embed=["title"], embedding_separator=" | "
        )
        data = embedder.to_dict()
        assert data["type"] == "haystack.components.embedders.mock_document_embedder.MockDocumentEmbedder"
        assert data["init_parameters"]["dimension"] == 8
        assert data["init_parameters"]["meta_fields_to_embed"] == ["title"]
        assert data["init_parameters"]["embedding_fn"] is None

        restored = MockDocumentEmbedder.from_dict(data)
        document = Document(content="hello", meta={"title": "t"})
        assert restored.run([document])["documents"][0].embedding == embedder.run([document])["documents"][0].embedding

    def test_to_dict_from_dict_with_embedding_fn(self):
        embedder = MockDocumentEmbedder(embedding_fn=_ones)
        data = embedder.to_dict()
        assert data["init_parameters"]["embedding_fn"].endswith("test_mock_document_embedder._ones")
        restored = MockDocumentEmbedder.from_dict(data)
        assert restored.run([Document(content="a")])["documents"][0].embedding == [1.0, 1.0, 1.0]

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("embedder", MockDocumentEmbedder(dimension=8))
        result = pipeline.run({"embedder": {"documents": [Document(content="hello")]}})
        assert len(result["embedder"]["documents"][0].embedding) == 8

        restored = Pipeline.from_dict(pipeline.to_dict())
        restored_result = restored.run({"embedder": {"documents": [Document(content="hello")]}})
        assert restored_result["embedder"]["documents"][0].embedding == result["embedder"]["documents"][0].embedding
