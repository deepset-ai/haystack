import pytest

import torch

import haystack
from haystack.schema import Document
from haystack.nodes import RAGenerator


class MockTokenizer:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def __call__(self, *a, **k):
        return {"input_ids": self}

    def to(self, *a, **k):
        return {}

    def batch_decode(self, *a, **k):
        return ["MOCK ANSWER"]


class MockModel:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def to(self, *a, **k):
        return None

    def question_encoder(self, *a, **k):
        return [torch.ones((1, 1))]

    def generate(self, *a, **k):
        return None


@pytest.fixture
def mock_models(monkeypatch):
    monkeypatch.setattr(torch, "bmm", lambda *a, **k: torch.ones((1, 1)))
    monkeypatch.setattr(haystack.nodes.answer_generator.transformers, "RagTokenForGeneration", MockModel)
    monkeypatch.setattr(haystack.nodes.answer_generator.transformers, "RagTokenizer", MockTokenizer)

    # Arguably not a great idea. To be decided.
    monkeypatch.setattr(RAGenerator, "_prepare_passage_embeddings", lambda *a, **k: torch.ones((1, 1)))
    monkeypatch.setattr(RAGenerator, "_get_contextualized_inputs", lambda *a, **k: (None, None))


@pytest.mark.unit
def test_rag_token_generator(mock_models):
    rag_generator = RAGenerator()
    generated_docs = rag_generator.predict(query="Test query", documents=[Document(content="test document")], top_k=1)
    answers = generated_docs["answers"]
    assert len(answers) == 1
    assert "MOCK ANSWER" in answers[0].answer
