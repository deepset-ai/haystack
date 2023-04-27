from typing import List
from uuid import UUID

from numpy import loadtxt

import pytest

from haystack.schema import Document
from haystack.nodes import Seq2SeqGenerator, SentenceTransformersRanker, TopPSampler, TransformersDocumentClassifier


@pytest.fixture
def docs_with_true_emb(test_rootdir):
    return [
        Document(
            content="The capital of Germany is the city state of Berlin.",
            embedding=loadtxt(test_rootdir / "samples" / "embeddings" / "embedding_1.txt"),
        ),
        Document(
            content="Berlin is the capital and largest city of Germany by both area and population.",
            embedding=loadtxt(test_rootdir / "samples" / "embeddings" / "embedding_2.txt"),
        ),
    ]


@pytest.fixture
def docs_with_ids(docs) -> List[Document]:
    # Should be already sorted
    uuids = [
        UUID("190a2421-7e48-4a49-a639-35a86e202dfb"),
        UUID("20ff1706-cb55-4704-8ae8-a3459774c8dc"),
        UUID("5078722f-07ae-412d-8ccb-b77224c4bacb"),
        UUID("81d8ca45-fad1-4d1c-8028-d818ef33d755"),
        UUID("f985789f-1673-4d8f-8d5f-2b8d3a9e8e23"),
    ]
    uuids.sort()
    for doc, uuid in zip(docs, uuids):
        doc.id = str(uuid)
    return docs


@pytest.fixture
def lfqa_generator(request):
    return Seq2SeqGenerator(model_name_or_path=request.param, min_length=100, max_length=200)


@pytest.fixture
def ranker_two_logits():
    return SentenceTransformersRanker(model_name_or_path="deepset/gbert-base-germandpr-reranking")


@pytest.fixture
def ranker():
    return SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")


@pytest.fixture
def top_p_sampler():
    return TopPSampler()


@pytest.fixture
def document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion", use_gpu=False, top_k=2
    )


@pytest.fixture
def zero_shot_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="cross-encoder/nli-distilroberta-base",
        use_gpu=False,
        task="zero-shot-classification",
        labels=["negative", "positive"],
    )


@pytest.fixture
def batched_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion", use_gpu=False, batch_size=16
    )


@pytest.fixture
def indexing_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion",
        use_gpu=False,
        batch_size=16,
        classification_field="class_field",
    )
