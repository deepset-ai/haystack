from pathlib import Path

import pytest

from haystack.nodes import QuestionGenerator, EmbeddingRetriever, PseudoLabelGenerator
from test.conftest import DOCS_WITH_EMBEDDINGS


@pytest.mark.generator
@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["embedding_sbert"], indirect=True)
def test_pseudo_label_generator(
    document_store, retriever: EmbeddingRetriever, question_generator: QuestionGenerator, tmp_path: Path
):
    document_store.write_documents(DOCS_WITH_EMBEDDINGS)
    psg = PseudoLabelGenerator(question_generator, retriever)
    train_examples = []
    for idx, doc in enumerate(document_store):
        output, stream = psg.run(documents=[doc])
        assert "gpl_labels" in output
        for item in output["gpl_labels"]:
            assert "question" in item and "pos_doc" in item and "neg_doc" in item and "score" in item
            train_examples.append(item)

    assert len(train_examples) > 0
    retriever.train(train_examples)
    retriever.save(tmp_path)


@pytest.mark.generator
@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["embedding_sbert"], indirect=True)
def test_pseudo_label_generator_using_question_document_pairs(
    document_store, retriever: EmbeddingRetriever, tmp_path: Path
):
    document_store.write_documents(DOCS_WITH_EMBEDDINGS)
    docs = [
        {
            "question": "What is the capital of Germany?",
            "document": "Berlin is the capital and largest city of Germany by both area and population.",
        },
        {
            "question": "What is the largest city in Germany by population and area?",
            "document": "Berlin is the capital and largest city of Germany by both area and population.",
        },
    ]
    psg = PseudoLabelGenerator(docs, retriever)
    train_examples = []
    for idx, doc in enumerate(document_store):
        # the documents passed here are ignored as we provided source documents in the constructor
        output, stream = psg.run(documents=[doc])
        assert "gpl_labels" in output
        for item in output["gpl_labels"]:
            assert "question" in item and "pos_doc" in item and "neg_doc" in item and "score" in item
            train_examples.append(item)

    assert len(train_examples) > 0

    retriever.train(train_examples)
    retriever.save(tmp_path)
