import pytest

from haystack.schema import Document
from haystack.nodes.document_classifier.base import BaseDocumentClassifier


@pytest.mark.integration
def test_document_classifier(document_classifier):
    assert isinstance(document_classifier, BaseDocumentClassifier)

    docs = [
        Document(
            content="""That's good. I like it.""" * 700,  # extra long text to check truncation
            meta={"name": "0"},
            id="1",
        ),
        Document(content="""That's bad. I don't like it.""", meta={"name": "1"}, id="2"),
    ]
    results = document_classifier.predict(documents=docs)
    expected_labels = ["joy", "sadness"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]


@pytest.mark.integration
def test_document_classifier_details(document_classifier):
    docs = [Document(content="""That's good. I like it."""), Document(content="""That's bad. I don't like it.""")]
    results = document_classifier.predict(documents=docs)
    for doc in results:
        assert "details" in doc.meta["classification"]
        if document_classifier.top_k is not None:
            assert len(doc.meta["classification"]["details"]) == document_classifier.top_k


@pytest.mark.integration
def test_document_classifier_batch_single_doc_list(document_classifier):
    docs = [
        Document(content="""That's good. I like it.""", meta={"name": "0"}, id="1"),
        Document(content="""That's bad. I don't like it.""", meta={"name": "1"}, id="2"),
    ]
    results = document_classifier.predict_batch(documents=docs)
    expected_labels = ["joy", "sadness"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]


@pytest.mark.integration
def test_document_classifier_batch_multiple_doc_lists(document_classifier):
    docs = [
        Document(content="""That's good. I like it.""", meta={"name": "0"}, id="1"),
        Document(content="""That's bad. I don't like it.""", meta={"name": "1"}, id="2"),
    ]
    results = document_classifier.predict_batch(documents=[docs, docs])
    assert len(results) == 2  # 2 Document lists
    expected_labels = ["joy", "sadness"]
    for i, doc in enumerate(results[0]):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]


@pytest.mark.integration
def test_zero_shot_document_classifier(zero_shot_document_classifier):
    assert isinstance(zero_shot_document_classifier, BaseDocumentClassifier)

    docs = [
        Document(
            content="""That's good. I like it.""" * 700,  # extra long text to check truncation
            meta={"name": "0"},
            id="1",
        ),
        Document(content="""That's bad. I don't like it.""", meta={"name": "1"}, id="2"),
    ]
    results = zero_shot_document_classifier.predict(documents=docs)
    expected_labels = ["positive", "negative"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]


@pytest.mark.integration
def test_zero_shot_document_classifier_details(zero_shot_document_classifier):
    docs = [Document(content="""That's good. I like it."""), Document(content="""That's bad. I don't like it.""")]
    results = zero_shot_document_classifier.predict(documents=docs)
    for doc in results:
        assert "details" in doc.meta["classification"]
        assert set(doc.meta["classification"]["details"].keys()) == set(zero_shot_document_classifier.labels)


@pytest.mark.integration
def test_document_classifier_batch_size(batched_document_classifier):
    assert isinstance(batched_document_classifier, BaseDocumentClassifier)

    docs = [
        Document(
            content="""That's good. I like it.""" * 700,  # extra long text to check truncation
            meta={"name": "0"},
            id="1",
        ),
        Document(content="""That's bad. I don't like it.""", meta={"name": "1"}, id="2"),
    ]
    results = batched_document_classifier.predict(documents=docs)
    expected_labels = ["joy", "sadness"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]


@pytest.mark.integration
def test_document_classifier_as_index_node(indexing_document_classifier):
    assert isinstance(indexing_document_classifier, BaseDocumentClassifier)

    docs = [
        {
            "content": """That's good. I like it.""" * 700,  # extra long text to check truncation
            "meta": {"name": "0"},
            "id": "1",
            "class_field": "That's bad.",
        },
        {"content": """That's bad. I like it.""", "meta": {"name": "1"}, "id": "2", "class_field": "That's good."},
    ]
    output, output_name = indexing_document_classifier.run(documents=docs, root_node="File")
    expected_labels = ["sadness", "joy"]
    for i, doc in enumerate(output["documents"]):
        assert doc["meta"]["classification"]["label"] == expected_labels[i]


@pytest.mark.integration
def test_document_classifier_as_query_node(document_classifier):
    assert isinstance(document_classifier, BaseDocumentClassifier)

    docs = [
        Document(
            content="""That's good. I like it.""" * 700,  # extra long text to check truncation
            meta={"name": "0"},
            id="1",
        ),
        Document(content="""That's bad. I don't like it.""", meta={"name": "1"}, id="2"),
    ]
    output, output_name = document_classifier.run(documents=docs, root_node="Query")
    expected_labels = ["joy", "sadness"]
    for i, doc in enumerate(output["documents"]):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]
