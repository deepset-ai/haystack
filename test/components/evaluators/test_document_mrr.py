# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from haystack import Document, default_from_dict
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.core.serialization import component_from_dict, component_to_dict


def test_to_dict():
    evaluator = DocumentMRREvaluator()
    data = evaluator.to_dict()
    assert data == {
        "type": "haystack.components.evaluators.document_mrr.DocumentMRREvaluator",
        "init_parameters": {"document_comparison_field": "content", "top_k": None},
    }


def test_to_dict_custom_field():
    evaluator = DocumentMRREvaluator(document_comparison_field="id")
    data = evaluator.to_dict()
    assert data == {
        "type": "haystack.components.evaluators.document_mrr.DocumentMRREvaluator",
        "init_parameters": {"document_comparison_field": "id", "top_k": None},
    }


def test_from_dict():
    data = {
        "type": "haystack.components.evaluators.document_mrr.DocumentMRREvaluator",
        "init_parameters": {"document_comparison_field": "id"},
    }
    evaluator = default_from_dict(DocumentMRREvaluator, data)
    assert evaluator.document_comparison_field == "id"


def test_run_with_id_comparison():
    evaluator = DocumentMRREvaluator(document_comparison_field="id")
    result = evaluator.run(
        ground_truth_documents=[[Document(id="doc1", content="foo")], [Document(id="doc2", content="bar")]],
        retrieved_documents=[[Document(id="doc1", content="different")], [Document(id="wrong", content="bar")]],
    )
    assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}


def test_run_with_meta_comparison():
    evaluator = DocumentMRREvaluator(document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="x", meta={"file_id": "a"})],
            [Document(content="y", meta={"file_id": "b"})],
        ],
        retrieved_documents=[
            [Document(content="z", meta={"file_id": "a"})],
            [Document(content="w", meta={"file_id": "c"})],
        ],
    )
    assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}


def test_run_with_empty_comparison_values_does_not_match(caplog):
    evaluator = DocumentMRREvaluator(document_comparison_field="meta.file_id")
    with caplog.at_level(logging.WARNING):
        result = evaluator.run(
            ground_truth_documents=[
                [Document(content="Strasbourg", meta={"file_id": ""})],
                [Document(content="Paris", meta={"file_id": "a"})],
            ],
            retrieved_documents=[
                [Document(content="pasta", meta={"file_id": ""})],
                [Document(content="pasta", meta={"file_id": ""}), Document(content="Paris", meta={"file_id": "a"})],
            ],
        )
    assert result == {"individual_scores": [0.0, 0.5], "score": 0.25}
    assert "none of them contain a valid comparison value" in caplog.text


def test_run_with_nested_meta_comparison():
    evaluator = DocumentMRREvaluator(document_comparison_field="meta.source.url")
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="x", meta={"source": {"url": "https://a.com"}})],
            [Document(content="y", meta={"source": {"url": "https://b.com"}})],
        ],
        retrieved_documents=[
            [Document(content="z", meta={"source": {"url": "https://a.com"}})],
            [Document(content="w", meta={"source": {"url": "https://c.com"}})],
        ],
    )
    assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}


def test_run_with_all_matching():
    evaluator = DocumentMRREvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
    )

    assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}


def test_run_with_no_matching():
    evaluator = DocumentMRREvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
    )

    assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}


def test_run_with_partial_matching():
    evaluator = DocumentMRREvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
    )

    assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}


def test_run_with_complex_data():
    evaluator = DocumentMRREvaluator()
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="9th")],
            [Document(content="classical music"), Document(content="classical")],
            [Document(content="11th century"), Document(content="the 11th")],
            [Document(content="Denmark, Iceland and Norway")],
            [Document(content="10th century"), Document(content="10th")],
        ],
        retrieved_documents=[
            [Document(content="France")],
            [Document(content="10th century"), Document(content="9th century"), Document(content="9th")],
            [Document(content="rock music"), Document(content="dubstep"), Document(content="classical")],
            [Document(content="11th"), Document(content="the 11th"), Document(content="11th century")],
            [Document(content="Denmark"), Document(content="Norway"), Document(content="Iceland")],
            [
                Document(content="10th century"),
                Document(content="the first half of the 10th century"),
                Document(content="10th"),
                Document(content="10th"),
            ],
        ],
    )

    assert result == {
        "individual_scores": [1.0, 0.5, 0.3333333333333333, 0.5, 0.0, 1.0],
        "score": pytest.approx(0.555555555555555),
    }


def test_run_with_different_lengths():
    with pytest.raises(ValueError):
        evaluator = DocumentMRREvaluator()
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )

    with pytest.raises(ValueError):
        evaluator = DocumentMRREvaluator()
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")]],
        )


def test_run_with_empty_inputs():
    evaluator = DocumentMRREvaluator()

    # Empty (equal-length) inputs previously fell through the length check and
    # crashed with a bare ZeroDivisionError when averaging over zero items.
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_documents=[], retrieved_documents=[])


def test_init_top_k_defaults_to_none():
    assert DocumentMRREvaluator().top_k is None


@pytest.mark.parametrize("top_k", [0, -1, 2.5, True, "3"])
def test_init_with_invalid_top_k_raises(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer or None"):
        DocumentMRREvaluator(top_k=top_k)


def test_init_with_zero_top_k_error_message():
    with pytest.raises(ValueError, match=r"^top_k must be a positive integer or None, got 0\.$"):
        DocumentMRREvaluator(top_k=0)


def test_top_k_serialization_round_trip():
    evaluator = DocumentMRREvaluator(document_comparison_field="id", top_k=3)
    data = component_to_dict(evaluator, name="evaluator")
    assert data["init_parameters"]["top_k"] == 3
    restored = component_from_dict(DocumentMRREvaluator, data, name="evaluator")
    assert restored.top_k == 3
    assert restored.document_comparison_field == evaluator.document_comparison_field


def test_from_dict_without_top_k_defaults_to_none():
    data = {
        "type": f"{DocumentMRREvaluator.__module__}.DocumentMRREvaluator",
        "init_parameters": {"document_comparison_field": "id"},
    }
    restored = component_from_dict(DocumentMRREvaluator, data, name="evaluator")
    assert restored.top_k is None


@pytest.mark.parametrize("top_k, expected_score", [(1, 0.0), (2, 0.0), (3, 1 / 3), (5, 1 / 3)])
def test_run_with_top_k_ignores_relevant_documents_below_cutoff(top_k, expected_score):
    evaluator = DocumentMRREvaluator(top_k=top_k)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="pipelines")]],
        retrieved_documents=[
            [
                Document(content="intro"),
                Document(content="faq"),
                Document(content="pipelines"),
                Document(content="agents"),
                Document(content="tools"),
            ]
        ],
    )
    assert result["individual_scores"] == [pytest.approx(expected_score)]
    assert result["score"] == pytest.approx(expected_score)


def test_run_with_top_k_averages_over_all_questions():
    evaluator = DocumentMRREvaluator(top_k=1)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="a")], [Document(content="b")]],
        retrieved_documents=[
            [Document(content="a"), Document(content="x")],
            [Document(content="x"), Document(content="b")],
        ],
    )
    assert result["individual_scores"] == [1.0, 0.0]
    assert result["score"] == 0.5


def test_run_with_top_k_larger_than_retrieved_documents_matches_no_cutoff():
    ground_truth_documents = [[Document(content="a")], [Document(content="c")]]
    retrieved_documents = [[Document(content="x"), Document(content="a")], [Document(content="y")]]
    with_cutoff = DocumentMRREvaluator(top_k=10).run(
        ground_truth_documents=ground_truth_documents, retrieved_documents=retrieved_documents
    )
    without_cutoff = DocumentMRREvaluator().run(
        ground_truth_documents=ground_truth_documents, retrieved_documents=retrieved_documents
    )
    assert with_cutoff == without_cutoff
