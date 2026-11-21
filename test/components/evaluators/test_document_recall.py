# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack import default_from_dict
from haystack.components.evaluators.document_recall import DocumentRecallEvaluator, RecallMode
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import Document


@pytest.mark.parametrize("mode", [RecallMode.SINGLE_HIT, RecallMode.MULTI_HIT])
@pytest.mark.parametrize(
    "truth, matching, different",
    [
        (["a", "b"], ["a", "b"], ["b", "a"]),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 2, "b": 1}),
        ({"ids": ["a", "b"]}, {"ids": ["a", "b"]}, {"ids": ["b", "a"]}),
        ([], [], {}),
        ([], [], ()),
        ("a", "a", "b"),
    ],
)
def test_container_metadata_comparison(mode: RecallMode, truth: Any, matching: Any, different: Any) -> None:
    evaluator = DocumentRecallEvaluator(mode=mode, document_comparison_field="meta.source_ids")
    result = evaluator.run(
        ground_truth_documents=[
            [
                Document(meta={"source_ids": truth}),
                Document(meta={"source_ids": truth}),
                Document(meta={"source_ids": different}),
            ]
        ],
        retrieved_documents=[[Document(meta={"source_ids": matching}), Document(meta={"source_ids": matching})]],
    )

    expected = 1.0 if mode == RecallMode.SINGLE_HIT else 0.5
    assert result == {"score": expected, "individual_scores": [expected]}


def test_init_with_unknown_mode_string():
    with pytest.raises(ValueError):
        DocumentRecallEvaluator(mode="unknown_mode")


def test_init_with_string_mode():
    evaluator = DocumentRecallEvaluator(mode="single_hit")
    assert evaluator.mode == RecallMode.SINGLE_HIT

    evaluator = DocumentRecallEvaluator(mode="multi_hit")
    assert evaluator.mode == RecallMode.MULTI_HIT


def test_init_default_comparison_field():
    evaluator = DocumentRecallEvaluator()
    assert evaluator.document_comparison_field == "content"


def test_run_with_id_comparison():
    evaluator = DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT, document_comparison_field="id")
    result = evaluator.run(
        ground_truth_documents=[[Document(id="doc1", content="foo")], [Document(id="doc2", content="bar")]],
        retrieved_documents=[[Document(id="doc1", content="different")], [Document(id="wrong", content="bar")]],
    )
    assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}


def test_run_with_meta_comparison():
    evaluator = DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT, document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="x", meta={"file_id": "a"}), Document(content="y", meta={"file_id": "b"})]
        ],
        retrieved_documents=[
            [Document(content="z", meta={"file_id": "a"}), Document(content="w", meta={"file_id": "c"})]
        ],
    )
    assert result == {"individual_scores": [0.5], "score": 0.5}


def test_run_with_nested_meta_comparison():
    evaluator = DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT, document_comparison_field="meta.source.url")
    result = evaluator.run(
        ground_truth_documents=[
            [
                Document(content="x", meta={"source": {"url": "https://a.com"}}),
                Document(content="y", meta={"source": {"url": "https://b.com"}}),
            ]
        ],
        retrieved_documents=[
            [
                Document(content="z", meta={"source": {"url": "https://a.com"}}),
                Document(content="w", meta={"source": {"url": "https://c.com"}}),
            ]
        ],
    )
    assert result == {"individual_scores": [0.5], "score": 0.5}


def test_run_with_empty_inputs():
    evaluator = DocumentRecallEvaluator()

    # Empty (equal-length) inputs previously fell through the length check and
    # crashed with a bare ZeroDivisionError when averaging over zero items.
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_documents=[], retrieved_documents=[])


class TestDocumentRecallEvaluatorSingleHit:
    @pytest.fixture
    def evaluator(self):
        return DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT)

    def test_run_with_all_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}

    def test_run_with_no_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}

    def test_run_with_partial_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}

    def test_run_with_complex_data(self, evaluator):
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
                [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
                [Document(content="classical"), Document(content="rock music"), Document(content="dubstep")],
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
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1, 1, 1, 1, 0, 1], "score": 0.8333333333333334}

    def test_run_with_different_lengths(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.run(
                ground_truth_documents=[[Document(content="Berlin")]],
                retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
            )

        with pytest.raises(ValueError):
            evaluator.run(
                ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
                retrieved_documents=[[Document(content="Berlin")]],
            )

    def test_to_dict(self, evaluator):
        data = evaluator.to_dict()
        assert data == {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "single_hit", "document_comparison_field": "content", "top_k": None},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "single_hit"},
        }
        new_evaluator = default_from_dict(DocumentRecallEvaluator, data)
        assert new_evaluator.mode == RecallMode.SINGLE_HIT

    def test_empty_ground_truth_documents(self, evaluator):
        ground_truth_documents: list[list[Document]] = [[]]
        retrieved_documents = [[Document(content="test")]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_empty_retrieved_documents(self, evaluator):
        ground_truth_documents = [[Document(content="test")]]
        retrieved_documents: list[list[Document]] = [[]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_empty_string_ground_truth_documents(self, evaluator):
        ground_truth_documents = [[Document(content="")]]
        retrieved_documents = [[Document(content="test")]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_empty_string_retrieved_documents(self, evaluator):
        ground_truth_documents = [[Document(content="test")]]
        retrieved_documents = [[Document(content="")]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_no_match_when_comparison_field_missing_on_all_documents(self):
        evaluator = DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT, document_comparison_field="meta.file_id")
        result = evaluator.run(
            ground_truth_documents=[[Document(content="real ground truth")]],
            retrieved_documents=[[Document(content="totally unrelated doc")]],
        )
        assert result == {"individual_scores": [0.0], "score": 0.0}

    def test_no_match_when_comparison_field_missing_on_some_documents(self):
        evaluator = DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT, document_comparison_field="meta.file_id")
        result = evaluator.run(
            ground_truth_documents=[[Document(meta={"file_id": "f1"}), Document(content="no file_id")]],
            retrieved_documents=[[Document(content="also no file_id"), Document(meta={"file_id": "f2"})]],
        )
        assert result == {"individual_scores": [0.0], "score": 0.0}


class TestDocumentRecallEvaluatorMultiHit:
    @pytest.fixture
    def evaluator(self):
        return DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT)

    def test_run_with_all_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}

    def test_run_with_no_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}

    def test_run_with_partial_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}

    def test_run_with_complex_data(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[
                [Document(content="France")],
                [Document(content="9th century"), Document(content="9th")],
                [Document(content="classical music"), Document(content="classical")],
                [Document(content="11th century"), Document(content="the 11th")],
                [
                    Document(content="Denmark"),
                    Document(content="Iceland"),
                    Document(content="Norway"),
                    Document(content="Denmark, Iceland and Norway"),
                ],
                [Document(content="10th century"), Document(content="10th")],
            ],
            retrieved_documents=[
                [Document(content="France")],
                [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
                [Document(content="classical"), Document(content="rock music"), Document(content="dubstep")],
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
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 1.0, 0.5, 1.0, 0.75, 1.0], "score": 0.875}

    def test_run_with_different_lengths(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.run(
                ground_truth_documents=[[Document(content="Berlin")]],
                retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
            )

        with pytest.raises(ValueError):
            evaluator.run(
                ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
                retrieved_documents=[[Document(content="Berlin")]],
            )

    def test_to_dict(self, evaluator):
        data = evaluator.to_dict()
        assert data == {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "multi_hit", "document_comparison_field": "content", "top_k": None},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "multi_hit"},
        }
        new_evaluator = default_from_dict(DocumentRecallEvaluator, data)
        assert new_evaluator.mode == RecallMode.MULTI_HIT

    def test_empty_ground_truth_documents(self, evaluator):
        ground_truth_documents: list[list[Document]] = [[]]
        retrieved_documents = [[Document(content="test")]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_empty_retrieved_documents(self, evaluator):
        ground_truth_documents = [[Document(content="test")]]
        retrieved_documents: list[list[Document]] = [[]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_empty_string_ground_truth_documents(self, evaluator):
        ground_truth_documents = [[Document(content="")]]
        retrieved_documents = [[Document(content="test")]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_empty_string_retrieved_documents(self, evaluator):
        ground_truth_documents = [[Document(content="test")]]
        retrieved_documents = [[Document(content="")]]
        score = evaluator.run(ground_truth_documents, retrieved_documents)
        assert score == {"individual_scores": [0.0], "score": 0.0}

    def test_no_match_when_comparison_field_missing_on_some_documents(self):
        evaluator = DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT, document_comparison_field="meta.file_id")
        result = evaluator.run(
            ground_truth_documents=[[Document(meta={"file_id": "f1"}), Document(content="no file_id")]],
            retrieved_documents=[[Document(content="also no file_id"), Document(meta={"file_id": "f2"})]],
        )
        assert result == {"individual_scores": [0.0], "score": 0.0}

    def test_ground_truth_missing_comparison_field_excluded_from_denominator(self):
        evaluator = DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT, document_comparison_field="meta.file_id")
        result = evaluator.run(
            ground_truth_documents=[[Document(meta={"file_id": "f1"}), Document(content="no file_id")]],
            retrieved_documents=[[Document(meta={"file_id": "f1"})]],
        )
        assert result == {"individual_scores": [1.0], "score": 1.0}


def test_init_top_k_defaults_to_none():
    assert DocumentRecallEvaluator().top_k is None


@pytest.mark.parametrize("top_k", [0, -1, 2.5, True, "3"])
def test_init_with_invalid_top_k_raises(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer or None"):
        DocumentRecallEvaluator(top_k=top_k)


def test_init_with_zero_top_k_error_message():
    with pytest.raises(ValueError, match=r"^top_k must be a positive integer or None, got 0\.$"):
        DocumentRecallEvaluator(top_k=0)


def test_top_k_serialization_round_trip():
    evaluator = DocumentRecallEvaluator(mode="multi_hit", document_comparison_field="id", top_k=3)
    data = component_to_dict(evaluator, name="evaluator")
    assert data["init_parameters"]["top_k"] == 3
    restored = component_from_dict(DocumentRecallEvaluator, data, name="evaluator")
    assert restored.top_k == 3
    assert restored.document_comparison_field == evaluator.document_comparison_field


def test_from_dict_without_top_k_defaults_to_none():
    data = {
        "type": f"{DocumentRecallEvaluator.__module__}.DocumentRecallEvaluator",
        "init_parameters": {"document_comparison_field": "id"},
    }
    restored = component_from_dict(DocumentRecallEvaluator, data, name="evaluator")
    assert restored.top_k is None


def _ranked(*contents: str) -> list[Document]:
    return [Document(content=content) for content in contents]


@pytest.mark.parametrize("top_k, expected_scores", [(1, [1.0, 0.0, 0.0]), (3, [1.0, 1.0, 0.0]), (5, [1.0, 1.0, 1.0])])
def test_run_single_hit_with_top_k_only_scores_first_k_documents(top_k, expected_scores):
    evaluator = DocumentRecallEvaluator(top_k=top_k)
    result = evaluator.run(
        ground_truth_documents=[_ranked("install"), _ranked("pipelines"), _ranked("qdrant")],
        retrieved_documents=[
            _ranked("install", "faq", "intro", "docs", "blog"),
            _ranked("intro", "faq", "pipelines", "agents", "tools"),
            _ranked("weaviate", "chroma", "faq", "intro", "qdrant"),
        ],
    )
    assert result["individual_scores"] == expected_scores
    assert result["score"] == pytest.approx(sum(expected_scores) / 3)


@pytest.mark.parametrize("top_k, expected_score", [(1, 1 / 3), (2, 1 / 3), (3, 2 / 3), (4, 1.0)])
def test_run_multi_hit_with_top_k_divides_by_all_ground_truth_documents(top_k, expected_score):
    evaluator = DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT, top_k=top_k)
    result = evaluator.run(
        ground_truth_documents=[_ranked("a", "b", "c")], retrieved_documents=[_ranked("a", "x", "b", "c")]
    )
    assert result["individual_scores"] == [pytest.approx(expected_score)]


@pytest.mark.parametrize("mode", [RecallMode.SINGLE_HIT, RecallMode.MULTI_HIT])
def test_run_with_top_k_larger_than_retrieved_documents_matches_no_cutoff(mode):
    ground_truth_documents = [_ranked("a", "b"), _ranked("c")]
    retrieved_documents = [_ranked("x", "a"), _ranked("c", "y", "z")]
    with_cutoff = DocumentRecallEvaluator(mode=mode, top_k=10).run(
        ground_truth_documents=ground_truth_documents, retrieved_documents=retrieved_documents
    )
    without_cutoff = DocumentRecallEvaluator(mode=mode).run(
        ground_truth_documents=ground_truth_documents, retrieved_documents=retrieved_documents
    )
    assert with_cutoff == without_cutoff


def test_run_with_top_k_does_not_modify_inputs():
    retrieved_documents = [_ranked("x", "y", "a")]
    DocumentRecallEvaluator(top_k=1).run(ground_truth_documents=[_ranked("a")], retrieved_documents=retrieved_documents)
    assert [doc.content for doc in retrieved_documents[0]] == ["x", "y", "a"]


def test_run_with_top_k_and_meta_comparison():
    evaluator = DocumentRecallEvaluator(document_comparison_field="meta.file_id", top_k=1)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="chunk", meta={"file_id": "f1"})]],
        retrieved_documents=[
            [Document(content="other", meta={"file_id": "f2"}), Document(content="chunk", meta={"file_id": "f1"})]
        ],
    )
    assert result["individual_scores"] == [0.0]
