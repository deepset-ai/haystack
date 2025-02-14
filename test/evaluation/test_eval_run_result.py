# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json

from haystack.evaluation import EvaluationRunResult
import pytest


def test_init_results_evaluator():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": {
            "reciprocal_rank": {"individual_scores": [0.378064, 0.534964], "score": 0.476932},
            "single_hit": {"individual_scores": [1, 1], "score": 0.75},
            "multi_hit": {"individual_scores": [0.706125, 0.454976], "score": 0.46428375},
            "context_relevance": {"individual_scores": [0.805466, 0.410251], "score": 0.58177975},
            "faithfulness": {"individual_scores": [0.135581, 0.695974], "score": 0.40585375},
            "semantic_answer_similarity": {"individual_scores": [0.971241, 0.159320], "score": 0.53757075},
        },
    }

    _ = EvaluationRunResult("testing_pipeline_1", inputs=data["inputs"], results=data["metrics"])

    with pytest.raises(ValueError, match="No inputs provided"):
        _ = EvaluationRunResult("testing_pipeline_1", inputs={}, results={})

    with pytest.raises(ValueError, match="Lengths of the inputs should be the same"):
        _ = EvaluationRunResult(
            "testing_pipeline_1",
            inputs={"query_id": ["53c3b3e6", "something else"], "question": ["What is the capital of France?"]},
            results={"some": {"score": 0.1, "individual_scores": [0.378064, 0.534964]}},
        )

    with pytest.raises(ValueError, match="Aggregate score missing"):
        _ = EvaluationRunResult(
            "testing_pipeline_1",
            inputs={
                "query_id": ["53c3b3e6", "something else"],
                "question": ["What is the capital of France?", "another"],
            },
            results={"some": {"individual_scores": [0.378064, 0.534964]}},
        )

    with pytest.raises(ValueError, match="Individual scores missing"):
        _ = EvaluationRunResult(
            "testing_pipeline_1",
            inputs={
                "query_id": ["53c3b3e6", "something else"],
                "question": ["What is the capital of France?", "another"],
            },
            results={"some": {"score": 0.378064}},
        )

    with pytest.raises(ValueError, match="Length of individual scores .* should be the same as the inputs"):
        _ = EvaluationRunResult(
            "testing_pipeline_1",
            inputs={
                "query_id": ["53c3b3e6", "something else"],
                "question": ["What is the capital of France?", "another"],
            },
            results={"some": {"score": 0.1, "individual_scores": [0.378064, 0.534964, 0.3]}},
        )


def test_score_report():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": {
            "reciprocal_rank": {"individual_scores": [0.378064, 0.534964], "score": 0.476932},
            "single_hit": {"individual_scores": [1, 1], "score": 0.75},
            "multi_hit": {"individual_scores": [0.706125, 0.454976], "score": 0.46428375},
            "context_relevance": {"individual_scores": [0.805466, 0.410251], "score": 0.58177975},
            "faithfulness": {"individual_scores": [0.135581, 0.695974], "score": 0.40585375},
            "semantic_answer_similarity": {"individual_scores": [0.971241, 0.159320], "score": 0.53757075},
        },
    }

    result = EvaluationRunResult("testing_pipeline_1", inputs=data["inputs"], results=data["metrics"])
    report = result.aggregated_report(output_format="json")

    assert report == (
        '{"metrics": ["reciprocal_rank", "single_hit", "multi_hit", "context_relevance", "faithfulness", '
        '"semantic_answer_similarity"], "score": [0.476932, 0.75, 0.46428375, 0.58177975, 0.40585375, 0.53757075]}'
    )


def test_to_df():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7", "53c3b3e6", "225f87f7"],
            "question": [
                "What is the capital of France?",
                "What is the capital of Spain?",
                "What is the capital of Luxembourg?",
                "What is the capital of Portugal?",
            ],
            "contexts": ["wiki_France", "wiki_Spain", "wiki_Luxembourg", "wiki_Portugal"],
            "answer": ["Paris", "Madrid", "Luxembourg", "Lisbon"],
            "predicted_answer": ["Paris", "Madrid", "Luxembourg", "Lisbon"],
        },
        "metrics": {
            "reciprocal_rank": {"score": 0.1, "individual_scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            "single_hit": {"score": 0.1, "individual_scores": [1, 1, 0, 1]},
            "multi_hit": {"score": 0.1, "individual_scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            "context_relevance": {"score": 0.1, "individual_scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            "faithfulness": {"score": 0.1, "individual_scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            "semantic_answer_similarity": {"score": 0.1, "individual_scores": [0.971241, 0.159320, 0.019722, 1]},
        },
    }

    result = EvaluationRunResult("testing_pipeline_1", inputs=data["inputs"], results=data["metrics"])
    assert result.detailed_report() == (
        '{"query_id": ["53c3b3e6", "225f87f7", "53c3b3e6", "225f87f7"], '
        '"question": ["What is the capital of France?",'
        ' "What is the capital of Spain?", "What is the capital of Luxembourg?", "What is the capital of Portugal?"], '
        '"contexts": ["wiki_France", "wiki_Spain", "wiki_Luxembourg", "wiki_Portugal"], "answer": ["Paris", "Madrid", '
        '"Luxembourg", "Lisbon"], "predicted_answer": ["Paris", "Madrid", "Luxembourg", "Lisbon"], "reciprocal_rank": '
        '[0.378064, 0.534964, 0.216058, 0.778642], "single_hit": [1, 1, 0, 1], "multi_hit": [0.706125, 0.454976, '
        '0.445512, 0.250522], "context_relevance": [0.805466, 0.410251, 0.75007, 0.361332], "faithfulness": [0.135581, '
        '0.695974, 0.749861, 0.041999], "semantic_answer_similarity": [0.971241, 0.15932, 0.019722, 1.0]}'
    )


def test_comparative_individual_scores_report():
    data_1 = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": {
            "reciprocal_rank": {"individual_scores": [0.378064, 0.534964], "score": 0.476932},
            "single_hit": {"individual_scores": [1, 1], "score": 0.75},
            "multi_hit": {"individual_scores": [0.706125, 0.454976], "score": 0.46428375},
            "context_relevance": {"individual_scores": [1, 1], "score": 1},
            "faithfulness": {"individual_scores": [0.135581, 0.695974], "score": 0.40585375},
            "semantic_answer_similarity": {"individual_scores": [0.971241, 0.159320], "score": 0.53757075},
        },
    }

    data_2 = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": {
            "reciprocal_rank": {"individual_scores": [0.378064, 0.534964], "score": 0.476932},
            "single_hit": {"individual_scores": [1, 1], "score": 0.75},
            "multi_hit": {"individual_scores": [0.706125, 0.454976], "score": 0.46428375},
            "context_relevance": {"individual_scores": [1, 1], "score": 1},
            "faithfulness": {"individual_scores": [0.135581, 0.695974], "score": 0.40585375},
            "semantic_answer_similarity": {"individual_scores": [0.971241, 0.159320], "score": 0.53757075},
        },
    }

    result1 = EvaluationRunResult("testing_pipeline_1", inputs=data_1["inputs"], results=data_1["metrics"])
    result2 = EvaluationRunResult("testing_pipeline_2", inputs=data_2["inputs"], results=data_2["metrics"])
    results = result1.comparative_detailed_report(result2, keep_columns=["predicted_answer"])

    assert list(json.loads(results).keys()) == [
        "query_id",
        "question",
        "contexts",
        "answer",
        "testing_pipeline_1_predicted_answer",
        "testing_pipeline_1_reciprocal_rank",
        "testing_pipeline_1_single_hit",
        "testing_pipeline_1_multi_hit",
        "testing_pipeline_1_context_relevance",
        "testing_pipeline_1_faithfulness",
        "testing_pipeline_1_semantic_answer_similarity",
        "testing_pipeline_2_predicted_answer",
        "testing_pipeline_2_reciprocal_rank",
        "testing_pipeline_2_single_hit",
        "testing_pipeline_2_multi_hit",
        "testing_pipeline_2_context_relevance",
        "testing_pipeline_2_faithfulness",
        "testing_pipeline_2_semantic_answer_similarity",
    ]
