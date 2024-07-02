# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
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
    report = result.score_report().to_json()

    assert report == (
        '{"metrics":{"0":"reciprocal_rank","1":"single_hit","2":"multi_hit","3":"context_relevance",'
        '"4":"faithfulness","5":"semantic_answer_similarity"},'
        '"score":{"0":0.476932,"1":0.75,"2":0.46428375,"3":0.58177975,"4":0.40585375,"5":0.53757075}}'
    )


def test_to_pandas():
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
    assert result.to_pandas().to_json() == (
        '{"query_id":{"0":"53c3b3e6","1":"225f87f7","2":"53c3b3e6","3":"225f87f7"},'
        '"question":{"0":"What is the capital of France?","1":"What is the capital of Spain?",'
        '"2":"What is the capital of Luxembourg?","3":"What is the capital of Portugal?"},'
        '"contexts":{"0":"wiki_France","1":"wiki_Spain","2":"wiki_Luxembourg","3":"wiki_Portugal"},'
        '"answer":{"0":"Paris","1":"Madrid","2":"Luxembourg","3":"Lisbon"},'
        '"predicted_answer":{"0":"Paris","1":"Madrid","2":"Luxembourg","3":"Lisbon"},'
        '"reciprocal_rank":{"0":0.378064,"1":0.534964,"2":0.216058,"3":0.778642},'
        '"single_hit":{"0":1,"1":1,"2":0,"3":1},'
        '"multi_hit":{"0":0.706125,"1":0.454976,"2":0.445512,"3":0.250522},'
        '"context_relevance":{"0":0.805466,"1":0.410251,"2":0.75007,"3":0.361332},'
        '"faithfulness":{"0":0.135581,"1":0.695974,"2":0.749861,"3":0.041999},'
        '"semantic_answer_similarity":{"0":0.971241,"1":0.15932,"2":0.019722,"3":1.0}}'
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
    results = result1.comparative_individual_scores_report(result2)

    expected = {
        "query_id": {0: "53c3b3e6", 1: "225f87f7"},
        "question": {0: "What is the capital of France?", 1: "What is the capital of Spain?"},
        "contexts": {0: "wiki_France", 1: "wiki_Spain"},
        "answer": {0: "Paris", 1: "Madrid"},
        "predicted_answer": {0: "Paris", 1: "Madrid"},
        "testing_pipeline_1_reciprocal_rank": {0: 0.378064, 1: 0.534964},
        "testing_pipeline_1_single_hit": {0: 1, 1: 1},
        "testing_pipeline_1_multi_hit": {0: 0.706125, 1: 0.454976},
        "testing_pipeline_1_context_relevance": {0: 1, 1: 1},
        "testing_pipeline_1_faithfulness": {0: 0.135581, 1: 0.695974},
        "testing_pipeline_1_semantic_answer_similarity": {0: 0.971241, 1: 0.15932},
        "testing_pipeline_2_reciprocal_rank": {0: 0.378064, 1: 0.534964},
        "testing_pipeline_2_single_hit": {0: 1, 1: 1},
        "testing_pipeline_2_multi_hit": {0: 0.706125, 1: 0.454976},
        "testing_pipeline_2_context_relevance": {0: 1, 1: 1},
        "testing_pipeline_2_faithfulness": {0: 0.135581, 1: 0.695974},
        "testing_pipeline_2_semantic_answer_similarity": {0: 0.971241, 1: 0.15932},
    }

    assert expected == results.to_dict()


def test_comparative_individual_scores_report_keep_truth_answer_in_df():
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
    results = result1.comparative_individual_scores_report(result2, keep_columns=["predicted_answer"])

    assert list(results.columns) == [
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
