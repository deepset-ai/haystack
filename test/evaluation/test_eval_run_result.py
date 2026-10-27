# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import builtins
import csv
from typing import Any
from unittest.mock import patch

import pytest

from haystack.evaluation import EvaluationRunResult


def test_init_results_evaluator():
    data: dict[str, dict[str, Any]] = {
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
    data: dict[str, dict[str, Any]] = {
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
        {
            "metrics": [
                "reciprocal_rank",
                "single_hit",
                "multi_hit",
                "context_relevance",
                "faithfulness",
                "semantic_answer_similarity",
            ],
            "score": [0.476932, 0.75, 0.46428375, 0.58177975, 0.40585375, 0.53757075],
        }
    )


def test_to_df():
    data: dict[str, dict[str, Any]] = {
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
        {
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
            "reciprocal_rank": [0.378064, 0.534964, 0.216058, 0.778642],
            "single_hit": [1, 1, 0, 1],
            "multi_hit": [0.706125, 0.454976, 0.445512, 0.250522],
            "context_relevance": [0.805466, 0.410251, 0.75007, 0.361332],
            "faithfulness": [0.135581, 0.695974, 0.749861, 0.041999],
            "semantic_answer_similarity": [0.971241, 0.15932, 0.019722, 1.0],
        }
    )


def test_detailed_report_csv_writes_utf8_regardless_of_locale(tmp_path):
    inputs = {"question": ["¿Cuál es la capital de España?", "東京は日本の首都ですか?"], "answer": ["Madrid", "はい"]}
    metrics = {"exact_match": {"score": 0.5, "individual_scores": [1, 0]}}
    result = EvaluationRunResult("testing_pipeline", inputs=inputs, results=metrics)
    csv_file = tmp_path / "report.csv"

    # simulate a Windows locale: without an explicit encoding, the file would be written as cp1252
    def open_with_cp1252_default(file, mode="r", newline=None, encoding=None):
        return builtins.open(file, mode, newline=newline, encoding=encoding or "cp1252")

    with patch("haystack.evaluation.eval_run_result.open", open_with_cp1252_default, create=True):
        message = result.detailed_report(output_format="csv", csv_file=str(csv_file))

    assert message == f"Data successfully written to {csv_file}"
    with open(csv_file, newline="", encoding="utf-8") as f:
        assert list(csv.reader(f)) == [
            ["question", "answer", "exact_match"],
            ["¿Cuál es la capital de España?", "Madrid", "1"],
            ["東京は日本の首都ですか?", "はい", "0"],
        ]


def test_comparative_individual_scores_report():
    data_1: dict[str, dict[str, Any]] = {
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

    data_2: dict[str, dict[str, Any]] = {
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

    assert isinstance(results, dict)
    assert list(results.keys()) == [
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


SCORES = {"q1": 1.0, "q2": 0.5, "q3": 0.0, "q4": 0.25}


def _run(name: str, ids: list[str], questions: list[str] | None = None) -> EvaluationRunResult:
    return EvaluationRunResult(
        name,
        inputs={
            "query_id": ids,
            "question": questions or [f"Question {qid}?" for qid in ids],
            "predicted_answer": [f"{name} answer to {qid}" for qid in ids],
        },
        results={"faithfulness": {"score": 0.5, "individual_scores": [SCORES[qid] for qid in ids]}},
    )


def test_comparative_detailed_report_align_on_matches_reordered_rows():
    run_a = _run("a", ["q1", "q2", "q3"])
    run_b = _run("b", ["q3", "q1", "q2"])

    report = run_a.comparative_detailed_report(run_b, align_on="query_id")
    assert isinstance(report, dict)

    assert report["query_id"] == ["q1", "q2", "q3"]
    assert report["a_faithfulness"] == [1.0, 0.5, 0.0]
    assert report["b_faithfulness"] == [1.0, 0.5, 0.0]


def test_comparative_detailed_report_align_on_keeps_rows_missing_in_either_run(caplog):
    run_a = _run("a", ["q1", "q2", "q3"])
    run_b = _run("b", ["q4", "q1", "q2"])

    report = run_a.comparative_detailed_report(run_b, align_on="query_id")
    assert isinstance(report, dict)

    assert report["query_id"] == ["q1", "q2", "q3", "q4"]
    assert report["a_faithfulness"] == [1.0, 0.5, 0.0, None]
    assert report["b_faithfulness"] == [1.0, 0.5, None, 0.25]
    assert "1 row(s) of 'a' have no match in 'b'" in caplog.text
    assert "1 row(s) of 'b' have no match in 'a'" in caplog.text


def test_comparative_detailed_report_align_on_pairs_repeated_values_in_order(caplog):
    run_a = _run("a", ["q1", "q2", "q1"])
    run_b = _run("b", ["q1", "q1", "q2"])
    run_b.results["faithfulness"]["individual_scores"] = [0.1, 0.2, 0.3]

    report = run_a.comparative_detailed_report(run_b, align_on="query_id")
    assert isinstance(report, dict)

    assert report["query_id"] == ["q1", "q2", "q1"]
    assert report["b_faithfulness"] == [0.1, 0.3, 0.2]
    assert caplog.text == ""


def test_comparative_detailed_report_align_on_ignores_whitespace_but_not_case():
    run_a = _run("a", ["q1", "q2"], questions=["How many days?", "Who approves?"])
    run_b = _run("b", ["q2", "q1"], questions=["Who  approves? ", "how many days?"])

    report = run_a.comparative_detailed_report(run_b, align_on="question")
    assert isinstance(report, dict)

    assert report["question"] == ["How many days?", "Who approves?", "how many days?"]
    assert report["b_faithfulness"] == [None, 0.5, 1.0]


def test_comparative_detailed_report_align_on_missing_column_raises():
    run_a = _run("a", ["q1"])
    run_b = _run("b", ["q1"])

    with pytest.raises(ValueError, match="'qid' is not an input column of the evaluation run 'a'"):
        run_a.comparative_detailed_report(run_b, align_on="qid")


def test_comparative_detailed_report_align_on_keeps_kept_columns_with_their_row():
    run_a = _run("a", ["q1", "q2"])
    run_b = _run("b", ["q2", "q1"])

    report = run_a.comparative_detailed_report(run_b, keep_columns=["predicted_answer"], align_on="query_id")
    assert isinstance(report, dict)

    assert report["a_predicted_answer"] == ["a answer to q1", "a answer to q2"]
    assert report["b_predicted_answer"] == ["b answer to q1", "b answer to q2"]


def test_comparative_detailed_report_same_run_names_keeps_both_runs():
    run_a = _run("prompt", ["q1", "q2"])
    run_b = _run("prompt", ["q1", "q2"])
    run_b.results["faithfulness"]["individual_scores"] = [0.1, 0.2]

    report = run_a.comparative_detailed_report(run_b)
    assert isinstance(report, dict)

    assert report["prompt_first_faithfulness"] == [1.0, 0.5]
    assert report["prompt_second_faithfulness"] == [0.1, 0.2]


@pytest.mark.parametrize(
    "ids_a, ids_b, expected_a, expected_b, message",
    [
        (["q1", "q2", "q3"], ["q1", "q2"], [1.0, 0.5, 0.0], [1.0, 0.5, None], "'a' has 3 rows and 'b' has 2"),
        (["q1", "q2"], ["q1", "q2", "q3"], [1.0, 0.5, None], [1.0, 0.5, 0.0], "'a' has 2 rows and 'b' has 3"),
    ],
)
def test_comparative_detailed_report_warns_and_pads_when_row_counts_differ(
    caplog, ids_a, ids_b, expected_a, expected_b, message
):
    run_a = _run("a", ids_a)
    run_b = _run("b", ids_b)

    report = run_a.comparative_detailed_report(run_b)
    assert isinstance(report, dict)

    assert report["query_id"] == ["q1", "q2", "q3"]
    assert report["a_faithfulness"] == expected_a
    assert report["b_faithfulness"] == expected_b
    assert message in caplog.text
