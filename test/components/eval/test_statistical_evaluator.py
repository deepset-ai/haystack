import pytest

from haystack.components.eval import StatisticalEvaluator


class TestStatisticalEvaluator:
    def test_init_default(self):
        labels = ["label1", "label2", "label3"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1)
        assert evaluator._labels == labels
        assert evaluator._metric == StatisticalEvaluator.Metric.F1
        assert evaluator._regexes_to_ignore is None
        assert evaluator._ignore_case is False
        assert evaluator._ignore_punctuation is False
        assert evaluator._ignore_numbers is False

    def test_to_dict(self):
        labels = ["label1", "label2", "label3"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1)

        expected_dict = {
            "type": "haystack.components.eval.statistical_evaluator.StatisticalEvaluator",
            "init_parameters": {
                "labels": labels,
                "metric": "F1",
                "regexes_to_ignore": None,
                "ignore_case": False,
                "ignore_punctuation": False,
                "ignore_numbers": False,
            },
        }
        assert evaluator.to_dict() == expected_dict

    def test_from_dict(self):
        evaluator = StatisticalEvaluator.from_dict(
            {
                "type": "haystack.components.eval.statistical_evaluator.StatisticalEvaluator",
                "init_parameters": {
                    "labels": ["label1", "label2", "label3"],
                    "metric": "F1",
                    "regexes_to_ignore": None,
                    "ignore_case": False,
                    "ignore_punctuation": False,
                    "ignore_numbers": False,
                },
            }
        )

        assert evaluator._labels == ["label1", "label2", "label3"]
        assert evaluator._metric == StatisticalEvaluator.Metric.F1
        assert evaluator._regexes_to_ignore is None
        assert evaluator._ignore_case is False
        assert evaluator._ignore_punctuation is False
        assert evaluator._ignore_numbers is False


class TestStatisticalEvaluatorF1:
    def test_run_with_empty_inputs(self):
        evaluator = StatisticalEvaluator(labels=[], metric=StatisticalEvaluator.Metric.F1)
        result = evaluator.run(predictions=[])
        assert len(result) == 1
        assert result["result"] == 0.0

    def test_run_with_different_lengths(self):
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1)

        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(predictions)

    def test_run_with_matching_predictions(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1)
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        result = evaluator.run(predictions=predictions)

        assert len(result) == 1
        assert result["result"] == 1.0

    def test_run_with_single_prediction(self):
        labels = ["Source"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1)

        result = evaluator.run(predictions=["Open Source"])
        assert len(result) == 1
        assert result["result"] == pytest.approx(2 / 3)

    def test_run_with_mismatched_predictions(self):
        labels = ["Source", "HaystackAI"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1)
        predictions = ["Open Source", "HaystackAI"]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 1
        assert result["result"] == pytest.approx(5 / 6)

    def test_run_with_ignore_case(self):
        labels = ["source", "HAYSTACKAI"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1, ignore_case=True)
        predictions = ["Open Source", "HaystackAI"]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 1
        assert result["result"] == pytest.approx(5 / 6)

    def test_run_with_ignore_punctuation(self):
        labels = ["Source", "HaystackAI"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1, ignore_punctuation=True)
        predictions = ["Open Source!", "Haystack.AI"]
        result = evaluator.run(predictions=predictions)

        assert result["result"] == pytest.approx(5 / 6)

    def test_run_with_ignore_numbers(self):
        labels = ["Source", "HaystackAI"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.F1, ignore_numbers=True)
        predictions = ["Open Source123", "HaystackAI"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == pytest.approx(5 / 6)

    def test_run_with_regex_to_ignore(self):
        labels = ["Source", "HaystackAI"]
        evaluator = StatisticalEvaluator(
            labels=labels, metric=StatisticalEvaluator.Metric.F1, regexes_to_ignore=[r"\d+"]
        )
        predictions = ["Open123 Source", "HaystackAI"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == pytest.approx(5 / 6)

    def test_run_with_multiple_regex_to_ignore(self):
        labels = ["Source", "HaystackAI"]
        evaluator = StatisticalEvaluator(
            labels=labels, metric=StatisticalEvaluator.Metric.F1, regexes_to_ignore=[r"\d+", r"[^\w\s]"]
        )
        predictions = ["Open123! Source", "Haystack.AI"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == pytest.approx(5 / 6)

    def test_run_with_multiple_ignore_parameters(self):
        labels = ["Source", "HaystackAI"]
        evaluator = StatisticalEvaluator(
            labels=labels,
            metric=StatisticalEvaluator.Metric.F1,
            ignore_numbers=True,
            ignore_punctuation=True,
            ignore_case=True,
            regexes_to_ignore=[r"[^\w\s\d]+"],
        )
        predictions = ["Open%123. !$Source", "Haystack.AI##"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == pytest.approx(5 / 6)


class TestStatisticalEvaluatorExactMatch:
    def test_run_with_empty_inputs(self):
        evaluator = StatisticalEvaluator(labels=[], metric=StatisticalEvaluator.Metric.EM)
        result = evaluator.run(predictions=[])
        assert len(result) == 1
        assert result["result"] == 0.0

    def test_run_with_different_lengths(self):
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.EM)

        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(predictions)

    def test_run_with_matching_predictions(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.EM)
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        result = evaluator.run(predictions=predictions)

        assert len(result) == 1
        assert result["result"] == 1.0

    def test_run_with_single_prediction(self):
        labels = ["OpenSource"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.EM)

        result = evaluator.run(predictions=["OpenSource"])
        assert len(result) == 1
        assert result["result"] == 1.0

    def test_run_with_mismatched_predictions(self):
        labels = ["Source", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.EM)
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 1
        assert result["result"] == 2 / 3

    def test_run_with_ignore_case(self):
        labels = ["opensource", "HAYSTACKAI", "llMs"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.EM, ignore_case=True)
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 1
        assert result["result"] == 1.0

    def test_run_with_ignore_punctuation(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.EM, ignore_punctuation=True)
        predictions = ["OpenSource!", "Haystack.AI", "LLMs,"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == 1.0

    def test_run_with_ignore_numbers(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(labels=labels, metric=StatisticalEvaluator.Metric.EM, ignore_numbers=True)
        predictions = ["OpenSource123", "HaystackAI", "LLMs456"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == 1.0

    def test_run_with_regex_to_ignore(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(
            labels=labels, metric=StatisticalEvaluator.Metric.EM, regexes_to_ignore=[r"\d+"]
        )
        predictions = ["Open123Source", "HaystackAI", "LLMs456"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == 1.0

    def test_run_with_multiple_regex_to_ignore(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(
            labels=labels, metric=StatisticalEvaluator.Metric.EM, regexes_to_ignore=[r"\d+", r"\W+"]
        )
        predictions = ["Open123!Source", "Haystack.AI", "LLMs456,"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == 1.0

    def test_run_with_multiple_ignore_parameters(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(
            labels=labels,
            metric=StatisticalEvaluator.Metric.EM,
            ignore_numbers=True,
            ignore_punctuation=True,
            ignore_case=True,
            regexes_to_ignore=[r"[^\w\s\d]+"],
        )
        predictions = ["Open%123!$Source", "Haystack.AI##", "^^LLMs456,"]
        result = evaluator.run(predictions=predictions)
        assert result["result"] == 1.0
