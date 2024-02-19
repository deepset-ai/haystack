import pytest

from haystack.components.evaluators import StatisticalEvaluator, StatisticalMetric


class TestStatisticalEvaluator:
    def test_init_default(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.F1)
        assert evaluator._metric == StatisticalMetric.F1

    def test_init_with_string(self):
        evaluator = StatisticalEvaluator(metric="exact_match")
        assert evaluator._metric == StatisticalMetric.EM

    def test_to_dict(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.F1)

        expected_dict = {
            "type": "haystack.components.evaluators.statistical_evaluator.StatisticalEvaluator",
            "init_parameters": {"metric": "f1"},
        }
        assert evaluator.to_dict() == expected_dict

    def test_from_dict(self):
        evaluator = StatisticalEvaluator.from_dict(
            {
                "type": "haystack.components.evaluators.statistical_evaluator.StatisticalEvaluator",
                "init_parameters": {"metric": "f1"},
            }
        )

        assert evaluator._metric == StatisticalMetric.F1


class TestStatisticalEvaluatorF1:
    def test_run_with_empty_inputs(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.F1)
        result = evaluator.run(labels=[], predictions=[])
        assert len(result) == 1
        assert result["result"] == 0.0

    def test_run_with_different_lengths(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.F1)
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(labels=labels, predictions=predictions)

    def test_run_with_matching_predictions(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.F1)
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        result = evaluator.run(labels=labels, predictions=predictions)

        assert len(result) == 1
        assert result["result"] == 1.0

    def test_run_with_single_prediction(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.F1)

        result = evaluator.run(labels=["Source"], predictions=["Open Source"])
        assert len(result) == 1
        assert result["result"] == pytest.approx(2 / 3)

    def test_run_with_mismatched_predictions(self):
        labels = ["Source", "HaystackAI"]
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.F1)
        predictions = ["Open Source", "HaystackAI"]
        result = evaluator.run(labels=labels, predictions=predictions)
        assert len(result) == 1
        assert result["result"] == pytest.approx(5 / 6)


class TestStatisticalEvaluatorExactMatch:
    def test_run_with_empty_inputs(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.EM)
        result = evaluator.run(predictions=[], labels=[])
        assert len(result) == 1
        assert result["result"] == 0.0

    def test_run_with_different_lengths(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.EM)
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(labels=labels, predictions=predictions)

    def test_run_with_matching_predictions(self):
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.EM)
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        result = evaluator.run(labels=labels, predictions=predictions)

        assert len(result) == 1
        assert result["result"] == 1.0

    def test_run_with_single_prediction(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.EM)
        result = evaluator.run(labels=["OpenSource"], predictions=["OpenSource"])
        assert len(result) == 1
        assert result["result"] == 1.0

    def test_run_with_mismatched_predictions(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.EM)
        labels = ["Source", "HaystackAI", "LLMs"]
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        result = evaluator.run(labels=labels, predictions=predictions)
        assert len(result) == 1
        assert result["result"] == 2 / 3


class TestStatisticalEvaluatorRecallSingleHit:
    def test_run(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.RECALL_SINGLE_HIT)
        labels = ["Eiffel Tower", "Louvre Museum", "Colosseum", "Trajan's Column"]
        predictions = [
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Eiffel Tower max height is 330 meters.",
            "Louvre Museum is the world's largest art museum and a historic monument in Paris, France.",
            "The Leaning Tower of Pisa is the campanile, or freestanding bell tower, of Pisa Cathedral.",
        ]
        result = evaluator.run(labels=labels, predictions=predictions)
        assert len(result) == 1
        assert result["result"] == 2 / 4

    def test_run_with_empty_labels(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.RECALL_SINGLE_HIT)
        predictions = [
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Eiffel Tower max height is 330 meters.",
            "Louvre Museum is the world's largest art museum and a historic monument in Paris, France.",
            "The Leaning Tower of Pisa is the campanile, or freestanding bell tower, of Pisa Cathedral.",
        ]
        result = evaluator.run(labels=[], predictions=predictions)
        assert len(result) == 1
        assert result["result"] == 0.0

    def test_run_with_empty_predictions(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.RECALL_SINGLE_HIT)
        labels = ["Eiffel Tower", "Louvre Museum", "Colosseum", "Trajan's Column"]
        result = evaluator.run(labels=labels, predictions=[])
        assert len(result) == 1
        assert result["result"] == 0.0


class TestStatisticalEvaluatorRecallMultiHit:
    def test_run(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.RECALL_MULTI_HIT)
        labels = ["Eiffel Tower", "Louvre Museum", "Colosseum", "Trajan's Column"]
        predictions = [
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Eiffel Tower max height is 330 meters.",
            "Louvre Museum is the world's largest art museum and a historic monument in Paris, France.",
            "The Leaning Tower of Pisa is the campanile, or freestanding bell tower, of Pisa Cathedral.",
        ]
        result = evaluator.run(labels=labels, predictions=predictions)
        assert len(result) == 1
        assert result["result"] == 0.75

    def test_run_with_empty_labels(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.RECALL_MULTI_HIT)
        predictions = [
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Eiffel Tower max height is 330 meters.",
            "Louvre Museum is the world's largest art museum and a historic monument in Paris, France.",
            "The Leaning Tower of Pisa is the campanile, or freestanding bell tower, of Pisa Cathedral.",
        ]
        result = evaluator.run(labels=[], predictions=predictions)
        assert len(result) == 1
        assert result["result"] == 0.0

    def test_run_with_empty_predictions(self):
        evaluator = StatisticalEvaluator(metric=StatisticalMetric.RECALL_MULTI_HIT)
        labels = ["Eiffel Tower", "Louvre Museum", "Colosseum", "Trajan's Column"]
        result = evaluator.run(labels=labels, predictions=[])
        assert len(result) == 1
        assert result["result"] == 0.0
