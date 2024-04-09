from statistics import quantiles
from typing import Any, Dict

from pandas import DataFrame


class ResultsEvaluator:
    def __init__(self, pipeline_name: str, results: Dict[str, Any]):
        self.results = results
        self.pipeline_name = pipeline_name

    def individual_aggregate_score_report(self) -> Dict[str, float]:
        """Calculate the average of the scores for each metric in the results."""
        return {entry["name"]: sum(entry["scores"]) / len(entry["scores"]) for entry in self.results["metrics"]}

    def comparative_aggregate_score_report(self, other: "ResultsEvaluator"):
        """
        Compare the average scores for each metric in the results of two different pipelines.

        :param other: The other pipeline to compare against.
        """
        return {
            f"{self.pipeline_name}": self.individual_aggregate_score_report(),
            f"{other.pipeline_name}": other.individual_aggregate_score_report(),
        }

    def individual_detailed_score_report(self) -> DataFrame:
        """Return a DataFrame with the scores for each metric in the results."""
        values = [[entry["scores"]] for entry in self.results["metrics"]["scores"]]
        columns = [entry["name"] for entry in self.results["metrics"]]
        return DataFrame(values, columns=columns)

    def comparative_detailed_score_report(self, other: "ResultsEvaluator") -> DataFrame:
        # check if the columns are the same
        if not self.individual_detailed_score_report().columns == other.individual_detailed_score_report().columns:
            raise ValueError("The two dataframes do not have the same columns.")

        # add the pipeline name to the columns
        # ToDo: except question, context, answer columns
        tmp_this = self.individual_detailed_score_report()
        tmp_other = other.individual_detailed_score_report()
        tmp_this.columns = [
            f"{self.pipeline_name}_{column}" for column in self.individual_detailed_score_report().columns
        ]
        tmp_other.columns = [
            f"{other.pipeline_name}_{column}" for column in other.individual_detailed_score_report().columns
        ]

        # merge the two dataframes
        return tmp_this.append(tmp_other)

    def find_thresholds(self, metric: str) -> Dict[str, float]:
        """
        Calculate the 25th percentile, 75th percentile, median, and average of the scores for a given metric.

        :param metric: The metric to calculate the thresholds for.
        :return: A dictionary with the thresholds.
        """

        values = self.results["metrics"][metric]
        if len(values) <= 4:
            raise Warning("The number of values is too low to calculate the thresholds.")

        thresholds = ["25th percentile", "75th percentile", "median", "average"]
        return {threshold: quantiles(values)[i] for i, threshold in enumerate(thresholds)}

    def find_inputs_below_threshold(self, metric: str, threshold: float):
        """
        Find the inputs that have a score below a given threshold for a given metric.

        :param metric: The metric to filter by.
        :param threshold: The threshold to filter by.
        :return: A list of inputs that have a score below the threshold.
        """
        return [
            self.results["inputs"][i] for i, score in enumerate(self.results["metrics"][metric]) if score < threshold
        ]
