from statistics import quantiles
from typing import Any, Dict

from pandas import DataFrame
from pandas import concat as pd_concat


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

        if self.pipeline_name == other.pipeline_name:
            raise ValueError("The pipelines have the same name.")

        return {
            f"{self.pipeline_name}": self.individual_aggregate_score_report(),
            f"{other.pipeline_name}": other.individual_aggregate_score_report(),
        }

    def individual_detailed_score_report(self) -> DataFrame:
        """
        Creates a DataFrame with the scores for each metric in the results.

        :return: A DataFrame with the scores for each metric.
        """
        inputs_columns = list(self.results["inputs"].keys())
        inputs_values = list(self.results["inputs"].values())
        inputs_values = list(map(list, zip(*inputs_values)))  # transpose the values
        df_inputs = DataFrame(inputs_values, columns=inputs_columns)

        scores_columns = [entry["name"] for entry in self.results["metrics"]]
        scores_values = [entry["scores"] for entry in self.results["metrics"]]
        scores_values = list(map(list, zip(*scores_values)))  # transpose the values
        df_scores = DataFrame(scores_values, columns=scores_columns)

        return df_inputs.join(df_scores)

    def comparative_detailed_score_report(self, other: "ResultsEvaluator") -> DataFrame:
        """
        Creates a DataFrame with the scores for each metric in the results of two different pipelines.

        :param other:
        :type other:
        :return:
        :rtype:
        """

        pipe_a_df = self.individual_detailed_score_report()
        pipe_b_df = other.individual_detailed_score_report()

        # check if the columns are the same except for query_id, question, context, and answer
        ignore = ["query_id", "question", "contexts", "answer"]
        columns_a = [column for column in pipe_a_df.columns if column not in ignore]
        columns_b = [column for column in pipe_b_df.columns if column not in ignore]
        if not columns_a == columns_b:
            raise ValueError("The two dataframes do not have the same columns.")

        # ToDo: check if they have the same number of rows

        # ToDo: check if query_id, question, context, and answer are the same, or some other way to figure that
        #  the evaluation comes from the same data

        # add the pipeline name to the columns
        pipe_b_df.drop(columns=ignore, inplace=True)
        pipe_b_df.columns = [f"{other.pipeline_name}_{column}" for column in pipe_b_df.columns]
        pipe_a_df.columns = [f"{self.pipeline_name}_{col}" if col not in ignore else col for col in pipe_a_df.columns]

        return pd_concat([pipe_a_df, pipe_b_df], axis=1)

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
