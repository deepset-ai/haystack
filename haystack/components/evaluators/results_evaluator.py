from typing import Any, Dict

from pandas import DataFrame
from pandas import concat as pd_concat


class EvaluationResults:
    def __init__(self, pipeline_name: str, results: Dict[str, Any]):
        self.results = results
        self.pipeline_name = pipeline_name

    def score_report(self) -> DataFrame:
        """Transforms the results into a DataFrame with the aggregated scores for each metric."""
        results = {entry["name"]: entry["score"] for entry in self.results["metrics"]}
        return DataFrame.from_dict(results, orient="index", columns=["score"])

    def to_pandas(self) -> DataFrame:
        """
        Creates a DataFrame containing the scores for each query and each metric.

        :return: A DataFrame with the scores.
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

    def comparative_individual_scores_report(self, other: "EvaluationResults") -> DataFrame:
        """
        Creates a DataFrame with the scores for each metric in the results of two different pipelines.

        :param other: The other EvaluationResults object to compare with.
        """
        pipe_a_df = self.to_pandas()
        pipe_b_df = other.to_pandas()

        # check if the columns are the same, i.e.: the same queries and evaluation pipeline
        columns_a = list(pipe_a_df.columns)
        columns_b = list(pipe_b_df.columns)
        if not columns_a == columns_b:
            raise ValueError(f"The two evaluation results do not have the same columns: {columns_a} != {columns_b}")

        # add the pipeline name to the column
        ignore = ["query_id", "question", "contexts", "answer"]
        pipe_b_df.drop(columns=ignore, inplace=True)
        pipe_b_df.columns = [f"{other.pipeline_name}_{column}" for column in pipe_b_df.columns]
        pipe_a_df.columns = [f"{self.pipeline_name}_{col}" if col not in ignore else col for col in pipe_a_df.columns]

        results_df = pd_concat([pipe_a_df, pipe_b_df], axis=1)

        return results_df
