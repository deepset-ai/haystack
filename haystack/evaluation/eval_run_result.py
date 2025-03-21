# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import csv
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pandas'") as pandas_import:
    from pandas import DataFrame

logger = logging.getLogger(__name__)


class EvaluationRunResult:
    """
    Contains the inputs and the outputs of an evaluation pipeline and provides methods to inspect them.
    """

    def __init__(self, run_name: str, inputs: Dict[str, List[Any]], results: Dict[str, Dict[str, Any]]):
        """
        Initialize a new evaluation run result.

        :param run_name:
            Name of the evaluation run.

        :param inputs:
            Dictionary containing the inputs used for the run. Each key is the name of the input and its value is a list
            of input values. The length of the lists should be the same.

        :param results:
            Dictionary containing the results of the evaluators used in the evaluation pipeline. Each key is the name
            of the metric and its value is dictionary with the following keys:
                - 'score': The aggregated score for the metric.
                - 'individual_scores': A list of scores for each input sample.
        """
        self.run_name = run_name
        self.inputs = deepcopy(inputs)
        self.results = deepcopy(results)

        if len(inputs) == 0:
            raise ValueError("No inputs provided.")
        if len({len(l) for l in inputs.values()}) != 1:
            raise ValueError("Lengths of the inputs should be the same.")

        expected_len = len(next(iter(inputs.values())))

        for metric, outputs in results.items():
            if "score" not in outputs:
                raise ValueError(f"Aggregate score missing for {metric}.")
            if "individual_scores" not in outputs:
                raise ValueError(f"Individual scores missing for {metric}.")

            if len(outputs["individual_scores"]) != expected_len:
                raise ValueError(
                    f"Length of individual scores for '{metric}' should be the same as the inputs. "
                    f"Got {len(outputs['individual_scores'])} but expected {expected_len}."
                )

    @staticmethod
    def _write_to_csv(csv_file: str, data: Dict[str, List[Any]]) -> str:
        """
        Write data to a CSV file.

        :param csv_file: Path to the CSV file to write
        :param data: Dictionary containing the data to write
        :return: Status message indicating success or failure
        """
        list_lengths = [len(value) for value in data.values()]

        if len(set(list_lengths)) != 1:
            raise ValueError("All lists in the JSON must have the same length")

        try:
            headers = list(data.keys())
            num_rows = list_lengths[0]
            rows = []

            for i in range(num_rows):
                row = [data[header][i] for header in headers]
                rows.append(row)

            with open(csv_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(rows)

            return f"Data successfully written to {csv_file}"
        except PermissionError:
            return f"Error: Permission denied when writing to {csv_file}"
        except IOError as e:
            return f"Error writing to {csv_file}: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _handle_output(
        data: Dict[str, List[Any]], output_format: Literal["json", "csv", "df"] = "csv", csv_file: Optional[str] = None
    ) -> Union[str, "DataFrame", Dict[str, List[Any]]]:
        """
        Handles output formatting based on `output_format`.

        :returns: DataFrame for 'df', dict for 'json', or confirmation message for 'csv'
        """
        if output_format == "json":
            return data

        elif output_format == "df":
            pandas_import.check()
            return DataFrame(data)

        elif output_format == "csv":
            if not csv_file:
                raise ValueError("A file path must be provided in 'csv_file' parameter to save the CSV output.")
            return EvaluationRunResult._write_to_csv(csv_file, data)

        else:
            raise ValueError(f"Invalid output format '{output_format}' provided. Choose from 'json', 'csv', or 'df'.")

    def aggregated_report(
        self, output_format: Literal["json", "csv", "df"] = "json", csv_file: Optional[str] = None
    ) -> Union[Dict[str, List[Any]], "DataFrame", str]:
        """
        Generates a report with aggregated scores for each metric.

        :param output_format: The output format for the report, "json", "csv", or "df", default to "json".
        :param csv_file: Filepath to save CSV output if `output_format` is "csv", must be provided.

        :returns:
            JSON or DataFrame with aggregated scores, in case the output is set to a CSV file, a message confirming the
            successful write or an error message.
        """
        results = {k: v["score"] for k, v in self.results.items()}
        data = {"metrics": list(results.keys()), "score": list(results.values())}
        return self._handle_output(data, output_format, csv_file)

    def detailed_report(
        self, output_format: Literal["json", "csv", "df"] = "json", csv_file: Optional[str] = None
    ) -> Union[Dict[str, List[Any]], "DataFrame", str]:
        """
        Generates a report with detailed scores for each metric.

        :param output_format: The output format for the report, "json", "csv", or "df", default to "json".
        :param csv_file: Filepath to save CSV output if `output_format` is "csv", must be provided.

        :returns:
            JSON or DataFrame with the detailed scores, in case the output is set to a CSV file, a message confirming
            the successful write or an error message.
        """

        combined_data = {col: self.inputs[col] for col in self.inputs}

        # enforce columns type consistency
        scores_columns = list(self.results.keys())
        for col in scores_columns:
            col_values = self.results[col]["individual_scores"]
            if any(isinstance(v, float) for v in col_values):
                col_values = [float(v) for v in col_values]
            combined_data[col] = col_values

        return self._handle_output(combined_data, output_format, csv_file)

    def comparative_detailed_report(
        self,
        other: "EvaluationRunResult",
        keep_columns: Optional[List[str]] = None,
        output_format: Literal["json", "csv", "df"] = "json",
        csv_file: Optional[str] = None,
    ) -> Union[str, "DataFrame", None]:
        """
        Generates a report with detailed scores for each metric from two evaluation runs for comparison.

        :param other: Results of another evaluation run to compare with.
        :param keep_columns: List of common column names to keep from the inputs of the evaluation runs to compare.
        :param output_format: The output format for the report, "json", "csv", or "df", default to "json".
        :param csv_file: Filepath to save CSV output if `output_format` is "csv", must be provided.

        :returns:
            JSON or DataFrame with a comparison of the detailed scores, in case the output is set to a CSV file,
             a message confirming the successful write or an error message.
        """

        if not isinstance(other, EvaluationRunResult):
            raise ValueError("Comparative scores can only be computed between EvaluationRunResults.")

        if not hasattr(other, "run_name") or not hasattr(other, "inputs") or not hasattr(other, "results"):
            raise ValueError("The 'other' parameter must have 'run_name', 'inputs', and 'results' attributes.")

        if self.run_name == other.run_name:
            logger.warning(
                "The run names of the two evaluation results are the same ('{run_name}')", run_name=self.run_name
            )

        if self.inputs.keys() != other.inputs.keys():
            logger.warning(
                "The input columns differ between the results; using the input columns of '{run_name}'",
                run_name=self.run_name,
            )

        # got both detailed reports
        detailed_a = self.detailed_report(output_format="json")
        detailed_b = other.detailed_report(output_format="json")

        # ensure both detailed reports are in dictionaries format
        if not isinstance(detailed_a, dict) or not isinstance(detailed_b, dict):
            raise ValueError("Detailed reports must be dictionaries.")

        # determine which columns to ignore
        if keep_columns is None:
            ignore = list(self.inputs.keys())
        else:
            ignore = [col for col in list(self.inputs.keys()) if col not in keep_columns]

        # filter out ignored columns from pipe_b_dict
        filtered_detailed_b = {
            f"{other.run_name}_{key}": value for key, value in detailed_b.items() if key not in ignore
        }

        # rename columns in pipe_a_dict based on ignore list
        renamed_detailed_a = {
            (key if key in ignore else f"{self.run_name}_{key}"): value for key, value in detailed_a.items()
        }

        # combine both detailed reports
        combined_results = {**renamed_detailed_a, **filtered_detailed_b}
        return self._handle_output(combined_results, output_format, csv_file)
