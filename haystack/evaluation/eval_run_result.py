# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Standard library imports
import csv
import json
from copy import deepcopy

# Typing-related imports
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from warnings import warn

# Third-party imports (conditionally imported if TYPE_CHECKING is used)
if TYPE_CHECKING:
    import pandas as pd


class CustomDataFrame:
    """
    A custom class to mimic pandas.DataFrame behavior for JSON serialization.
    """

    def __init__(self, data: Dict[str, List[Any]]):
        """
        Initialize the CustomDataFrame with a dictionary of data.

        :param data:
            A dictionary where keys are column names and values are lists of column data.
        """
        self.data = data

    @property
    def columns(self) -> List[str]:
        """
        Return the column names of the CustomDataFrame.

        :returns:
            A list of column names.
        """
        return list(self.data.keys())

    def to_json(self) -> str:
        """
        Convert the CustomDataFrame to a JSON string.

        :returns:
            A JSON string representation of the data.
        """

        json_data = {k: {str(i): v[i] for i in range(len(v))} for k, v in self.data.items()}
        # Use json.dumps with separators to minimize the spaces
        return json.dumps(json_data, separators=(",", ":"))

    def to_dict(self) -> Dict[str, List[Any]]:
        """
        Convert the CustomDataFrame to a dictionary.

        :returns:
            A dictionary representation of the data.
        """
        return {k: {i: v[i] for i in range(len(v))} for k, v in self.data.items()}

    def to_pandas_dataframe(self) -> Optional["pd.DataFrame"]:
        """
        Convert the CustomDataFrame to a pandas DataFrame.

        :returns:
            A pandas DataFrame representation of the data, or None if pandas is not installed.
        """
        try:
            import pandas as pd  # Lazy importing pandas here when needed to return as pd.DataFrame

            return pd.DataFrame(self.data)
        except ImportError:
            warn("Pandas is not installed. Install pandas to use this method.")
            return None

    def to_csv_file_with_name(self, file_name: Optional[str]) -> None:
        """
        Save the CustomDataFrame to a CSV file (provide name to save).

        :param file_name:
            The name of the CSV file to save the data to.

        :returns:
             A string representation of the CSV if no file name is provided, otherwise None.
        """
        # Transpose the data to write rows
        rows = list(zip(*self.data.values()))
        header = self.data.keys()
        if file_name:
            # Write to a file if a file name is provided
            with open(file_name, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(header)  # Write header
                writer.writerows(rows)  # Write rows
            print(f"Data saved to {file_name}")
            return None
        else:
            import io  # Return CSV as a string if no file name is provided

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(header)  # Write header
            writer.writerows(rows)  # Write rows
            return output.getvalue()


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
            Dictionary containing the inputs used for the run.
            Each key is the name of the input and its value is
            a list of input values. The length of the lists should
            be the same.
        :param results:
            Dictionary containing the results of the evaluators
            used in the evaluation pipeline. Each key is the name
            of the metric and its value is dictionary with the following
            keys:
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

    def _handle_output(
        self, data: Dict[str, List[Any]], output_format: Optional[str] = None, csv_file: Optional[str] = None
    ) -> Union[Dict[str, List[Any]], None]:
        """
        Handles output formatting based on the class's `output_format`.

        :param data:
            The data to format.
        :param output_format:
            The default output format for all methods ("customdataframe", "csv", or "pandas").
        :param csv_file:
            Filepath to save CSV output if `output_format` is "csv".
        :returns:
            The formatted data (dict, DataFrame, or None for CSV).
        """
        temp_dataframe = CustomDataFrame(data)
        if output_format:
            format_touse = output_format
        else:
            return temp_dataframe  # return the CustomDataFrame type if no output format specified

        if format_touse == "customdataframe":
            return temp_dataframe

        elif format_touse == "csv":
            if csv_file:
                temp_dataframe.to_csv_file_with_name(csv_file)
            else:
                return temp_dataframe.to_csv_file_with_name(csv_file)

        elif format_touse == "pandas":
            return temp_dataframe.to_pandas_dataframe()

        else:
            raise ValueError(f"Invalid output_format, choose from [dict, csv, dataframe]: {format_touse}")

    def score_report(
        self, output_format: Optional[str] = None, csv_file: Optional[str] = None
    ) -> Union[Dict[str, List[Any]], "pd.DataFrame", "CustomDataFrame", None]:
        """
        Transforms the results into a report with aggregated scores for each metric.

        :param csv_file:
            Filepath to save CSV output if `output_format` is "csv".  Can be left empty even if desired format is .csv.
        :param output_format:
            Can explicitly override default output format i.e. 'CustomDataFrame'
        :returns:
            The report in the specified output format.
        """
        results = {k: v["score"] for k, v in self.results.items()}
        data = {"metrics": list(results.keys()), "score": list(results.values())}

        # Handle different output formats [Dict, csv, pandas.DataFrame]
        return self._handle_output(data, output_format, csv_file)

    def to_pandas(
        self, output_format: Optional[str] = None, csv_file: Optional[str] = None
    ) -> Union[Dict[str, List[Any]], "CustomDataFrame", "pd.DataFrame", None]:
        """
        Creates a Data Structure containing the scores of each metric for every input sample.

        :param output_format:
            Can explicitly override default output format i.e. 'CustomDataFrame'
        :param csv_file:
            Filepath to save CSV output if `output_format` is "csv". Can be left empty even if desired format is .csv.
        :returns:
            Data Structure with the scores.
        """
        # Extract input columns and values (transpose the values)
        # inputs_columns = list(self.inputs.keys())
        inputs_values = list(self.inputs.values())
        inputs_values = list(map(list, zip(*inputs_values)))  # Transpose rows into columns

        # Extract score columns and transpose the values
        scores_columns = list(self.results.keys())
        scores_values = [v["individual_scores"] for v in self.results.values()]
        scores_values = list(map(list, zip(*scores_values)))  # Transpose rows into columns

        # Combine input data and score data into a single dictionary
        combined_data = {}
        for col in self.inputs:
            combined_data[col] = self.inputs[col]

        # For score columns, we want to enforce type consistency.
        # self.results is a dict mapping each metric to a dict with "individual_scores"
        scores_columns = list(self.results.keys())
        for col in scores_columns:
            col_values = self.results[col]["individual_scores"]
            # If any value in the column is a float, convert all values to float.
            if any(isinstance(v, float) for v in col_values):
                col_values = [float(v) for v in col_values]
            # Otherwise (if all are integers) leave col_values as is.
            combined_data[col] = col_values

        return self._handle_output(combined_data, output_format, csv_file)

    def comparative_individual_scores_report(
        self,
        other: "EvaluationRunResult",
        keep_columns: Optional[List[str]] = None,
        csv_file: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """
        Creates a dictionary with the scores for each metric in the results of two different evaluation runs.

        The inputs to both evaluation runs are assumed to be the same.

        :param other:
            Results of another evaluation run to compare with.
        :param output_format:
            Can explicitly override default output format i.e. 'CustomDataFrame'
        :param keep_columns:
            List of common column names to keep from the inputs of the evaluation runs to compare.
        :param csv_file:
            Filepath to save CSV output if `output_format` is "csv". Cannot be left empty if desired format is .csv
        :returns:
            Data Structure with the score comparison.
        """

        if not hasattr(other, "run_name") or not hasattr(other, "inputs") or not hasattr(other, "results"):
            raise ValueError("The 'other' parameter must have 'run_name', 'inputs', and 'results' attributes.")
        if not isinstance(other, EvaluationRunResult):
            raise ValueError("Comparative scores can only be computed between EvaluationRunResults.")

        # Handle run names
        this_name = self.run_name
        other_name = other.run_name
        if this_name == other_name:
            warn(f"The run names of the two evaluation results are the same ('{this_name}')")
            this_name = f"{this_name}_first"
            other_name = f"{other_name}_second"

        # Check for input column consistency
        if self.inputs.keys() != other.inputs.keys():
            warn(f"The input columns differ between the results; using the input columns of '{this_name}'.")

        # Get data from both runs as dictionaries

        pipe_a_dict = self.to_pandas().data  # Returns Dict[str, List[Any]]
        pipe_b_dict = other.to_pandas().data  # Returns Dict[str, List[Any]]
        # Determine which columns to ignore
        if keep_columns is None:
            ignore = list(self.inputs.keys())
        else:
            ignore = [col for col in list(self.inputs.keys()) if col not in keep_columns]

        # Filter out ignored columns from pipe_b_dict
        filtered_pipe_b_dict = {f"{other_name}_{key}": value for key, value in pipe_b_dict.items() if key not in ignore}

        # Rename columns in pipe_a_dict based on ignore list
        renamed_pipe_a_dict = {
            (key if key in ignore else f"{this_name}_{key}"): value for key, value in pipe_a_dict.items()
        }

        # Combine both dictionaries into a single dictionary
        combined_results = {**renamed_pipe_a_dict, **filtered_pipe_b_dict}
        # return self._handle_output(combined_results, "csv" if csv_file else None, csv_file)
        return self._handle_output(combined_results, output_format, csv_file)
