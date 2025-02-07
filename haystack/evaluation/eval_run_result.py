# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import csv
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from warnings import warn

if TYPE_CHECKING:
    import pandas as pd


class EvaluationRunResult:
    """
    Contains the inputs and the outputs of an evaluation pipeline and provides methods to inspect them.

    """

    def __init__(
        self,
        run_name: str,
        inputs: Dict[str, List[Any]],
        results: Dict[str, Dict[str, Any]],
        output_format: str = "dict",
    ):
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
        :param output_format:
            The default output format for all methods ("dict", "csv", or "dataframe").
        """
        self.run_name = run_name
        self.inputs = deepcopy(inputs)
        self.results = deepcopy(results)
        self.output_format = output_format

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
            The default output format for all methods ("dict", "csv", or "dataframe").
        :param csv_file:
            Filepath to save CSV output if `output_format` is "csv".
        :returns:
            The formatted data (dict, DataFrame, or None for CSV).
        """
        if output_format:
            format_touse = output_format
        else:
            format_touse = self.output_format

        if format_touse == "dict":
            return data

        elif format_touse == "csv":
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

            if csv_file:
                with open(csv_file, mode="w", newline="") as file:
                    file.write(output.getvalue())
                print(f"Results written to {csv_file}")
                return None  # Don't return anything if written to a file
            else:
                return output.getvalue()  # Return CSV string if no file is provided

        elif format_touse == "dataframe":
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError("Pandas is required for dataframe output.") from e
            return pd.DataFrame(data)

        else:
            raise ValueError(f"Invalid output_format, choose from [dict, csv, dataframe]: {format_touse}")

    def score_report(self, csv_file: Optional[str] = None) -> Union[Dict[str, List[Any]], "pd.DataFrame", None]:
        """
        Transforms the results into a report with aggregated scores for each metric.

        :param csv_file:
            Filepath to save CSV output if `output_format` is "csv".  Can be left empty even if desired format is .csv.
        :returns:
            The report in the specified output format.
        """
        results = {k: v["score"] for k, v in self.results.items()}
        data = {"metrics": list(results.keys()), "score": list(results.values())}

        # Handle different output formats [Dict, csv, pandas.DataFrame]
        return self._handle_output(data, csv_file)

    def to_pandas(
        self, output_format: Optional[str] = None, csv_file: Optional[str] = None
    ) -> Union[Dict[str, List[Any]], "pd.DataFrame", None]:
        """
        Creates a Data Structure containing the scores of each metric for every input sample.

        :param output_format:
            Can explicitly override self.output_format
        :param csv_file:
            Filepath to save CSV output if `output_format` is "csv". Can be left empty even if desired format is .csv.
        :returns:
            Data Structure with the scores.
        """
        # Extract input columns and values (transpose the values)
        inputs_columns = list(self.inputs.keys())
        inputs_values = list(self.inputs.values())
        inputs_values = list(map(list, zip(*inputs_values)))  # Transpose rows into columns

        # Create a dictionary for input data
        inputs_data = dict(zip(inputs_columns, inputs_values))

        # Extract score columns and transpose the values
        scores_columns = list(self.results.keys())
        scores_values = [v["individual_scores"] for v in self.results.values()]
        scores_values = list(map(list, zip(*scores_values)))  # Transpose rows into columns

        # Create a dictionary for score data
        scores_data = dict(zip(scores_columns, scores_values))

        # Combine input data and score data into a single dictionary
        combined_data = {**inputs_data, **scores_data}

        # Handling different Output formats [Dict, csv, pandas.DataFrame]
        if output_format:
            return self._handle_output(combined_data, output_format, csv_file)
        else:
            return self._handle_output(combined_data, csv_file)

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

        pipe_a_dict = self.to_pandas(output_format="dict")  # Returns Dict[str, List[Any]]
        pipe_b_dict = other.to_pandas(output_format="dict")  # Returns Dict[str, List[Any]]
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
        return self._handle_output(combined_results, output_format or self.output_format, csv_file)
