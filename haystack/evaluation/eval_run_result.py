# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import csv
from collections import Counter
from copy import deepcopy
from typing import Any, Literal, Union

from haystack import logging
from haystack.lazy_imports import LazyImport
from haystack.utils.misc import _make_metadata_value_hashable

with LazyImport("Run 'pip install pandas'") as pandas_import:
    from pandas import DataFrame

logger = logging.getLogger(__name__)


def _row_keys(values: list[Any]) -> list[tuple[Any, int]]:
    """
    Build a key for each row from its value and how many times that value appeared before.

    Strings are compared with whitespace collapsed but case kept, since IDs can be case-sensitive. The occurrence
    number pairs repeated values in the order they appear.
    """
    seen: Counter = Counter()
    keys = []
    for value in values:
        key = " ".join(value.split()) if isinstance(value, str) else _make_metadata_value_hashable(value)
        keys.append((key, seen[key]))
        seen[key] += 1
    return keys


class EvaluationRunResult:
    """
    Contains the inputs and the outputs of an evaluation pipeline and provides methods to inspect them.
    """

    def __init__(self, run_name: str, inputs: dict[str, list[Any]], results: dict[str, dict[str, Any]]) -> None:
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
        if len({len(lst) for lst in inputs.values()}) != 1:
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
    def _write_to_csv(csv_file: str, data: dict[str, list[Any]]) -> str:
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

            with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(rows)

            return f"Data successfully written to {csv_file}"
        except PermissionError:
            return f"Error: Permission denied when writing to {csv_file}"
        except OSError as e:
            return f"Error writing to {csv_file}: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _handle_output(
        data: dict[str, list[Any]], output_format: Literal["json", "csv", "df"] = "csv", csv_file: str | None = None
    ) -> Union[str, "DataFrame", dict[str, list[Any]]]:
        """
        Handles output formatting based on `output_format`.

        :returns: DataFrame for 'df', dict for 'json', or confirmation message for 'csv'
        """
        if output_format == "json":
            return data

        if output_format == "df":
            pandas_import.check()
            return DataFrame(data)

        if output_format == "csv":
            if not csv_file:
                raise ValueError("A file path must be provided in 'csv_file' parameter to save the CSV output.")
            return EvaluationRunResult._write_to_csv(csv_file, data)

        raise ValueError(f"Invalid output format '{output_format}' provided. Choose from 'json', 'csv', or 'df'.")

    def aggregated_report(
        self, output_format: Literal["json", "csv", "df"] = "json", csv_file: str | None = None
    ) -> Union[dict[str, list[Any]], "DataFrame", str]:
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
        self, output_format: Literal["json", "csv", "df"] = "json", csv_file: str | None = None
    ) -> Union[dict[str, list[Any]], "DataFrame", str]:
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
        keep_columns: list[str] | None = None,
        output_format: Literal["json", "csv", "df"] = "json",
        csv_file: str | None = None,
        *,
        align_on: str | None = None,
    ) -> Union[str, "DataFrame", None]:
        """
        Generates a report with detailed scores for each metric from two evaluation runs for comparison.

        :param other: Results of another evaluation run to compare with.
        :param keep_columns: List of common column names to keep from the inputs of the evaluation runs to compare.
        :param output_format: The output format for the report, "json", "csv", or "df", default to "json".
        :param csv_file: Filepath to save CSV output if `output_format` is "csv", must be provided.
        :param align_on: Name of an input column that identifies each row, such as a question ID or the question
            itself. If set, rows of the two runs are matched by this column instead of by position, so runs that
            evaluated the same questions in a different order are compared correctly. Strings are matched with
            whitespace collapsed but case kept. Repeated values are paired in the order they appear. Rows without a
            match are kept, with `None` for the other run, and rows only in `other` are added at the end. If not
            set, rows are matched by position and a warning is logged when they don't seem to line up.

        :returns:
            JSON or DataFrame with a comparison of the detailed scores, in case the output is set to a CSV file,
             a message confirming the successful write or an error message.
        :raises TypeError: If `other` is not an EvaluationRunResult instance, or if the detailed reports are not
            dictionaries.
        :raises ValueError: If the `other` parameter is missing required attributes, or if `align_on` is not an input
            column of both runs.
        """

        if not isinstance(other, EvaluationRunResult):
            raise TypeError("Comparative scores can only be computed between EvaluationRunResults.")

        if not hasattr(other, "run_name") or not hasattr(other, "inputs") or not hasattr(other, "results"):
            raise ValueError("The 'other' parameter must have 'run_name', 'inputs', and 'results' attributes.")

        name_a, name_b = self.run_name, other.run_name
        if name_a == name_b:
            # Keep both runs' columns instead of letting the second run overwrite the first
            name_a, name_b = f"{self.run_name}_first", f"{other.run_name}_second"
            logger.warning(
                "The run names of the two evaluation results are the same ('{run_name}'); "
                "their columns are named '{name_a}' and '{name_b}'",
                run_name=self.run_name,
                name_a=name_a,
                name_b=name_b,
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
            raise TypeError("Detailed reports must be dictionaries.")

        # Pairs of (row in self, row in other); None where a row has no counterpart in the other run
        rows = self._aligned_rows(other, align_on) if align_on is not None else self._positional_rows(other)

        # determine which columns to ignore
        if keep_columns is None:
            ignore = list(self.inputs.keys())
        else:
            ignore = [col for col in list(self.inputs.keys()) if col not in keep_columns]

        combined_results: dict[str, list[Any]] = {}
        for key, values in detailed_a.items():
            if key in ignore:
                # Shared input columns are shown once; rows only in the other run take its values
                values_b = detailed_b.get(key)
                combined_results[key] = [
                    values[i] if i is not None else (values_b[j] if values_b is not None and j is not None else None)
                    for i, j in rows
                ]
            else:
                combined_results[f"{name_a}_{key}"] = [values[i] if i is not None else None for i, _ in rows]
        for key, values in detailed_b.items():
            if key not in ignore:
                combined_results[f"{name_b}_{key}"] = [values[j] if j is not None else None for _, j in rows]

        return self._handle_output(combined_results, output_format, csv_file)

    def _positional_rows(self, other: "EvaluationRunResult") -> list[tuple[int | None, int | None]]:
        """
        Pair rows by position and warn if the two runs have a different number of rows.
        """
        len_a = len(next(iter(self.inputs.values())))
        len_b = len(next(iter(other.inputs.values())))
        if len_a != len_b:
            logger.warning(
                "'{run_a}' has {len_a} rows and '{run_b}' has {len_b}, so scores could be compared with the wrong "
                "rows. Pass `align_on` with a column that identifies each row to match them.",
                run_a=self.run_name,
                run_b=other.run_name,
                len_a=len_a,
                len_b=len_b,
            )
        rows: list[tuple[int | None, int | None]] = [(i, i) for i in range(min(len_a, len_b))]
        rows.extend((i, None) for i in range(len_b, len_a))
        rows.extend((None, j) for j in range(len_a, len_b))
        return rows

    def _aligned_rows(self, other: "EvaluationRunResult", align_on: str) -> list[tuple[int | None, int | None]]:
        """
        Pair rows that have the same value in the `align_on` column.
        """
        for run in (self, other):
            if align_on not in run.inputs:
                raise ValueError(f"'{align_on}' is not an input column of the evaluation run '{run.run_name}'.")
        try:
            keys_a = _row_keys(self.inputs[align_on])
            keys_b = _row_keys(other.inputs[align_on])
            position_b = {key: j for j, key in enumerate(keys_b)}
            keys_a_set = set(keys_a)
        except TypeError as error:
            raise ValueError(
                f"The values of the '{align_on}' column can't be used to match rows. Use a column with strings, "
                "numbers, or lists and dictionaries of them."
            ) from error

        rows: list[tuple[int | None, int | None]] = [(i, position_b.get(key)) for i, key in enumerate(keys_a)]
        only_b = [j for j, key in enumerate(keys_b) if key not in keys_a_set]
        rows.extend((None, j) for j in only_b)

        missing_in_b = sum(1 for _, j in rows if j is None)
        if missing_in_b:
            logger.warning(
                "{count} row(s) of '{run_a}' have no match in '{run_b}' on '{align_on}'",
                count=missing_in_b,
                run_a=self.run_name,
                run_b=other.run_name,
                align_on=align_on,
            )
        if only_b:
            logger.warning(
                "{count} row(s) of '{run_b}' have no match in '{run_a}' on '{align_on}' and are added at the end",
                count=len(only_b),
                run_a=self.run_name,
                run_b=other.run_name,
                align_on=align_on,
            )
        return rows
