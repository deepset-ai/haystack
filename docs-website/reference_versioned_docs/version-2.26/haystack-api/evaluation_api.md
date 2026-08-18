---
title: "Evaluation"
id: evaluation-api
description: "Represents the results of evaluation."
slug: "/evaluation-api"
---


## eval_run_result

### EvaluationRunResult

Contains the inputs and the outputs of an evaluation pipeline and provides methods to inspect them.

#### __init__

```python
__init__(
    run_name: str,
    inputs: dict[str, list[Any]],
    results: dict[str, dict[str, Any]],
)
```

Initialize a new evaluation run result.

**Parameters:**

- **run_name** (<code>str</code>) – Name of the evaluation run.
- **inputs** (<code>dict\[str, list\[Any\]\]</code>) – Dictionary containing the inputs used for the run. Each key is the name of the input and its value is a list
  of input values. The length of the lists should be the same.
- **results** (<code>dict\[str, dict\[str, Any\]\]</code>) – Dictionary containing the results of the evaluators used in the evaluation pipeline. Each key is the name
  of the metric and its value is dictionary with the following keys:
  - 'score': The aggregated score for the metric.
  - 'individual_scores': A list of scores for each input sample.

#### aggregated_report

```python
aggregated_report(
    output_format: Literal["json", "csv", "df"] = "json",
    csv_file: str | None = None,
) -> Union[dict[str, list[Any]], DataFrame, str]
```

Generates a report with aggregated scores for each metric.

**Parameters:**

- **output_format** (<code>Literal['json', 'csv', 'df']</code>) – The output format for the report, "json", "csv", or "df", default to "json".
- **csv_file** (<code>str | None</code>) – Filepath to save CSV output if `output_format` is "csv", must be provided.

**Returns:**

- <code>Union\[dict\[str, list\[Any\]\], DataFrame, str\]</code> – JSON or DataFrame with aggregated scores, in case the output is set to a CSV file, a message confirming the
  successful write or an error message.

#### detailed_report

```python
detailed_report(
    output_format: Literal["json", "csv", "df"] = "json",
    csv_file: str | None = None,
) -> Union[dict[str, list[Any]], DataFrame, str]
```

Generates a report with detailed scores for each metric.

**Parameters:**

- **output_format** (<code>Literal['json', 'csv', 'df']</code>) – The output format for the report, "json", "csv", or "df", default to "json".
- **csv_file** (<code>str | None</code>) – Filepath to save CSV output if `output_format` is "csv", must be provided.

**Returns:**

- <code>Union\[dict\[str, list\[Any\]\], DataFrame, str\]</code> – JSON or DataFrame with the detailed scores, in case the output is set to a CSV file, a message confirming
  the successful write or an error message.

#### comparative_detailed_report

```python
comparative_detailed_report(
    other: EvaluationRunResult,
    keep_columns: list[str] | None = None,
    output_format: Literal["json", "csv", "df"] = "json",
    csv_file: str | None = None,
) -> Union[str, DataFrame, None]
```

Generates a report with detailed scores for each metric from two evaluation runs for comparison.

**Parameters:**

- **other** (<code>EvaluationRunResult</code>) – Results of another evaluation run to compare with.
- **keep_columns** (<code>list\[str\] | None</code>) – List of common column names to keep from the inputs of the evaluation runs to compare.
- **output_format** (<code>Literal['json', 'csv', 'df']</code>) – The output format for the report, "json", "csv", or "df", default to "json".
- **csv_file** (<code>str | None</code>) – Filepath to save CSV output if `output_format` is "csv", must be provided.

**Returns:**

- <code>Union\[str, DataFrame, None\]</code> – JSON or DataFrame with a comparison of the detailed scores, in case the output is set to a CSV file,
  a message confirming the successful write or an error message.

**Raises:**

- <code>TypeError</code> – If `other` is not an EvaluationRunResult instance, or if the detailed reports are not
  dictionaries.
- <code>ValueError</code> – If the `other` parameter is missing required attributes.
