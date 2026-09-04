---
title: Evaluation
id: evaluation-api
description: Represents the results of evaluation.
slug: "/evaluation-api"
---

<a id="eval_run_result"></a>

# Module eval\_run\_result

<a id="eval_run_result.EvaluationRunResult"></a>

## EvaluationRunResult

Contains the inputs and the outputs of an evaluation pipeline and provides methods to inspect them.

<a id="eval_run_result.EvaluationRunResult.__init__"></a>

#### EvaluationRunResult.\_\_init\_\_

```python
def __init__(run_name: str, inputs: dict[str, list[Any]],
             results: dict[str, dict[str, Any]])
```

Initialize a new evaluation run result.

**Arguments**:

- `run_name`: Name of the evaluation run.
- `inputs`: Dictionary containing the inputs used for the run. Each key is the name of the input and its value is a list
of input values. The length of the lists should be the same.
- `results`: Dictionary containing the results of the evaluators used in the evaluation pipeline. Each key is the name
of the metric and its value is dictionary with the following keys:
- 'score': The aggregated score for the metric.
- 'individual_scores': A list of scores for each input sample.

<a id="eval_run_result.EvaluationRunResult.aggregated_report"></a>

#### EvaluationRunResult.aggregated\_report

```python
def aggregated_report(
    output_format: Literal["json", "csv", "df"] = "json",
    csv_file: Optional[str] = None
) -> Union[dict[str, list[Any]], "DataFrame", str]
```

Generates a report with aggregated scores for each metric.

**Arguments**:

- `output_format`: The output format for the report, "json", "csv", or "df", default to "json".
- `csv_file`: Filepath to save CSV output if `output_format` is "csv", must be provided.

**Returns**:

JSON or DataFrame with aggregated scores, in case the output is set to a CSV file, a message confirming the
successful write or an error message.

<a id="eval_run_result.EvaluationRunResult.detailed_report"></a>

#### EvaluationRunResult.detailed\_report

```python
def detailed_report(
    output_format: Literal["json", "csv", "df"] = "json",
    csv_file: Optional[str] = None
) -> Union[dict[str, list[Any]], "DataFrame", str]
```

Generates a report with detailed scores for each metric.

**Arguments**:

- `output_format`: The output format for the report, "json", "csv", or "df", default to "json".
- `csv_file`: Filepath to save CSV output if `output_format` is "csv", must be provided.

**Returns**:

JSON or DataFrame with the detailed scores, in case the output is set to a CSV file, a message confirming
the successful write or an error message.

<a id="eval_run_result.EvaluationRunResult.comparative_detailed_report"></a>

#### EvaluationRunResult.comparative\_detailed\_report

```python
def comparative_detailed_report(
        other: "EvaluationRunResult",
        keep_columns: Optional[list[str]] = None,
        output_format: Literal["json", "csv", "df"] = "json",
        csv_file: Optional[str] = None) -> Union[str, "DataFrame", None]
```

Generates a report with detailed scores for each metric from two evaluation runs for comparison.

**Arguments**:

- `other`: Results of another evaluation run to compare with.
- `keep_columns`: List of common column names to keep from the inputs of the evaluation runs to compare.
- `output_format`: The output format for the report, "json", "csv", or "df", default to "json".
- `csv_file`: Filepath to save CSV output if `output_format` is "csv", must be provided.

**Returns**:

JSON or DataFrame with a comparison of the detailed scores, in case the output is set to a CSV file,
a message confirming the successful write or an error message.
