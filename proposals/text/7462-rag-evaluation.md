- Title: Proposal for presentation of evaluation results
- Decision driver: David S. Batista
- Start Date: 2024-04-03
- Proposal PR: #7462
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/7398

# Summary

Add a new component to Haystack allowing users interact with the results of evaluating the performance of a RAG model.


# Motivation

RAG models are one of them most popular use cases for Haystack. We are adding support for evaluations metrics, but there is no way to present the results of the evaluation.


# Detailed design

The output results of an evaluation pipeline composed of `evaluator` components are passed to a `EvaluationResults`
(this is a placeholder name) which stores them internally and acts as an interface to access and present the results.

The examples below are just for illustrative purposes and are subject to change.

Example of the data structure that the `EvaluationResults` class will receive for initialization:

```python

data = {
    "inputs": {
        "query_id": ["53c3b3e6", "225f87f7"],
        "question": ["What is the capital of France?", "What is the capital of Spain?"],
        "contexts": ["wiki_France", "wiki_Spain"],
        "answer": ["Paris", "Madrid"],
        "predicted_answer": ["Paris", "Madrid"]
    },
    "metrics":
        [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]}
         ],
    },

```

The `EvaluationResults` class provides the following methods to different types of users:

Basic users:
- `individual_aggregate_score_report()`
- `comparative_aggregate_score_report()`

Intermediate users:
- `individual_detailed_score_report()`
- `comparative_detailed_score_report`

Advanced users:
- `find_thresholds()`
- `find_inputs_below_threshold()`


### Methods description
An evaluation report that provides a summary of the performance of the model across all queries, showing the
aggregated scores for all available metrics.

```python
def individual_aggregate_score_report():
```

Example output

```bash
{'Reciprocal Rank': 0.448,
 'Single Hit': 0.5,
 'Multi Hit': 0.540,
 'Context Relevance': 0.537,
 'Faithfulness': 0.452,
 'Semantic Answer Similarity': 0.478
 }
 ```

A detailed evaluation report that provides the scores of all available metrics for all queries or a subset of queries.

```python
def individual_detailed_score_report(queries: Union[List[str], str] = "all"):
```

Example output

```bash
| query_id | reciprocal_rank | single_hit | multi_hit | context_relevance | faithfulness | semantic_answer_similarity |
|----------|-----------------|------------|-----------|-------------------|-------------|----------------------------|
| 53c3b3e6 | 0.378064        | 1          | 0.706125  | 0.805466          | 0.135581    | 0.971241                   |
| 225f87f7 | 0.534964        | 1          | 0.454976  | 0.410251          | 0.695974    | 0.159320                   |
| 8ac473ec | 0.216058        | 0          | 0.445512  | 0.750070          | 0.749861    | 0.019722                   |
| 97d284ca | 0.778642        | 1          | 0.250522  | 0.361332          | 0.041999    | 1                          |
```

### Comparative Evaluation Report

A comparative summary that compares the performance of the model with another model based on the aggregated scores
for all available metrics.

```python
def comparative_aggregate_score_report(self, other: "EvaluationResults"):
```

```bash
{
    "model_1": {
        'Reciprocal Rank': 0.448,
        'Single Hit': 0.5,
        'Multi Hit': 0.540,
        'Context Relevance': 0.537,
        'Faithfulness': 0.452,
        'Semantic Answer Similarity': 0.478
    },
    "model_2": {
        'Reciprocal Rank': 0.448,
        'Single Hit': 0.5,
        'Multi Hit': 0.540,
        'Context Relevance': 0.537,
        'Faithfulness': 0.452,
        'Semantic Answer Similarity': 0.478
    }
}

```

A detailed comparative summary that compares the performance of the model with another model based on the scores of all
available metrics for all queries.


```python
def comparative_detailed_score_report(self, other: "EvaluationResults"):
```

```bash
| query_id | reciprocal_rank_model_1 | single_hit_model_1 | multi_hit_model_1 | context_relevance_model_1 | faithfulness_model_1 | semantic_answer_similarity_model_1 | reciprocal_rank_model_2 | single_hit_model_2 | multi_hit_model_2 | context_relevance_model_2 | faithfulness_model_2 | semantic_answer_similarity_model_2 |
|----------|-------------------------|--------------------|-------------------|---------------------------|----------------------|------------------------------------|-------------------------|--------------------|-------------------|---------------------------|----------------------|------------------------------------|
| 53c3b3e6 | 0.378064                | 1                  | 0.706125          | 0.805466                  | 0.135581            | 0.971241                           | 0.378064                | 1                  | 0.706125          | 0.805466                  | 0.135581            | 0.971241                           |
| 225f87f7 | 0.534964                | 1                  | 0.454976          | 0.410251                  | 0.695974            | 0.159320                           | 0.534964                | 1                  | 0.454976          | 0.410251                  | 0.695974            | 0.159320                           |
| 8ac473ec | 0.216058                | 0                  | 0.445512          | 0.750070                  | 0.749861            | 0.019722                           | 0.216058                | 0                  | 0.445512          | 0.750070                  | 0.749861            | 0.019722                           |
| 97d284ca | 0.778642                | 1                  | 0.250522          | 0.361332                  | 0.041999            | 1                                  | 0.778642                | 1                  | 0.250522          | 0.361332                  | 0.041999            | 1                                  |
```


Have a method to find interesting scores thresholds, typically used for error analysis, for all metrics available.
Some potentially interesting thresholds to find are: the 25th percentile, the 75th percentile, the mean , the median.

```python
def find_thresholds(self, metrics: List[str]) -> Dict[str, float]:
```

```bash
data  = {
    "thresholds": ["25th percentile", "75th percentile", "mean", "average"],
    "reciprocal_rank": [0.378064, 0.534964, 0.216058, 0.778642],
    "context_relevance": [0.805466, 0.410251, 0.750070, 0.361332],
    "faithfulness": [0.135581, 0.695974, 0.749861, 0.041999],
    "semantic_answer_similarity": [0.971241, 0.159320, 0.019722, 1],
}
````

Then have another method that  

```python
def find_inputs_below_threshold(self, metric: str, threshold: float):
    """Get the all the queries with a score below a certain threshold for a given metric"""  
```

# Drawbacks

- Having the output in a format table may not be flexible enough, and maybe too verbose, for datasets with a large number of queries.
- Maybe a JSON format would be better with the option to export to a .csv file.


# Adoption strategy

- Doesn't introduce any breaking change, it is a new feature that can be adopted by users as they see fit for their use cases.

# How we teach this

- A tutorial would be the best approach to teach users how to use this feature.
- Adding a new entry to the documentation.
