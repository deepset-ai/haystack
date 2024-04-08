- Title: Proposal for presentation of RAG evaluation results
- Decision driver: David S. Batista
- Start Date: 2024-04-03
- Proposal PR: #7462
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/7398

# Summary

Add a new component to Haystack allowing users interact with the results of evaluating the performance of a RAG model.


# Motivation

RAG models are one of them most popular use cases for Haystack. We are adding support for evaluations metrics, but there is no way to present the results of the evaluation.


# Detailed design

The output results of an evaluation pipeline composed of `evaluator` components are passed to a `RAGPipelineEvaluation`
(this is a placeholder name) which stores them internally and acts as an interface to access and present the results.

Example of the data structure that the `RAGPipelineEvaluation` class will receive for initialization:

```python

data = {
    "queries": {
        "query_id": ["53c3b3e6", "225f87f7"],
        "question": ["What is the capital of France?", "What is the capital of Spain?"],
        "contexts": ["wiki_France", "wiki_Spain"],
        "answer": ["Paris", "Madrid"]
    },
    "metrics":
        [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]}
         ]
    }
```

- At least the `query_id` or the `question and context` should be present in the data structure.
- At least one of the metrics should be present in the data structure.


The `RAGPipelineEvaluation` class provides the following methods to different types of users:

Basic users:
- `evaluation_report()`
- `comparative_evaluation_summary()`

Intermediate users:
- `detailed_evaluation_report()`
- `comparative_detailed_evaluation_report()`

Advanced users:
- `find_thresholds()`
- `find_scores_below_threshold()`


### Methods description
An evaluation report that provides a summary of the performance of the model across all queries, showing the
aggregated scores for all available metrics.

```python
def evaluation_report():
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
def get_detailed_scores(query_ids: Union[List[str], str] = "all"):
```

Example output

```bash
data  = {
    "query_id": ["53c3b3e6, 225f87f7, 8ac473ec, 97d284ca"],
    "reciprocal_rank": [0.378064, 0.534964, 0.216058, 0.778642],
    "single_hit": [1, 1, 0, 1],
    "multi_hit": [0.706125, 0.454976, 0.445512, 0.250522],
    "context_relevance": [0.805466, 0.410251, 0.750070, 0.361332],
    "faithfulness": [0.135581, 0.695974, 0.749861, 0.041999],
    "semantic_answer_similarity": [0.971241, 0.159320, 0.019722, 1],
    "aggregated_score":
      {
          'Reciprocal Rank': 0.448,
          'Single Hit': 0.5,
          'Multi Hit': 0.540,
          'Context Relevance': 0.537,
          'Faithfulness': 0.452,
          'Semantic Answer Similarity': 0.478
      }
}
```

### Comparative Evaluation Report

A comparative summary that compares the performance of the model with another model based on the aggregated scores
for all available metrics.

```python
def comparative_summary(self, other: "RAGPipelineEvaluation"):
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
def detailed_comparative_summary(self, other: "RAGPipelineEvaluation"):
```

```bash
{
    "pipeline_1": {
        "query_id": ["53c3b3e6, 225f87f7, 8ac473ec, 97d284ca"],
        "reciprocal_rank": [0.378064, 0.534964, 0.216058, 0.778642],
        "single_hit": [1, 1, 0, 1],
        "multi_hit": [0.706125, 0.454976, 0.445512, 0.250522],
        "context_relevance": [0.805466, 0.410251, 0.750070, 0.361332],
        "faithfulness": [0.135581, 0.695974, 0.749861, 0.041999],
        "semantic_answer_similarity": [0.971241, 0.159320, 0.019722, 1],
        "aggregated_score":
          {
              'Reciprocal Rank': 0.448,
              'Single Hit': 0.5,
              'Multi Hit': 0.540,
              'Context Relevance': 0.537,
              'Faithfulness': 0.452,
              'Semantic Answer Similarity': 0.478
          }
    },
    "pipeline_2": {
        "query_id": ["53c3b3e6, 225f87f7, 8ac473ec, 97d284ca"],
        "reciprocal_rank": [0.378064, 0.534964, 0.216058, 0.778642],
        "single_hit": [1, 1, 0, 1],
        "multi_hit": [0.706125, 0.454976, 0.445512, 0.250522],
        "context_relevance": [0.805466, 0.410251, 0.750070, 0.361332],
        "faithfulness": [0.135581, 0.695974, 0.749861, 0.041999],
        "semantic_answer_similarity": [0.971241, 0.159320, 0.019722, 1],
        "aggregated_score":
          {
              'Reciprocal Rank': 0.448,
              'Single Hit': 0.5,
              'Multi Hit': 0.540,
              'Context Relevance': 0.537,
              'Faithfulness': 0.452,
              'Semantic Answer Similarity': 0.478
          }
      }
}
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
def get_scores_below_threshold(self, metric: str, threshold: float):
    """Get the all the queries with a score below a certain threshold for a given metric"""  
```


# Drawbacks

- Relying on pandas DataFrame internally makes it easy to perform many of the operations.
- Nevertheless, it can be burden, since we are making `pandas` a dependency of `haystack-ai`.
- Ideally all the proposed methods should be implemented in a way that doesn't require `pandas`.


# Adoption strategy

- Doesn't introduce any breaking change, it is a new feature that can be adopted by users as they see fit for their use cases.

# How we teach this

- A tutorial would be the best approach to teach users how to use this feature.
- Adding a new entry to the documentation.

# Unresolved questions

- The `comparative_summary()` and the `comparative_detailed_summary()` methods need to be adopted to different definitions of what a correct answer is.
