- Title: Proposal for presentation of RAG evaluation results
- Decision driver: David S. Batista
- Start Date: 2024-04-03
- Proposal PR: (fill in after opening the PR)
- Github Issue or Discussion: (only if available, link the original request for this change)

# Summary

Add a new component to Haystack allowing users to evaluate the performance of a RAG model.

# Basic example

```python

data  = {
    "query_id": ["53c3b3e6, 225f87f7, 8ac473ec, 97d284ca"],
    "reciprocal_rank": [0.378064, 0.534964, 0.216058, 0.778642],
    "single_hit": [1, 1, 0, 1],
    "multi_hit": [0.706125, 0.454976, 0.445512, 0.250522],
    "context_relevance": [0.805466, 0.410251, 0.750070, 0.361332],
    "faithfulness": [0.135581, 0.695974, 0.749861, 0.041999],
    "semantic_answer_similarity": [0.971241, 0.159320, 0.019722, 1],
    "exact_match": [0, 0, 0, 1]
}

evaluation = RAGPipelineEvaluation(name="pipeline_1", data=data)
evaluation.classification_report()
```

```bash
{'Reciprocal Rank': 0.448,
 'Single Hit': 0.5,
 'Multi Hit': 0.540,
 'Context Relevance': 0.537,
 'Faithfulness': 0.452,
 'Semantic Answer Similarity': 0.478,
 'Exact Match': 0.442}
````

# Motivation

RAG models are one of them most popular use cases for Haystack. We are adding support for evaluations metrics, but
there is no way to present the results of the evaluation.


# Detailed design

A new class  `RAGPipelineEvaluation` that receives the results of some or all metric scores over a set of queries/questions
given to a RAG model.

Example:

```
data  = {
    "query_id": ["53c3b3e6, 225f87f7, 8ac473ec, 97d284ca"],
    "reciprocal_rank": [0.378064, 0.534964, 0.216058, 0.778642],
    "single_hit": [1, 1, 0, 1],
    "multi_hit": [0.706125, 0.454976, 0.445512, 0.250522],
    "context_relevance": [0.805466, 0.410251, 0.750070, 0.361332],
    "faithfulness": [0.135581, 0.695974, 0.749861, 0.041999],
    "semantic_answer_similarity": [0.971241, 0.159320, 0.019722, 1],
    "exact_match": [0, 0, 0, 1]
}
```

These scores are computed using features already available in Haystack, in the `evaluators` module.

`RAGPipelineEvaluation` stores internally this data as a `pd.DataFrame` and provides methods to operate on it presenting
results of the evaluation.

```python
class RAGPipelineEvaluation:

    def __init__(self, name: str, data: Union[pd.DataFrame, List[Dict[str, Union[str, float]]]]):
        self.name = name
        self.data = self._get_mocked_dataframe_single_k_value(n_queries=50)  # this is just to have numbers to show

    @staticmethod
    def _get_mocked_dataframe_single_k_value(n_queries: int):
        """
        Generate a mocked dataframe for evaluation purposes.

        - Reciprocal Rank: 1 / rank of the first correct answer - range [0, 1]
        - Single Hit: 1 if the first retrieved document is correct, 0 otherwise - binary
        - Multi Hit: proportion of correct documents in the top k retrieved documents - range [0,1]

        - Context Relevance:
           for a given query q:
            - the system first retrieves some context c(q) and then generates an answer a(q)
            - the context relevance is the number of extracted sentences / number of sentences in the context c(q)
            - [0,1]

        - Faithfulness:
            - we say that the answer as(q) is faithful to the context c(q) if the claims that are made in the answer
              can be inferred from the context.
            - |V| number of statements that were supported according to the LLM
            - |S| is the total number of statements.
            - Faithfulness = |V| / |S|
            - [0,1]

        - Semantic Answer Similarity: cosine similarity between the generated answer and the correct answer - range [0,1]
        - Exact Match: 1 if the generated answer is exactly the same as the correct answer, 0 otherwise - binary
        """

        columns = ['query_id', 'reciprocal rank', 'single hit', 'multi hit', 'context relevance', 'faithfulness',
                   'semantic answer similarity', 'exact match']

        query_id = [str(uuid.uuid4()) for _ in range(n_queries)]
        reciprocal_rank = [random() for _ in range(n_queries)]
        single_hit = [randint(0, 1) for _ in range(n_queries)]
        multi_hit = [random() for _ in range(n_queries)]
        context_relevance = [random() for _ in range(n_queries)]
        faithfulness = [random() for _ in range(n_queries)]
        semantic_similarity = [random() for _ in range(n_queries)]
        exact_match = [randint(0, 1) for _ in range(n_queries)]

        values = list(
            zip(query_id, reciprocal_rank, single_hit, multi_hit, context_relevance,
                faithfulness, semantic_similarity, exact_match)
        )

        return pd.DataFrame(values, columns=columns)

    def evaluation_report(self) -> Dict[str, float]:
        """Get the classification report for the different metrics"""

        mrr = self.get_aggregated_scores('reciprocal rank')
        single_hit = self.get_aggregated_scores('single hit')
        multi_hit = self.get_aggregated_scores('multi hit')
        faithfulness = self.get_aggregated_scores('faithfulness')
        context_relevance = self.get_aggregated_scores('context relevance')
        semantic_similarity = self.get_aggregated_scores('semantic answer similarity')
        exact_match = self.get_aggregated_scores('exact match')
        correct_queries = self.data[self.data['exact match'] == 1].shape[0]

        return {
            'Reciprocal Rank': mrr,
            'Single Hit': single_hit,
            'Multi Hit': multi_hit,
            'Context Relevance': context_relevance,
            'Faithfulness': faithfulness,
            'Semantic Answer Similarity': semantic_similarity,
            'Exact Match': exact_match,
            'nr_correct_queries': correct_queries,
            'nr_incorrect_queries': self.data.shape[0] - correct_queries,
        }

    def get_aggregated_scores(self, metric: str) -> float:
        if metric in ['reciprocal rank', 'multi hit', 'context relevance', 'semantic answer similarity']:
            return self.data[metric].mean()
        if metric in ['single hit', 'exact match']:
            return self.data[metric].sum() / len(self.data)

    def get_detailed_scores(self, metric: str, query_ids: List[str]) -> pd.DataFrame:
        """Get the detailed scores for all queries or a for a subset of the queries for a given metric"""
        pass

    def find_thresholds(self, metrics: List[str]) -> Dict[str, float]:
        """
        Use the `statistics` module to find the thresholds for the different metrics.

        Some potentially interesting thresholds to find:
            - the 25th percentile
            - the 75th percentile
            - the mean
            - the median
        """
        pass

    def get_scores_below_threshold(self, metric: str, threshold: float):
        """Get the all the queries with a score below a certain threshold for a given metric"""
        return self.data[self.data[metric] < threshold]

    def comparative_detailed_summary(self, other: "PipelineEvaluationAPI") -> pd.DataFrame:
        """
        - Queries that are answered correctly by both pipelines
        - Queries that are answered incorrectly by both pipelines
        - Queries that are answered correctly by only one pipeline
        """

        # correct by both pipelines
        both_correct = self.data[(self.data['exact match'] == 1) & (other.data['exact match'] == 1)]['query_id'].tolist()

        # incorrectly by both pipelines
        both_incorrect = self.data[(self.data['exact match'] == 0) & (other.data['exact match'] == 0)]['query_id'].tolist()

        # queries that are answered correctly by only one pipeline
        only_this_correct = self.data[(self.data['exact match'] == 1) & (other.data['exact match'] == 0)]['query_id'].tolist()
        only_other_correct = self.data[(self.data['exact match'] == 0) & (other.data['exact match'] == 1)]['query_id'].tolist()

        columns = ["both_correct", "both_incorrect", f"only_{self.name}_correct", f"only_{other.name}_correct"]

        # make all lists the same length, fill with None, so that we can create a DataFrame
        max_len = max(len(both_correct), len(both_incorrect))
        both_correct += ["None"] * (max_len - len(both_correct))
        both_incorrect += ["None"] * (max_len - len(both_incorrect))
        only_this_correct += ["None"] * (max_len - len(only_this_correct))
        only_other_correct += ["None"] * (max_len - len(only_other_correct))

        values = list(zip(both_correct, both_incorrect, only_this_correct, only_other_correct))

        return pd.DataFrame(values, columns=columns)

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
