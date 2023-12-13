import itertools
import logging
from collections import defaultdict
from math import inf
from typing import List, Optional
from haystack.core.component.types import Variadic

from haystack import component, Answer


logger = logging.getLogger(__name__)


@component
class AnswerJoiner:
    """
    A component that joins input lists of Answers from multiple connections and outputs them as one list.

    Example usage in a hybrid retrieval pipeline:
    ```python
    answer_store = InMemoryAnswerStore()
    p = Pipeline()
    p.add_component(instance=InMemoryBM25Retriever(answer_store=answer_store), name="bm25_retriever")
    p.add_component(
            instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
            name="text_embedder",
        )
    p.add_component(instance=InMemoryEmbeddingRetriever(answer_store=answer_store), name="embedding_retriever")
    p.add_component(instance=AnswerJoiner(), name="joiner")
    p.connect("bm25_retriever", "joiner")
    p.connect("embedding_retriever", "joiner")
    p.connect("text_embedder", "embedding_retriever")
    query = "What is the capital of France?"
    p.run(data={"bm25_retriever": {"query": query},
                "text_embedder": {"text": query}})
    ```
    """

    def __init__(self, top_k: Optional[int] = None, sort_by_score: bool = True):
        """
        Initialize the AnswerJoiner.

        :param top_k: The maximum number of Answers to be returned as output. By default, returns all Answers.
        :param sort_by_score: Whether the output list of Answers should be sorted by Answer scores in descending order.
                              By default, the output is sorted.
                              Answers without score are handled as if their score was -infinity.
        """
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(answers=List[Answer])
    def run(self, answers: Variadic[List[Answer]]):
        """
        Run the AnswerJoiner. This method joins the input lists of Answers into one output list.

        :param answers: An arbitrary number of lists of Answers to join.
        """
        output_answers = self._concatenate(answers)

        if self.sort_by_score:
            output_answers = sorted(
                output_answers,
                key=lambda answer: answer.score if hasattr(answer, "score") and answer.score is not None else -inf,
                reverse=True,
            )
            if any(not hasattr(answer, "score") or answer.score is None for answer in output_answers):
                logger.info(
                    "Some of the Answers AnswerJoiner got have score=None. It was configured to sort Answers by "
                    "score, so those with score=None were sorted as if they had a score of -infinity."
                )

        if self.top_k:
            output_answers = output_answers[: self.top_k]
        return {"answers": output_answers}

    def _concatenate(self, answer_lists):
        """
        Concatenate multiple lists of Answers and return only the Answer with the highest score for duplicate Answers.
        """
        output = []
        answers_per_data = defaultdict(list)
        for answer in itertools.chain.from_iterable(answer_lists):
            answers_per_data[answer.data].append(answer)
        for answers in answers_per_data.values():
            answers_with_best_score = max(
                answers, key=lambda answer: answer.score if hasattr(answer, "score") and answer.score else -inf
            )
            output.append(answers_with_best_score)
        return output
