import logging
import warnings
from collections import defaultdict
from typing import List, Union, Optional, Dict, Literal

from dateutil.parser import parse, ParserError
from haystack.errors import NodeError

from haystack.nodes.ranker.base import BaseRanker
from haystack.schema import Document

logger = logging.getLogger(__name__)


class RecentnessRanker(BaseRanker):
    outgoing_edges = 1

    def __init__(
        self,
        date_meta_field: str,
        weight: float = 0.5,
        top_k: Optional[int] = None,
        ranking_mode: Literal["reciprocal_rank_fusion", "score"] = "reciprocal_rank_fusion",
    ):
        """
        This Node is used to rerank retrieved documents based on their age. Newer documents will rank higher.
        The importance of recentness is parametrized through the weight parameter.

        :param date_meta_field: Identifier pointing to the date field in the metadata.
                This is a required parameter, since we need dates for sorting.
        :param weight: in range [0,1].
                0 disables sorting by age.
                0.5 content and age have the same impact.
                1 means sorting only by age, most recent comes first.
        :param top_k: (optional) How many documents to return. If not provided, all documents will be returned.
                It can make sense to have large top-k values from the initial retrievers and filter docs down in the
                RecentnessRanker with this top_k parameter.
        :param ranking_mode: The mode used to combine retriever and recentness. Possible values are 'reciprocal_rank_fusion' (default) and 'score'.
                Make sure to use 'score' mode only with retrievers/rankers that give back OK score in range [0,1].
        """

        super().__init__()
        self.date_meta_field = date_meta_field
        self.weight = weight
        self.top_k = top_k
        self.ranking_mode = ranking_mode

        if self.weight < 0 or self.weight > 1:
            raise NodeError(
                """
                Param <weight> needs to be in range [0,1] but was set to '{}'.\n
                '0' disables sorting by recency, '0.5' gives equal weight to previous relevance scores and recency, and '1' ranks by recency only.\n
                Please change param <weight> when initializing the RecentnessRanker.
                """.format(
                    self.weight
                )
            )

    # pylint: disable=arguments-differ
    def predict(  # type: ignore
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Document]:
        """
        This method is used to rank a list of documents based on their age and relevance by:
        1. Adjusting the relevance score from the previous node (or, for RRF, calculating it from scratch, then adjusting) based on the chosen weight in initial parameters.
        2. Sorting the documents based on their age in the metadata, calculating the recentness score, adjusting it by weight as well.
        3. Returning top-k documents (or all, if top-k not provided) in the documents dictionary sorted by final score (relevance score + recentness score).

        :param query: Not used in practice (so can be left blank), as this ranker does not perform sorting based on semantic closeness of documents to the query.
        :param documents: Documents provided for ranking.
        :param top_k: (optional) How many documents to return at the end. If not provided, all documents will be returned, sorted by relevance and recentness (adjusted by weight).
        """

        try:
            sorted_by_date = sorted(documents, reverse=True, key=lambda x: parse(x.meta[self.date_meta_field]))
        except KeyError:
            raise NodeError(
                """
                Param <date_meta_field> was set to '{}', but document(s) {} do not contain this metadata key.\n
                Please double-check the names of existing metadata fields of your documents \n
                and set <date_meta_field> to the name of the field that contains dates.
                """.format(
                    self.date_meta_field,
                    ",".join([doc.id for doc in documents if self.date_meta_field not in doc.meta]),
                )
            )

        except ParserError:
            logger.error(
                """
                Could not parse date information for dates: %s\n
                Continuing without sorting by date.
                """,
                " - ".join([doc.meta.get(self.date_meta_field, "identifier wrong") for doc in documents]),
            )

            return documents

        # merge scores for documents sorted both by content and by date.
        # If ranking mode is set to 'reciprocal_rank_fusion', then that is used to combine previous ranking with recency ranking.
        # If ranking mode is set to 'score', then documents will be assigned a recency score in [0,1] and will be re-ranked based on both their recency score and their pre-existing relevance score.
        scores_map: Dict = defaultdict(int)
        if self.ranking_mode not in ["reciprocal_rank_fusion", "score"]:
            raise NodeError(
                """
                Param <ranking_mode> needs to be 'reciprocal_rank_fusion' or 'score' but was set to '{}'. \n
                Please change the <ranking_mode> when initializing the RecentnessRanker.
                """.format(
                    self.ranking_mode
                )
            )

        for i, doc in enumerate(documents):
            if self.ranking_mode == "reciprocal_rank_fusion":
                scores_map[doc.id] += self._calculate_rrf(rank=i) * (1 - self.weight)
            elif self.ranking_mode == "score":
                score = float(0)
                if doc.score is None:
                    warnings.warn("The score was not provided; defaulting to 0")
                elif doc.score < 0 or doc.score > 1:
                    warnings.warn(
                        "The score {} for document {} is outside the [0,1] range; defaulting to 0".format(
                            doc.score, doc.id
                        )
                    )
                else:
                    score = doc.score

                scores_map[doc.id] += score * (1 - self.weight)

        for i, doc in enumerate(sorted_by_date):
            if self.ranking_mode == "reciprocal_rank_fusion":
                scores_map[doc.id] += self._calculate_rrf(rank=i) * self.weight
            elif self.ranking_mode == "score":
                scores_map[doc.id] += self._calc_recentness_score(rank=i, amount=len(sorted_by_date)) * self.weight

        top_k = top_k or self.top_k or len(documents)

        for doc in documents:
            doc.score = scores_map[doc.id]

        return sorted(documents, key=lambda doc: doc.score if doc.score is not None else -1, reverse=True)[:top_k]

    # pylint: disable=arguments-differ
    def predict_batch(  # type: ignore
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        """
        This method is used to rank A) a list or B) a list of lists (in case the previous node is JoinDocuments) of documents based on their age and relevance.
        In case A, the predict method defined earlier is applied to the provided list.
        In case B, predict method is applied to each individual list in the list of lists provided, then the results are returned as list of lists.

        :param queries: Not used in practice (so can be left blank), as this ranker does not perform sorting based on semantic closeness of documents to the query.
        :param documents: Documents provided for ranking in a list or a list of lists.
        :param top_k: (optional) How many documents to return at the end (per list). If not provided, all documents will be returned, sorted by relevance and recentness (adjusted by weight).
        :param batch_size:  Not used in practice, so can be left blank.
        """

        if isinstance(documents[0], Document):
            return self.predict("", documents=documents, top_k=top_k)  # type: ignore
        nested_docs = []
        for docs in documents:
            results = self.predict("", documents=docs, top_k=top_k)  # type: ignore
            nested_docs.append(results)

        return nested_docs

    @staticmethod
    def _calculate_rrf(rank: int, k: int = 61) -> float:
        """
        Calculates the reciprocal rank fusion. The constant K is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper [https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf] used 1-based ranking).
        """
        return 1 / (k + rank)

    @staticmethod
    def _calc_recentness_score(rank: int, amount: int) -> float:
        """
        Calculate recentness score as a linear score between most recent and oldest document.
        This linear scaling is useful to
          a) reduce the effect of outliers and
          b) create recentness scoress that are meaningfully distributed in [0,1],
             similar to scores coming from a retriever/ranker.
        """
        return (amount - rank) / amount
