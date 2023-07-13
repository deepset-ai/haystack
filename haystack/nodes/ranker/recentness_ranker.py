import logging
import warnings
from collections import defaultdict
from typing import List, Union, Optional, Tuple, Dict, Literal

from dateutil.parser import parse, ParserError

from haystack.nodes.ranker.base import BaseRanker
from haystack.schema import Document

logger = logging.getLogger(__name__)


class RecentnessRanker(BaseRanker):
    outgoing_edges = 1

    def __init__(
        self,
        date_identifier: str,
        weight: float = 0.5,
        top_k: Optional[int] = None,
        method: Literal["reciprocal_rank_fusion", "score"] = "reciprocal_rank_fusion",
    ):
        """
        This Node is used to rerank retrieved documents based on their age. Newer documents will rank higher.
        The importance of recentness is parametrized through the weight parameter.

        :param date_identifier: Identifier pointing to the date field in the metadata.
                This is a required parameter, since we need dates for sorting.
        :param weight: in range [0,1].
                0 disables sorting by age.
                0.5 content and age have the same impact
                1 means sorting only by age, most recent comes first.
        :param top_k: (optional) How many documents to return.
                It can make sense to have large top-k values from the initial retrievers and filter docs down in the
                RecentnessRanker with this top_k parameter.
        :param method: Method used to combine retriever and recentness. Possible values are 'reciprocal_rank_fusion' (default) and 'score'
                Make sure to use 'score' method only with retrievers/rankers that give back OK score in range [0,1]
        """

        super().__init__()
        self.date_identifier = date_identifier
        self.weight = weight
        self.top_k = top_k
        self.method = method

    # pylint: disable=arguments-differ
    def predict(  # type: ignore
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Document]:
        # sort documents based on age, newest comes first
        try:
            sorted_by_date = sorted(documents, reverse=True, key=lambda x: parse(x.meta[self.date_identifier]))
        except KeyError:
            logger.error(
                """
                Param <date_identifier> seems wrong.\n
                You supplied: '%s'.\n
                Document[0] contains metadata with keys: %s.\n
                Continuing without sorting by date.
                """,
                self.date_identifier,
                ",".join(list(documents[0].meta.keys())),
            )

            return documents
        except ParserError:
            logger.error(
                """
                Could not parse date information for dates: %s\n
                Continuing without sorting by date.
                """,
                " - ".join([x.meta.get(self.date_identifier, "identifier wrong") for x in documents]),
            )

            return documents

        # merge scores for both documents sorted by content and date.
        # If method is set to 'reciprocal_rank_fusion', then that is used to combine previous ranking with recency ranking.
        # If method is set to 'score', then documents will be assigned a recency score in [0,1] and will be re-ranked based on both their recency score and their pre-existing relevance score.
        scores_map: Dict = defaultdict(int)
        document_map = {doc.id: doc for doc in documents}
        weight = self.weight
        for i, doc in enumerate(documents):
            if self.method == "reciprocal_rank_fusion":
                scores_map[doc.id] += self._calculate_rrf(rank=i) * (1 - weight)
            elif self.method == "score":
                score = float(0)
                if doc.score is None:
                    warnings.warn("The score was not provided; defaulting to 0")
                elif doc.score < 0 or doc.score > 1:
                    warnings.warn("The score is outside the [0,1] range; defaulting to 0")
                else:
                    score = doc.score

                scores_map[doc.id] += score * (1 - weight)
            else:
                logger.error(
                    """
                    Param <method> seems wrong.\n
                    You supplied: '%s'.\n
                    It should be 'reciprocal_rank_fusion' or 'score'
                    """,
                    self.method,
                )
        for i, doc in enumerate(sorted_by_date):
            if self.method == "reciprocal_rank_fusion":
                scores_map[doc.id] += self._calculate_rrf(rank=i) * weight
            elif self.method == "score":
                scores_map[doc.id] += self._calc_recentness_score(rank=i, amount=len(sorted_by_date)) * weight
        sorted_doc_ids = sorted(scores_map.items(), key=lambda d: d[1] if d[1] is not None else -1, reverse=True)

        top_k = top_k or self.top_k or len(sorted_doc_ids)
        docs = []
        for idx, score in sorted_doc_ids[:top_k]:
            doc = document_map[idx]
            doc.score = score
            docs.append(doc)

        return docs

    # pylint: disable=arguments-differ
    def predict_batch(  # type: ignore
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        if isinstance(documents[0], Document):
            return self.run("", documents=documents, top_k=top_k)  # type: ignore
        nested_docs = []
        for docs in documents:
            temp = self.run("", documents=docs, top_k=top_k)  # type: ignore
            nested_docs.append(temp[0]["documents"])

        return nested_docs

    def _calculate_rrf(self, rank: int, k: int = 61) -> float:
        """
        Calculates the reciprocal rank fusion. The constant K is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper [https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf] used 1-based ranking).
        """
        return 1 / (k + rank)

    def _calc_recentness_score(self, rank: int, amount: int) -> float:
        """
        Calculate recentness score as a linear score between most recent and oldest document.
        This linear scaling is useful to
          a) reduce the effect of outliers and
          b) create recentness scoress that are meaningfully distributed in [0,1],
             similar to scores coming from a retriever/ranker
        """
        return (amount - rank) / amount
