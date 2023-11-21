import logging
import warnings
from collections import defaultdict
from typing import List, Dict, Any, Optional, Literal

from haystack.preview import ComponentError, Document, component, default_to_dict

logger = logging.getLogger(__name__)


@component
class MetaFieldRanker:
    """
    Ranks Documents based on the value of their specific metadata field. The ranking is done in a descending order.

    Usage example:
    ```
    from haystack.preview import Document
    from haystack.preview.components.rankers import MetaFieldRanker

    ranker = MetaFieldRanker(metadata_field="rating")
    docs = [
        Document(text="Paris", metadata={"rating": 1.3}),
        Document(text="Berlin", metadata={"rating": 0.7}),
        Document(text="Barcelona", metadata={"rating": 2.1}),
    ]

    output = ranker.run(documents=docs)
    docs = output["documents"]
    assert docs[0].text == "Barcelona"
    """

    def __init__(
        self,
        metadata_field: str,
        weight: float = 1.0,
        top_k: Optional[int] = None,
        ranking_mode: Literal["reciprocal_rank_fusion", "linear_score"] = "reciprocal_rank_fusion",
    ):
        """
        Creates an instance of MetaFieldRanker.

        :param metadata_field: The name of the metadata field to rank by.
        :param weight: In range [0,1].
                0 disables ranking by a metadata field.
                0.5 content and metadata fields have the same impact for the ranking.
                1 means ranking by a metadata field only. The highest value comes first.
        :param top_k: The maximum number of Documents you want the Ranker to return per query.
        :param ranking_mode: The mode used to combine the Retriever's and Ranker's scores.
                Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
                Use the 'score' mode only with Retrievers or Rankers that return a score in range [0,1].
        """

        self.metadata_field = metadata_field
        self.weight = weight
        self.top_k = top_k
        self.ranking_mode = ranking_mode

        if self.weight < 0 or self.weight > 1:
            raise ValueError(
                """
                Parameter <weight> must be in range [0,1] but is currently set to '{}'.\n
                '0' disables sorting by a metadata field, '0.5' assigns equal weight to the previous relevance scores and the metadata field, and '1' ranks by the metadata field only.\n
                Change the <weight> parameter to a value in range 0 to 1 when initializing the MetaFieldRanker.
                """.format(
                    self.weight
                )
            )

        if self.ranking_mode not in ["reciprocal_rank_fusion", "linear_score"]:
            raise ValueError(
                """
                The value of parameter <ranking_mode> must be 'reciprocal_rank_fusion' or 'linear_score', but is currently set to '{}'. \n
                Change the <ranking_mode> value to 'reciprocal_rank_fusion' or 'linear_score' when initializing the MetaFieldRanker.
                """.format(
                    self.ranking_mode
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize object to a dictionary.
        """
        return default_to_dict(
            self,
            metadata_field=self.metadata_field,
            weight=self.weight,
            top_k=self.top_k,
            ranking_mode=self.ranking_mode,
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], top_k: Optional[int] = None):
        """
        Use this method to rank a list of Documents based on the selected metadata field by:
        1. Sorting the Documents by the metadata field in descending order.
        2. Merging the scores from the metadata field with the scores from the previous component according to the strategy and weight provided.
        3. Returning the top-k documents.

        :param documents: Documents to be ranked.
        :param top_k: (optional) The number of Documents you want the Ranker to return. If not provided, the Ranker returns all Documents it received.
        """
        if not documents:
            return {"documents": []}

        if top_k is None:
            top_k = self.top_k
        elif top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        try:
            sorted_by_metadata = sorted(documents, key=lambda doc: doc.meta[self.metadata_field], reverse=True)
        except KeyError:
            raise ComponentError(
                """
                The parameter <metadata_field> is currently set to '{}' but the Documents {} don't have this metadata key.\n
                Double-check the names of the metadata fields in your documents \n
                and set <metadata_field> to the name of the field that contains the metadata you want to use for ranking.
                """.format(
                    self.metadata_field, ",".join([doc.id for doc in documents if self.metadata_field not in doc.meta])
                )
            )

        if self.weight > 0:
            sorted_documents = self._merge_scores(documents, sorted_by_metadata)
            return {"documents": sorted_documents[:top_k]}
        else:
            return {"documents": sorted_by_metadata[:top_k]}

    def _merge_scores(self, documents: List[Document], sorted_documents: List[Document]) -> List[Document]:
        """
        Merge scores for Documents sorted both by their content and by their metadata field.
        """
        scores_map: Dict = defaultdict(int)

        if self.ranking_mode == "reciprocal_rank_fusion":
            for i, (doc, sorted_doc) in enumerate(zip(documents, sorted_documents)):
                scores_map[doc.id] += self._calculate_rrf(rank=i) * (1 - self.weight)
                scores_map[sorted_doc.id] += self._calculate_rrf(rank=i) * self.weight
        elif self.ranking_mode == "linear_score":
            for i, (doc, sorted_doc) in enumerate(zip(documents, sorted_documents)):
                score = float(0)
                if doc.score is None:
                    warnings.warn("The score wasn't provided; defaulting to 0.")
                elif doc.score < 0 or doc.score > 1:
                    warnings.warn(
                        "The score {} for Document {} is outside the [0,1] range; defaulting to 0".format(
                            doc.score, doc.id
                        )
                    )
                else:
                    score = doc.score

                scores_map[doc.id] += score * (1 - self.weight)
                scores_map[sorted_doc.id] += self._calc_linear_score(rank=i, amount=len(sorted_documents)) * self.weight

        for doc in documents:
            doc.score = scores_map[doc.id]

        new_sorted_documents = sorted(documents, key=lambda doc: doc.score if doc.score else -1, reverse=True)
        return new_sorted_documents

    @staticmethod
    def _calculate_rrf(rank: int, k: int = 61) -> float:
        """
        Calculates the reciprocal rank fusion. The constant K is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the [paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) used 1-based ranking).
        """
        return 1 / (k + rank)

    @staticmethod
    def _calc_linear_score(rank: int, amount: int) -> float:
        """
        Calculate the metadata field score as a linear score between the greatest and the lowest score in the list.
        This linear scaling is useful for:
          - Reducing the effect of outliers
          - Creating scores that are meaningfully distributed in the range [0,1],
             similar to scores coming from a Retriever or Ranker.
        """
        return (amount - rank) / amount
