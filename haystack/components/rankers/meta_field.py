# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional

from dateutil.parser import parse as date_parse

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class MetaFieldRanker:
    """
    Ranks Documents based on the value of their specific meta field.

    The ranking can be performed in descending order or ascending order.

    Usage example:

    ```python
    from haystack import Document
    from haystack.components.rankers import MetaFieldRanker

    ranker = MetaFieldRanker(meta_field="rating")
    docs = [
        Document(content="Paris", meta={"rating": 1.3}),
        Document(content="Berlin", meta={"rating": 0.7}),
        Document(content="Barcelona", meta={"rating": 2.1}),
    ]

    output = ranker.run(documents=docs)
    docs = output["documents"]
    assert docs[0].content == "Barcelona"
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        meta_field: str,
        weight: float = 1.0,
        top_k: Optional[int] = None,
        ranking_mode: Literal["reciprocal_rank_fusion", "linear_score"] = "reciprocal_rank_fusion",
        sort_order: Literal["ascending", "descending"] = "descending",
        missing_meta: Literal["drop", "top", "bottom"] = "bottom",
        meta_value_type: Optional[Literal["float", "int", "date"]] = None,
    ):
        """
        Creates an instance of MetaFieldRanker.

        :param meta_field:
            The name of the meta field to rank by.
        :param weight:
            In range [0,1].
            0 disables ranking by a meta field.
            0.5 ranking from previous component and based on meta field have the same weight.
            1 ranking by a meta field only.
        :param top_k:
            The maximum number of Documents to return per query.
            If not provided, the Ranker returns all documents it receives in the new ranking order.
        :param ranking_mode:
            The mode used to combine the Retriever's and Ranker's scores.
            Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
            Use the 'linear_score' mode only with Retrievers or Rankers that return a score in range [0,1].
        :param sort_order:
            Whether to sort the meta field by ascending or descending order.
            Possible values are `descending` (default) and `ascending`.
        :param missing_meta:
            What to do with documents that are missing the sorting metadata field.
            Possible values are:
                - 'drop' will drop the documents entirely.
                - 'top' will place the documents at the top of the metadata-sorted list
                    (regardless of 'ascending' or 'descending').
                - 'bottom' will place the documents at the bottom of metadata-sorted list
                    (regardless of 'ascending' or 'descending').
        :param meta_value_type:
            Parse the meta value into the data type specified before sorting.
            This will only work if all meta values stored under `meta_field` in the provided documents are strings.
            For example, if we specified `meta_value_type="date"` then for the meta value `"date": "2015-02-01"`
            we would parse the string into a datetime object and then sort the documents by date.
            The available options are:
            - 'float' will parse the meta values into floats.
            - 'int' will parse the meta values into integers.
            - 'date' will parse the meta values into datetime objects.
            - 'None' (default) will do no parsing.
        """

        self.meta_field = meta_field
        self.weight = weight
        self.top_k = top_k
        self.ranking_mode = ranking_mode
        self.sort_order = sort_order
        self.missing_meta = missing_meta
        self._validate_params(
            weight=self.weight,
            top_k=self.top_k,
            ranking_mode=self.ranking_mode,
            sort_order=self.sort_order,
            missing_meta=self.missing_meta,
            meta_value_type=meta_value_type,
        )
        self.meta_value_type = meta_value_type

    def _validate_params(
        self,
        *,
        weight: float,
        top_k: Optional[int],
        ranking_mode: Literal["reciprocal_rank_fusion", "linear_score"],
        sort_order: Literal["ascending", "descending"],
        missing_meta: Literal["drop", "top", "bottom"],
        meta_value_type: Optional[Literal["float", "int", "date"]],
    ):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0, but got %s" % top_k)

        if weight < 0 or weight > 1:
            raise ValueError(
                "Parameter <weight> must be in range [0,1] but is currently set to '%s'.\n'0' disables sorting by a "
                "meta field, '0.5' assigns equal weight to the previous relevance scores and the meta field, and "
                "'1' ranks by the meta field only.\nChange the <weight> parameter to a value in range 0 to 1 when "
                "initializing the MetaFieldRanker." % weight
            )

        if ranking_mode not in ["reciprocal_rank_fusion", "linear_score"]:
            raise ValueError(
                "The value of parameter <ranking_mode> must be 'reciprocal_rank_fusion' or 'linear_score', but is "
                "currently set to '%s'.\nChange the <ranking_mode> value to 'reciprocal_rank_fusion' or "
                "'linear_score' when initializing the MetaFieldRanker." % ranking_mode
            )

        if sort_order not in ["ascending", "descending"]:
            raise ValueError(
                "The value of parameter <sort_order> must be 'ascending' or 'descending', "
                "but is currently set to '%s'.\n"
                "Change the <sort_order> value to 'ascending' or 'descending' when initializing the "
                "MetaFieldRanker." % sort_order
            )

        if missing_meta not in ["drop", "top", "bottom"]:
            raise ValueError(
                "The value of parameter <missing_meta> must be 'drop', 'top', or 'bottom', "
                "but is currently set to '%s'.\n"
                "Change the <missing_meta> value to 'drop', 'top', or 'bottom' when initializing the "
                "MetaFieldRanker." % missing_meta
            )

        if meta_value_type not in ["float", "int", "date", None]:
            raise ValueError(
                "The value of parameter <meta_value_type> must be 'float', 'int', 'date' or None but is "
                "currently set to '%s'.\n"
                "Change the <meta_value_type> value to 'float', 'int', 'date' or None when initializing the "
                "MetaFieldRanker." % meta_value_type
            )

    @component.output_types(documents=List[Document])
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        documents: List[Document],
        top_k: Optional[int] = None,
        weight: Optional[float] = None,
        ranking_mode: Optional[Literal["reciprocal_rank_fusion", "linear_score"]] = None,
        sort_order: Optional[Literal["ascending", "descending"]] = None,
        missing_meta: Optional[Literal["drop", "top", "bottom"]] = None,
        meta_value_type: Optional[Literal["float", "int", "date"]] = None,
    ):
        """
        Ranks a list of Documents based on the selected meta field by:

        1. Sorting the Documents by the meta field in descending or ascending order.
        2. Merging the rankings from the previous component and based on the meta field according to ranking mode and
        weight.
        3. Returning the top-k documents.

        :param documents:
            Documents to be ranked.
        :param top_k:
            The maximum number of Documents to return per query.
            If not provided, the top_k provided at initialization time is used.
        :param weight:
            In range [0,1].
            0 disables ranking by a meta field.
            0.5 ranking from previous component and based on meta field have the same weight.
            1 ranking by a meta field only.
            If not provided, the weight provided at initialization time is used.
        :param ranking_mode:
            (optional) The mode used to combine the Retriever's and Ranker's scores.
            Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
            Use the 'score' mode only with Retrievers or Rankers that return a score in range [0,1].
            If not provided, the ranking_mode provided at initialization time is used.
        :param sort_order:
            Whether to sort the meta field by ascending or descending order.
            Possible values are `descending` (default) and `ascending`.
            If not provided, the sort_order provided at initialization time is used.
        :param missing_meta:
            What to do with documents that are missing the sorting metadata field.
            Possible values are:
            - 'drop' will drop the documents entirely.
            - 'top' will place the documents at the top of the metadata-sorted list
                (regardless of 'ascending' or 'descending').
            - 'bottom' will place the documents at the bottom of metadata-sorted list
                (regardless of 'ascending' or 'descending').
            If not provided, the missing_meta provided at initialization time is used.
        :param meta_value_type:
            Parse the meta value into the data type specified before sorting.
            This will only work if all meta values stored under `meta_field` in the provided documents are strings.
            For example, if we specified `meta_value_type="date"` then for the meta value `"date": "2015-02-01"`
            we would parse the string into a datetime object and then sort the documents by date.
            The available options are:
            -'float' will parse the meta values into floats.
            -'int' will parse the meta values into integers.
            -'date' will parse the meta values into datetime objects.
            -'None' (default) will do no parsing.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of Documents sorted by the specified meta field.

        :raises ValueError:
            If `top_k` is not > 0.
            If `weight` is not in range [0,1].
            If `ranking_mode` is not 'reciprocal_rank_fusion' or 'linear_score'.
            If `sort_order` is not 'ascending' or 'descending'.
            If `meta_value_type` is not 'float', 'int', 'date' or `None`.
        """
        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        weight = weight if weight is not None else self.weight
        ranking_mode = ranking_mode or self.ranking_mode
        sort_order = sort_order or self.sort_order
        missing_meta = missing_meta or self.missing_meta
        meta_value_type = meta_value_type or self.meta_value_type
        self._validate_params(
            weight=weight,
            top_k=top_k,
            ranking_mode=ranking_mode,
            sort_order=sort_order,
            missing_meta=missing_meta,
            meta_value_type=meta_value_type,
        )

        # If the weight is 0 then ranking by meta field is disabled and the original documents should be returned
        if weight == 0:
            return {"documents": documents[:top_k]}

        docs_with_meta_field = [doc for doc in documents if self.meta_field in doc.meta]
        docs_missing_meta_field = [doc for doc in documents if self.meta_field not in doc.meta]

        # If all docs are missing self.meta_field return original documents
        if len(docs_with_meta_field) == 0:
            logger.warning(
                "The parameter <meta_field> is currently set to '{meta_field}', but none of the provided "
                "Documents with IDs {document_ids} have this meta key.\n"
                "Set <meta_field> to the name of a field that is present within the provided Documents.\n"
                "Returning the <top_k> of the original Documents since there are no values to rank.",
                meta_field=self.meta_field,
                document_ids=",".join([doc.id for doc in documents]),
            )
            return {"documents": documents[:top_k]}

        if len(docs_missing_meta_field) > 0:
            warning_start = (
                f"The parameter <meta_field> is currently set to '{self.meta_field}' but the Documents "
                f"with IDs {','.join([doc.id for doc in docs_missing_meta_field])} don't have this meta key.\n"
            )

            if missing_meta == "bottom":
                logger.warning(
                    "{warning_start}Because the parameter <missing_meta> is set to 'bottom', these Documents will be "
                    "placed at the end of the sorting order.",
                    warning_start=warning_start,
                )
            elif missing_meta == "top":
                logger.warning(
                    "{warning_start}Because the parameter <missing_meta> is set to 'top', these Documents will be "
                    "placed at the top of the sorting order.",
                    warning_start=warning_start,
                )
            else:
                logger.warning(
                    "{warning_start}Because the parameter <missing_meta> is set to 'drop', these Documents will be "
                    "removed from the list of retrieved Documents.",
                    warning_start=warning_start,
                )

        # If meta_value_type is provided try to parse the meta values
        parsed_meta = self._parse_meta(docs_with_meta_field=docs_with_meta_field, meta_value_type=meta_value_type)
        tuple_parsed_meta_and_docs = list(zip(parsed_meta, docs_with_meta_field))

        # Sort the documents by self.meta_field
        reverse = sort_order == "descending"
        try:
            tuple_sorted_by_meta = sorted(tuple_parsed_meta_and_docs, key=lambda x: x[0], reverse=reverse)
        except TypeError as error:
            # Return original documents if mixed types that are not comparable are returned (e.g. int and list)
            logger.warning(
                "Tried to sort Documents with IDs {document_ids}, but got TypeError with the message: {error}\n"
                "Returning the <top_k> of the original Documents since meta field ranking is not possible.",
                document_ids=",".join([doc.id for doc in docs_with_meta_field]),
                error=error,
            )
            return {"documents": documents[:top_k]}

        # Merge rankings and handle missing meta fields as specified in the missing_meta parameter
        sorted_by_meta = [doc for meta, doc in tuple_sorted_by_meta]
        if missing_meta == "bottom":
            sorted_documents = sorted_by_meta + docs_missing_meta_field
            sorted_documents = self._merge_rankings(documents, sorted_documents, weight, ranking_mode)
        elif missing_meta == "top":
            sorted_documents = docs_missing_meta_field + sorted_by_meta
            sorted_documents = self._merge_rankings(documents, sorted_documents, weight, ranking_mode)
        else:
            sorted_documents = sorted_by_meta
            sorted_documents = self._merge_rankings(docs_with_meta_field, sorted_documents, weight, ranking_mode)

        return {"documents": sorted_documents[:top_k]}

    def _parse_meta(
        self, docs_with_meta_field: List[Document], meta_value_type: Optional[Literal["float", "int", "date"]]
    ) -> List[Any]:
        """
        Parse the meta values stored under `self.meta_field` for the Documents provided in `docs_with_meta_field`.
        """
        if meta_value_type is None:
            return [d.meta[self.meta_field] for d in docs_with_meta_field]

        unique_meta_values = {doc.meta[self.meta_field] for doc in docs_with_meta_field}
        if not all(isinstance(meta_value, str) for meta_value in unique_meta_values):
            logger.warning(
                "The parameter <meta_value_type> is currently set to '{meta_field}', but not all of meta values in the "
                "provided Documents with IDs {document_ids} are strings.\n"
                "Skipping parsing of the meta values.\n"
                "Set all meta values found under the <meta_field> parameter to strings to use <meta_value_type>.",
                meta_field=meta_value_type,
                document_ids=",".join([doc.id for doc in docs_with_meta_field]),
            )
            return [d.meta[self.meta_field] for d in docs_with_meta_field]

        parse_fn: Callable
        if meta_value_type == "float":
            parse_fn = float
        elif meta_value_type == "int":
            parse_fn = int
        else:
            parse_fn = date_parse

        try:
            meta_values = [parse_fn(d.meta[self.meta_field]) for d in docs_with_meta_field]
        except ValueError as error:
            logger.warning(
                "Tried to parse the meta values of Documents with IDs {document_ids}, but got ValueError with the "
                "message: {error}\n"
                "Skipping parsing of the meta values.",
                document_ids=",".join([doc.id for doc in docs_with_meta_field]),
                error=error,
            )
            meta_values = [d.meta[self.meta_field] for d in docs_with_meta_field]

        return meta_values

    def _merge_rankings(
        self,
        documents: List[Document],
        sorted_documents: List[Document],
        weight: float,
        ranking_mode: Literal["reciprocal_rank_fusion", "linear_score"],
    ) -> List[Document]:
        """
        Merge the two different rankings for Documents sorted both by their content and by their meta field.
        """
        scores_map: Dict = defaultdict(int)

        if ranking_mode == "reciprocal_rank_fusion":
            for i, (document, sorted_doc) in enumerate(zip(documents, sorted_documents)):
                scores_map[document.id] += self._calculate_rrf(rank=i) * (1 - weight)
                scores_map[sorted_doc.id] += self._calculate_rrf(rank=i) * weight
        elif ranking_mode == "linear_score":
            for i, (document, sorted_doc) in enumerate(zip(documents, sorted_documents)):
                score = float(0)
                if document.score is None:
                    logger.warning("The score wasn't provided; defaulting to 0.")
                elif document.score < 0 or document.score > 1:
                    logger.warning(
                        "The score {score} for Document {document_id} is outside the [0,1] range; defaulting to 0",
                        score=document.score,
                        document_id=document.id,
                    )
                else:
                    score = document.score

                scores_map[document.id] += score * (1 - weight)
                scores_map[sorted_doc.id] += self._calc_linear_score(rank=i, amount=len(sorted_documents)) * weight

        for document in documents:
            document.score = scores_map[document.id]

        new_sorted_documents = sorted(documents, key=lambda doc: doc.score if doc.score else -1, reverse=True)
        return new_sorted_documents

    @staticmethod
    def _calculate_rrf(rank: int, k: int = 61) -> float:
        """
        Calculates the reciprocal rank fusion.

        The constant K is set to 61 (60 was suggested by the original paper, plus 1 as python lists are 0-based and
        the [paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) used 1-based ranking).
        """
        return 1 / (k + rank)

    @staticmethod
    def _calc_linear_score(rank: int, amount: int) -> float:
        """
        Calculate the meta field score as a linear score between the greatest and the lowest score in the list.

        This linear scaling is useful for:
        - Reducing the effect of outliers
        - Creating scores that are meaningfully distributed in the range [0,1],
        similar to scores coming from a Retriever or Ranker.
        """
        return (amount - rank) / amount
