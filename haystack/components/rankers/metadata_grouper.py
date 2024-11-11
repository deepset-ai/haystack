# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class MetaDataGrouper:
    """
    It reorders the documents by grouping them based on metadata keys.

    It groups the documents based on metadata keys. It can group based on a:
     - single metadata key 'group_by'
     - further subgroup them based on another metadata key 'subgroup_by'.

    It can also sort the documents within each group or subgroup based on a specified metadata key, 'sort_docs_by'

    It returns a flat list with the original input documents ordered by the 'group_by' and 'subgroup_by' metadata
    values, and all documents without a group are included at the end of the list.

    This helps to ensure the documents are properly organized leading to a better performance of a following LLM.

    ### Usage example

    ```python
    from haystack.components.rankers import MetaDataGrouper
    from haystack.dataclasses import Document


    docs = [
        Document(content="Javascript is a popular programming language", meta={"group": "42", "split_id": 7, "subgroup": "subB"}),
        Document(content="Python is a popular programming language",meta={"group": "42", "split_id": 4, "subgroup": "subB"}),
        Document(content="A chromosome is a package of DNA", meta={"group": "314", "split_id": 2, "subgroup": "subC"}),
        Document(content="An octopus has three hearts", meta={"group": "11", "split_id": 2, "subgroup": "subD"}),
        Document(content="Java is a popular programming language", meta={"group": "42", "split_id": 3, "subgroup": "subB"})
    ]

    sample_meta_aggregator = MetaDataGrouper(group_by="group",subgroup_by="subgroup", sort_docs_by="split_id")
    result = sample_meta_aggregator.run(documents=docs)
    print(result["documents"])
    ```
    """  # noqa: E501

    def __init__(self, group_by: str, subgroup_by: Optional[str] = None, sort_docs_by: Optional[str] = None):
        """
        Creates an instance of DeepsetMetadataGrouper.

        :param group_by: The metadata key to aggregate the documents by.
        :param subgroup_by: The metadata key to aggregate the documents within a group that was created by the
                            `group_by` key.
        :param sort_docs_by: Determines which metadata keys are used to sort the documents. If not provided, the
                             documents within the groups or subgroups are not sorted and are kept in the same order as
                             they were inserted in the subgroups.

        """
        self.group_by = group_by
        self.sort_docs_by = sort_docs_by
        self.subgroup_by = subgroup_by

    def _buck_into_groups(self, documents: List[Document]) -> Tuple[Dict[str, List], List, List]:
        """
        Go through all documents and bucket them based on the 'group_by' value.

        If no 'group_by' value is present in the document, the document is added to a list of documents without a group.

        :param documents:
        :returns:
            A tuple with the following elements:
            - document_groups: A dictionary with the 'group_by' values as keys and the documents as values.
            - no_group: A list of documents without a 'group_by' value.
            - ordered_keys: A list of 'group_by' values in the order they were encountered.
        """

        document_groups: Dict[str, List] = defaultdict(list)
        ordered_keys: List = []
        no_group: List = []

        for document in documents:
            if self.group_by in document.meta:
                document_groups[str(document.meta[self.group_by])].append(document)
                if str(document.meta[self.group_by]) not in ordered_keys:
                    ordered_keys.append(str(document.meta[self.group_by]))
            else:
                no_group.append(document)

        return document_groups, no_group, ordered_keys

    def _buck_into_subgroups(self, document_groups: Dict[str, List]) -> Tuple[Dict[str, Dict], List]:
        """
        Buckets the documents within the groups based on the 'subgroup_by' value.

        If no 'subgroup_by' value is present in the document, the document is added to a subgroup with the key
        'no_subgroup'.

        :param document_groups: A dictionary with the 'group_by' values as keys and the documents as values.
        :returns:
            A tuple with the following elements:
            - document_subgroups: A dictionary with the 'subgroup_by' values as keys and the documents as values.
            - subgroup_ordered_keys: A list of 'subgroup_by' values in the order they were encountered
        """
        subgroup_ordered_keys = []
        document_subgroups: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))
        if self.subgroup_by:
            for key, value in document_groups.items():
                for doc in value:
                    if self.subgroup_by in doc.meta:
                        document_subgroups[key][str(doc.meta[self.subgroup_by])].append(doc)
                        if str(doc.meta[self.subgroup_by]) not in subgroup_ordered_keys:
                            subgroup_ordered_keys.append(str(doc.meta[self.subgroup_by]))
                    else:
                        document_subgroups[key]["no_subgroup"].append(doc)
                        if "no_subgroup" not in subgroup_ordered_keys:
                            subgroup_ordered_keys.append("no_subgroup")
        return document_subgroups, subgroup_ordered_keys

    def _sort_docs(
        self,
        document_groups: Dict[str, List],
        document_subgroups: Dict[str, Dict],
        no_group: List,
        ordered_keys: List,
        subgroup_ordered_keys: List,
    ):
        """
        Sorts the documents within the groups or subgroups based on the 'sort_docs_by' value.

        If 'sort_docs_by' is not provided, the documents are kept in the same order as they were inserted in the groups
        and subgroups.

        :param document_groups: A dictionary with the 'group_by' values as keys and the documents as values.
        :param document_subgroups: A dictionary with the 'subgroup_by' values as keys and the documents as values.
        :param no_group: A list of documents without a 'group_by' value.
        :param ordered_keys: A list of 'group_by' values in the order they were encountered.
        :param subgroup_ordered_keys: A list of 'subgroup_by' values in the order they were encountered.
        :returns:
            A list of documents ordered by the 'sort_docs_by' metadata values.
        """
        result_docs = []
        if self.sort_docs_by and not self.subgroup_by:
            for key in document_groups:
                result_docs += sorted(document_groups[key], key=lambda d: d.meta.get(self.sort_docs_by, float("inf")))

        elif self.sort_docs_by and self.subgroup_by:
            for group in ordered_keys:
                for subgroup in subgroup_ordered_keys:
                    result_docs += sorted(
                        document_subgroups[group][subgroup], key=lambda d: d.meta.get(self.sort_docs_by, float("inf"))
                    )
        else:
            for key in document_groups:
                result_docs += document_groups[key]
        for doc in no_group:
            if doc not in result_docs:
                result_docs += [doc]

        return result_docs

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:  # disable = too-many-positional-arguments
        """
        Groups the provided list of documents based on the `group_by` parameter and optionally the `subgroup_by`.

        The output is a list of documents re-ordered based on how they were grouped.

        :param documents: The list of documents to group.
        :returns:
            A dictionary with the following keys:
            - documents: The list of documents ordered by the `group_by` and `subgroup_by` metadata values.
        """
        if len(documents) == 0:
            return {"documents": []}

        # docs based on the 'group_by' value.
        document_groups, no_group, ordered_keys = self._buck_into_groups(documents)

        # further bucking of the document groups based on the 'subgroup_by' value
        document_subgroups, subgroup_ordered_keys = self._buck_into_subgroups(document_groups)

        # sort the docs within the groups or subgroups if necessary
        result_docs = self._sort_docs(
            document_groups, document_subgroups, no_group, ordered_keys, subgroup_ordered_keys
        )

        return {"documents": result_docs}
