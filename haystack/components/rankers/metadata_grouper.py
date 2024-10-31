from collections import defaultdict
from typing import Any, Dict, List, Optional

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
        :param sort_docs_by: Determines which metadata keys are used to sort the documents.

        """
        self.group_by = group_by
        self.sort_docs_by = sort_docs_by
        self.subgroup_by = subgroup_by

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:  # noqa: C901, PLR0912
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
        ordered_keys = []
        document_groups: Dict[str, List] = defaultdict(list)
        no_group: list = []

        # Go through all documents and bucket them based on the 'group_by' value
        for document in documents:
            if self.group_by in document.meta:
                document_groups[str(document.meta[self.group_by])].append(document)
                if str(document.meta[self.group_by]) not in ordered_keys:
                    ordered_keys.append(str(document.meta[self.group_by]))
            else:
                no_group.append(document)

        # Go through all documents and bucket them based on the 'subgroup_by' value
        subgroup_ordered_keys = []
        document_subgroups: Dict[str, dict] = defaultdict(lambda: defaultdict(list))
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

        result_docs = []

        # Sort the docs within the groups or subgroups if necessary
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

        return {"documents": result_docs}
