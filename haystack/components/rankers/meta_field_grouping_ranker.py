# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Dict, List, Optional, cast

from haystack import Document, component


@component
class MetaFieldGroupingRanker:
    """
    Reorders the documents by grouping them based on metadata keys.

    The MetaFieldGroupingRanker can group documents by a primary metadata key `group_by`, and subgroup them with an optional
    secondary key, `subgroup_by`.
    Within each group or subgroup, it can also sort documents by a metadata key `sort_docs_by`.

    The output is a flat list of documents ordered by `group_by` and `subgroup_by` values.
    Any documents without a group are placed at the end of the list.

    The proper organization of documents helps improve the efficiency and performance of subsequent processing by an LLM.

    ### Usage example

    ```python
    from haystack.components.rankers import MetaFieldGroupingRanker
    from haystack.dataclasses import Document


    docs = [
        Document(content="Javascript is a popular programming language", meta={"group": "42", "split_id": 7, "subgroup": "subB"}),
        Document(content="Python is a popular programming language",meta={"group": "42", "split_id": 4, "subgroup": "subB"}),
        Document(content="A chromosome is a package of DNA", meta={"group": "314", "split_id": 2, "subgroup": "subC"}),
        Document(content="An octopus has three hearts", meta={"group": "11", "split_id": 2, "subgroup": "subD"}),
        Document(content="Java is a popular programming language", meta={"group": "42", "split_id": 3, "subgroup": "subB"})
    ]

    ranker = MetaFieldGroupingRanker(group_by="group",subgroup_by="subgroup", sort_docs_by="split_id")
    result = ranker.run(documents=docs)
    print(result["documents"])

    # [
    #     Document(id=d665bbc83e52c08c3d8275bccf4f22bf2bfee21c6e77d78794627637355b8ebc,
    #             content: 'Java is a popular programming language', meta: {'group': '42', 'split_id': 3, 'subgroup': 'subB'}),
    #     Document(id=a20b326f07382b3cbf2ce156092f7c93e8788df5d48f2986957dce2adb5fe3c2,
    #             content: 'Python is a popular programming language', meta: {'group': '42', 'split_id': 4, 'subgroup': 'subB'}),
    #     Document(id=ce12919795d22f6ca214d0f161cf870993889dcb146f3bb1b3e1ffdc95be960f,
    #             content: 'Javascript is a popular programming language', meta: {'group': '42', 'split_id': 7, 'subgroup': 'subB'}),
    #     Document(id=d9fc857046c904e5cf790b3969b971b1bbdb1b3037d50a20728fdbf82991aa94,
    #             content: 'A chromosome is a package of DNA', meta: {'group': '314', 'split_id': 2, 'subgroup': 'subC'}),
    #     Document(id=6d3b7bdc13d09aa01216471eb5fb0bfdc53c5f2f3e98ad125ff6b85d3106c9a3,
    #             content: 'An octopus has three hearts', meta: {'group': '11', 'split_id': 2, 'subgroup': 'subD'})
    # ]
    ```
    """  # noqa: E501

    def __init__(self, group_by: str, subgroup_by: Optional[str] = None, sort_docs_by: Optional[str] = None):
        """
        Creates an instance of MetaFieldGroupingRanker.

        :param group_by: The metadata key to aggregate the documents by.
        :param subgroup_by: The metadata key to aggregate the documents within a group that was created by the
                            `group_by` key.
        :param sort_docs_by: Determines which metadata key is used to sort the documents. If not provided, the
                             documents within the groups or subgroups are not sorted and are kept in the same order as
                             they were inserted in the subgroups.

        """
        self.group_by = group_by
        self.sort_docs_by = sort_docs_by
        self.subgroup_by = subgroup_by

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Groups the provided list of documents based on the `group_by` parameter and optionally the `subgroup_by`.

        The output is a list of documents reordered based on how they were grouped.

        :param documents: The list of documents to group.
        :returns:
            A dictionary with the following keys:
            - documents: The list of documents ordered by the `group_by` and `subgroup_by` metadata values.
        """

        if not documents:
            return {"documents": []}

        document_groups: Dict[str, Dict[str, List[Document]]] = defaultdict(lambda: defaultdict(list))
        no_group_docs = []

        for doc in documents:
            group_value = str(doc.meta.get(self.group_by, ""))

            if group_value:
                subgroup_value = "no_subgroup"
                if self.subgroup_by and self.subgroup_by in doc.meta:
                    subgroup_value = doc.meta[self.subgroup_by]

                document_groups[group_value][subgroup_value].append(doc)
            else:
                no_group_docs.append(doc)

        ordered_docs = []
        for subgroups in document_groups.values():
            for docs in subgroups.values():
                if self.sort_docs_by:
                    docs.sort(key=lambda d: d.meta.get(cast(str, self.sort_docs_by), float("inf")))
                ordered_docs.extend(docs)

        ordered_docs.extend(no_group_docs)

        return {"documents": ordered_docs}
