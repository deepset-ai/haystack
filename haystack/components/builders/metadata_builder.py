import logging
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


@component
class MetadataBuilder:
    """
    A component to add the output of a Generator to a list of Documents as metadata.
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], replies: List[str], meta: Optional[List[Dict[str, Any]]] = None):
        """
        The MetadataBuilder component takes a list of Documents, the output of a Generator to which these Documents were passed,
        and adds the output from the Generator as metadata to the Documents.
        The Generator takes a list of Documents, and returns replies and metadata.
        The MetadataBuilder component takes these replies and metadata and adds them to the Documents.
        It does this by adding the replies and metadata to the metadata of the Document.

        :param documents: The documents used as input to the Generator. A list of `Document` objects.
        :param replies: The output of the Generator. A list of strings.
        :param metadata: The metadata returned by the Generator. An optional list of dictionaries. If not specified,
                            the generated documents will contain no additional metadata.
        """
        if not meta:
            meta = [{}] * len(replies)

        if not len(documents) == len(replies) == len(meta):
            raise ValueError(
                f"Number of Documents ({len(documents)}), replies ({len(replies)}), and metadata ({len(meta)})"
                " must match."
            )

        for doc, reply, meta_val in zip(documents, replies, meta):
            doc.meta.update({"reply": reply, **meta_val})

        return {"documents": documents}
