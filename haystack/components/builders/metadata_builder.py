import logging
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


class MetadataBuilder:
    def __init__(self, meta_keys: List[str]):
        self.meta_keys = meta_keys

    @component.output_types(documents=List[Document])
    def run(
        self, documents: List[Document], data: Dict[str, Any], meta: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[Document]]:
        """
        The MetadataBuilder component takes a list of Documents, the output of a component to which these Documents were passed,
        and adds the output from the component as metadata to the Documents.
        The MetadataBuilder component takes these replies and metadata and adds them to the Documents.
        It does this by adding the replies and metadata to the metadata of the Document.
        :param documents: The documents used as input to the Generator. A list of `Document` objects.
        :param data: The output of a component (Generator , TextEmbedder, EntityExtractor).
        :param meta: The metadata returned by the component.
        """
        if not meta:
            meta = [{}] * len(data)

        if not len(documents) == len(data) == len(meta):
            raise ValueError(
                f"Number of Documents ({len(documents)}), data ({len(data)}), and metadata ({len(meta)})" " must match."
            )

        meta = {key: data[key] for key in self.meta_keys}

        for i, doc in enumerate(documents):
            doc.meta.update(meta)

        return {"documents": documents}
