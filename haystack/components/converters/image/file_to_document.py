# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class ImageFileToDocument:
    """
    Converts image file references into empty Document objects with associated metadata.

    This component is useful in pipelines where image file paths need to be wrapped in `Document` objects to be
    processed by downstream components such as the `SentenceTransformersImageDocumentEmbedder`.

    It does **not** extract any content from the image files, instead it creates `Document` objects with `None` as
    their content and attaches metadata such as file path and any user-provided values.

    ### Usage example
    ```python
    from haystack.components.converters.image import ImageFileToDocument

    converter = ImageFileToDocument()

    sources = ["image.jpg", "another_image.png"]

    result = converter.run(sources=sources)
    documents = result["documents"]

    print(documents)

    # [Document(id=..., meta: {'file_path': 'image.jpg'}),
    # Document(id=..., meta: {'file_path': 'another_image.png'})]
    ```
    """

    def __init__(self, *, store_full_path: bool = False):
        """
        Initialize the ImageFileToDocument component.

        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        self.store_full_path = store_full_path

    @component.output_types(documents=List[Document])
    def run(
        self,
        *,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """
        Convert image files into empty Document objects with metadata.

        This method accepts image file references (as file paths or ByteStreams) and creates `Document` objects
        without content. These documents are enriched with metadata derived from the input source and optional
        user-provided metadata.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources, as they are zipped together.
            For ByteStream objects, their `meta` is added to the output documents.

        :returns:
            A dictionary containing:
            - `documents`: A list of `Document` objects with empty content and associated metadata.
        """

        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)
            document = Document(content=None, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
