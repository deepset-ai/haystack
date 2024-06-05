# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install trafilatura'") as trafilatura_import:
    from trafilatura import extract


@component
class HTMLToDocument:
    """
    Converts an HTML file to a Document.

    Usage example:
    ```python
    from haystack.components.converters import HTMLToDocument

    converter = HTMLToDocument()
    results = converter.run(sources=["path/to/sample.html"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the HTML file.'
    ```
    """

    def __init__(
        self,
        extractor_type: Optional[str] = None,
        try_others: Optional[bool] = None,
        extraction_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an HTMLToDocument component.

        :param extractor_type: Ignored. This parameter is kept for compatibility with previous versions. It will be
            removed in Haystack 2.4.0. To customize the extraction, use the `extraction_kwargs` parameter.
        :param try_others: Ignored. This parameter is kept for compatibility with previous versions. It will be
            removed in Haystack 2.4.0.
        :param extraction_kwargs: A dictionary containing keyword arguments to customize the extraction process. These
            are passed to the underlying Trafilatura `extract` function. For the full list of available arguments, see
            the [Trafilatura documentation](https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract).
        """
        trafilatura_import.check()
        if extractor_type is not None:
            warnings.warn(
                "The `extractor_type` parameter is ignored and will be removed in Haystack 2.4.0. "
                "To customize the extraction, use the `extraction_kwargs` parameter.",
                DeprecationWarning,
            )
        if try_others is not None:
            warnings.warn(
                "The `try_others` parameter is ignored and will be removed in Haystack 2.4.0. ", DeprecationWarning
            )

        self.extraction_kwargs = extraction_kwargs or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, extraction_kwargs=self.extraction_kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HTMLToDocument":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        extraction_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Converts a list of HTML files to Documents.

        :param sources:
            List of HTML file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.
        :param extraction_kwargs:
            Additional keyword arguments to customize the extraction process.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """

        merged_extraction_kwargs = {**self.extraction_kwargs, **(extraction_kwargs or {})}

        documents = []
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            try:
                text = extract(bytestream.data.decode("utf-8"), **merged_extraction_kwargs)
            except Exception as conversion_e:
                logger.warning(
                    "Failed to extract text from {source}. Skipping it. Error: {error}",
                    source=source,
                    error=conversion_e,
                )
                continue

            document = Document(content=text, meta={**bytestream.meta, **metadata})
            documents.append(document)

        return {"documents": documents}
