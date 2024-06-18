# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install python-pptx'") as pptx_import:
    from pptx import Presentation


logger = logging.getLogger(__name__)


@component
class PPTXToDocument:
    """
    Converts PPTX files to Documents.

    Usage example:
    ```python
    from haystack.components.converters.pptx import PPTXToDocument

    converter = PPTXToDocument()
    results = converter.run(sources=["sample.pptx"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is the text from the PPTX file.'
    ```
    """

    def __init__(self):
        """
        Create an PPTXToDocument component.
        """
        pptx_import.check()

    def _convert(self, file_content: io.BytesIO) -> str:
        """
        Converts the PPTX file to text.
        """
        pptx_presentation = Presentation(file_content)
        text_all_slides = []
        for slide in pptx_presentation.slides:
            text_on_slide = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_on_slide.append(shape.text)
            text_all_slides.append("\n".join(text_on_slide))
        text = "\f".join(text_all_slides)
        return text

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts PPTX files to Documents.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                text = self._convert(io.BytesIO(bytestream.data))
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            documents.append(Document(content=text, meta=merged_metadata))

        return {"documents": documents}
