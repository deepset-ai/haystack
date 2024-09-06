# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class CSVToDocument:
    """
    Converts CSV files to Documents.

    By default, it uses UTF-8 encoding when converting files but
    you can also set a custom encoding.
    It can attach metadata to the resulting documents.

    ### Usage example

    ```python
    from haystack.components.converters.csv import CSVToDocument
    converter = CSVToDocument()
    results = converter.run(sources=["sample.csv"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'col1,col2\now1,row1\nrow2row2\n'
    ```
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Creates a CSVToDocument component.

        :param encoding:
            The encoding of the csv files to convert.
            If the encoding is specified in the metadata of a source ByteStream,
            it overrides this value.
        """
        self.encoding = encoding

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts a CSV file to a Document.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output documents.
        :returns:
            A dictionary with the following keys:
            - `documents`: Created documents
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
                encoding = bytestream.meta.get("encoding", self.encoding)
                data = io.BytesIO(bytestream.data).getvalue().decode(encoding=encoding)
            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document = Document(content=data, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
