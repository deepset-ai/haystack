# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

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

    def __init__(self, encoding: str = "utf-8", store_full_path: bool = False, split_by_row: bool = False):
        """
        Creates a CSVToDocument component.

        :param encoding:
            The encoding of the csv files to convert.
            If the encoding is specified in the metadata of a source ByteStream,
            it overrides this value.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        :param split_by_row:
            If True, each row of the CSV file is converted into a separate document.
            If False, the entire CSV file is converted into a single document.
        """
        self.encoding = encoding
        self.store_full_path = store_full_path
        self.split_by_row = split_by_row

    def _convert_file_mode(self, data: str, metadata: Dict[str, Any]) -> List[Document]:
        """Convert entire CSV file into a single document"""
        return [Document(content=data, meta=metadata)]

    def _convert_row_mode(self, data: str, metadata: Dict[str, Any]) -> List[Document]:
        """Convert each CSV row into a separate document"""
        try:
            df = pd.read_csv(io.StringIO(data))
            documents = []
            header = ",".join(df.columns)
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_values = ",".join(str(v) for v in row_dict.values())
                content = f"{header}\n{row_values}"

                doc = Document(content=content, meta=metadata)
                documents.append(doc)
            return documents
        except Exception as e:
            logger.warning("Error converting CSV rows to documents: {error}", error=e)
            return []

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

            if not self.store_full_path and "file_path" in bytestream.meta:
                file_path = bytestream.meta.get("file_path")
                if file_path:  # Ensure the value is not None for pylint
                    merged_metadata["file_path"] = os.path.basename(file_path)

            if self.split_by_row:
                documents.extend(self._convert_row_mode(data, merged_metadata))
            else:
                documents.extend(self._convert_file_mode(data, merged_metadata))

        return {"documents": documents}
