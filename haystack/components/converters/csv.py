# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

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

    def __init__(
        self,
        encoding: str = "utf-8",
        store_full_path: bool = False,
        *,
        conversion_mode: Literal["file", "row"] = "file",
        content_column: Optional[str] = None,
        delimiter: str = ",",
        quotechar: str = '"',
    ):
        """
        Creates a CSVToDocument component.

        :param encoding:
            The encoding of the csv files to convert.
            If the encoding is specified in the metadata of a source ByteStream,
            it overrides this value.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        :param conversion_mode:
            - "file" (default): current behavior, one Document per CSV file whose content is the raw CSV text.
            - "row": convert each CSV row to its own Document.
        :param content_column:
            When ``conversion_mode="row"``, the column to use as ``Document.content``.
            If ``None``, the content will be a human-readable "key: value" listing of that row.
        :param delimiter:
            CSV delimiter used when parsing in row mode (passed to ``csv.DictReader``).
        :param quotechar:
            CSV quote character used when parsing in row mode (passed to ``csv.DictReader``).
        """

        self.encoding = encoding
        self.store_full_path = store_full_path
        self.conversion_mode = conversion_mode
        self.content_column = content_column
        self.delimiter = delimiter
        self.quotechar = quotechar

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
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
        documents: list[Document] = []

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

            # Mode: file (backward-compatible default) -> one Document per file
            if self.conversion_mode == "file":
                documents.append(Document(content=data, meta=merged_metadata))
                continue

            # Mode: row -> one Document per CSV row
            try:
                reader = csv.DictReader(io.StringIO(data), delimiter=self.delimiter, quotechar=self.quotechar)
            except Exception as e:
                logger.warning(
                    "Could not parse CSV rows for {source}. Falling back to file mode. Error: {error}",
                    source=source,
                    error=e,
                )
                documents.append(Document(content=data, meta=merged_metadata))
                continue

            for i, row in enumerate(reader):
                row_meta = dict(merged_metadata)  # start with file-level/meta param  bytestream meta

                # Determine content from selected column or fallback to a friendly listing
                if self.content_column:
                    content = row.get(self.content_column, "")
                    if content is None:
                        content = ""
                else:
                    # "key: value" per line for readability
                    content = "\n".join(f"{k}: {v if v is not None else ''}" for k, v in row.items())

                # Add remaining columns into meta (don't override existing keys like file_path, encoding, etc.)
                for k, v in row.items():
                    if self.content_column and k == self.content_column:
                        continue
                    if k not in row_meta:
                        row_meta[k] = "" if v is None else v

                row_meta["row_number"] = i
                documents.append(Document(content=content, meta=row_meta))

        return {"documents": documents}
