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

_ROW_MODE_SIZE_WARN_BYTES = 5 * 1024 * 1024  # ~5MB; warn when parsing rows might be memory-heavy


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

        # Basic validation (reviewer suggestion)
        if len(self.delimiter) != 1:
            raise ValueError("CSVToDocument: delimiter must be a single character.")
        if len(self.quotechar) != 1:
            raise ValueError("CSVToDocument: quotechar must be a single character.")

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
                raw = io.BytesIO(bytestream.data).getvalue()
                data = raw.decode(encoding=encoding)

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

            # Reviewer note: Warn for very large CSVs in row mode (memory consideration)
            try:
                size_bytes = len(raw)
                if size_bytes > _ROW_MODE_SIZE_WARN_BYTES:
                    logger.warning(
                        "CSVToDocument(row): parsing a large CSV (~{mb:.1f} MB). "
                        "Consider chunking/streaming if you hit memory issues.",
                        mb=size_bytes / (1024 * 1024),
                    )
            except Exception:
                pass

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

            # Validate content_column presence; fall back to listing if missing
            effective_content_col = self.content_column
            header = reader.fieldnames or []
            if effective_content_col and effective_content_col not in header:
                logger.warning(
                    "CSVToDocument(row): content_column='{col}' not found in header for {source}; "
                    "falling back to key: value listing.",
                    col=effective_content_col,
                    source=source,
                )
                effective_content_col = None

            for i, row in enumerate(reader):
                # Protect against malformed rows (reviewer suggestion)
                try:
                    doc = self._build_document_from_row(
                        row=row, base_meta=merged_metadata, row_index=i, content_column=effective_content_col
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(
                        "CSVToDocument(row): skipping malformed row {row_index} in {source}. Error: {error}",
                        row_index=i,
                        source=source,
                        error=e,
                    )

        return {"documents": documents}

    # ----- helpers -----
    def _safe_value(self, value: Any) -> str:
        """Normalize CSV cell values: None -> '', everything -> str."""
        return "" if value is None else str(value)

    def _build_document_from_row(
        self, row: dict[str, Any], base_meta: dict[str, Any], row_index: int, content_column: Optional[str]
    ) -> Document:
        """
        Create a Document from a single CSV row. Does not catch exceptions; caller wraps.
        """
        row_meta = dict(base_meta)

        # content
        if content_column:
            content = self._safe_value(row.get(content_column))
        else:
            content = "\n".join(f"{k}: {self._safe_value(v)}" for k, v in row.items())

        # merge remaining columns into meta with collision handling
        for k, v in row.items():
            if content_column and k == content_column:
                continue
            key_to_use = k
            if key_to_use in row_meta:
                # Avoid clobbering existing meta like file_path/encoding; prefix and de-dupe
                base_key = f"csv_{key_to_use}"
                key_to_use = base_key
                suffix = 1
                while key_to_use in row_meta:
                    key_to_use = f"{base_key}_{suffix}"
                    suffix = 1
            row_meta[key_to_use] = self._safe_value(v)

        row_meta["row_number"] = row_index
        return Document(content=content, meta=row_meta)
