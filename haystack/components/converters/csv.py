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
    # 'col1,col2\\nrow1,row1\\nrow2,row2\\n'
    ```
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        store_full_path: bool = False,
        *,
        conversion_mode: Literal["file", "row"] = "file",
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
            - "file" (default): one Document per CSV file whose content is the raw CSV text.
            - "row": convert each CSV row to its own Document (requires `content_column` in `run()`).
        :param delimiter:
            CSV delimiter used when parsing in row mode (passed to ``csv.DictReader``).
        :param quotechar:
            CSV quote character used when parsing in row mode (passed to ``csv.DictReader``).
        """
        self.encoding = encoding
        self.store_full_path = store_full_path
        self.conversion_mode = conversion_mode
        self.delimiter = delimiter
        self.quotechar = quotechar

        # Basic validation
        if len(self.delimiter) != 1:
            raise ValueError("CSVToDocument: delimiter must be a single character.")
        if len(self.quotechar) != 1:
            raise ValueError("CSVToDocument: quotechar must be a single character.")

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[Union[str, Path, ByteStream]],
        *,
        content_column: Optional[str] = None,
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
    ):
        """
        Converts CSV files to a Document (file mode) or to one Document per row (row mode).

        :param sources:
            List of file paths or ByteStream objects.
        :param content_column:
            **Required when** ``conversion_mode="row"``.
            The column name whose values become ``Document.content`` for each row.
            The column must exist in the CSV header.
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

            # --- ROW MODE (strict) ---
            # Require content_column in run(); no fallback
            if not content_column:
                raise ValueError(
                    "CSVToDocument(row): 'content_column' is required in run() when conversion_mode='row'."
                )

            # Warn for large CSVs in row mode (memory consideration)
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

            # Create DictReader; if this fails, raise (no fallback)
            try:
                reader = csv.DictReader(io.StringIO(data), delimiter=self.delimiter, quotechar=self.quotechar)
            except Exception as e:
                raise RuntimeError(f"CSVToDocument(row): could not parse CSV rows for {source}: {e}") from e

            # Validate header contains content_column; strict error if missing
            header = reader.fieldnames or []
            if content_column not in header:
                raise ValueError(
                    f"CSVToDocument(row): content_column='{content_column}' not found in header "
                    f"for {source}. Available columns: {header}"
                )

            # Build documents; if a row processing fails, raise immediately (no skip)
            for i, row in enumerate(reader):
                try:
                    doc = self._build_document_from_row(
                        row=row, base_meta=merged_metadata, row_index=i, content_column=content_column
                    )
                except Exception as e:
                    raise RuntimeError(f"CSVToDocument(row): failed to process row {i} for {source}: {e}") from e
                documents.append(doc)

        return {"documents": documents}

    # ----- helpers -----
    def _safe_value(self, value: Any) -> str:
        """Normalize CSV cell values: None -> '', everything -> str."""
        return "" if value is None else str(value)

    def _build_document_from_row(
        self, row: dict[str, Any], base_meta: dict[str, Any], row_index: int, content_column: str
    ) -> Document:
        """
        Build a ``Document`` from one parsed CSV row.

        :param row: Mapping of column name to cell value for the current row
            (as produced by ``csv.DictReader``).
        :param base_meta: File-level and user-provided metadata to start from
            (for example: ``file_path``, ``encoding``).
        :param row_index: Zero-based row index in the CSV; stored as
            ``row_number`` in the output document's metadata.
        :param content_column: Column name to use for ``Document.content``.
        :returns: A ``Document`` with chosen content and merged metadata.
            Remaining row columns are added to ``meta`` with collision-safe
            keys (prefixed with ``csv_`` if needed).
        """
        row_meta = dict(base_meta)

        # content (strict: content_column must exist; validated by caller)
        content = self._safe_value(row.get(content_column))

        # merge remaining columns into meta with collision handling
        for k, v in row.items():
            if k == content_column:
                continue
            key_to_use = k
            if key_to_use in row_meta:
                # Avoid clobbering existing meta like file_path/encoding; prefix and de-dupe
                base_key = f"csv_{key_to_use}"
                key_to_use = base_key
                suffix = 1
                while key_to_use in row_meta:
                    key_to_use = f"{base_key}_{suffix}"
                    suffix += 1
            row_meta[key_to_use] = self._safe_value(v)

        row_meta["row_number"] = row_index
        return Document(content=content, meta=row_meta)
