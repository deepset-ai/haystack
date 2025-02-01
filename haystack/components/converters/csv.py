# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import os
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

    def __init__(self, encoding: str = "utf-8", store_full_path: bool = False):
        """
        Creates a CSVToDocument component.

        :param encoding:
            The encoding of the csv files to convert.
            If the encoding is specified in the metadata of a source ByteStream,
            it overrides this value.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        self.encoding = encoding
        self.store_full_path = store_full_path

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

            document = Document(content=data, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}


@component
class CSVSplitter:
    """
    A component to split a CSV document into multiple CSV documents

    It is based on:
    1) Consecutive empty rows (the "classic" approach).
    2) Optionally detecting side-by-side tables (via a BFS search for non-empty cells).

    Attributes:
        split_threshold (int): The number of consecutive empty rows required to split the CSV. Default is 2.
        delimiter (str): The delimiter used in the CSV. Default is ','.
        quotechar (str): The quote character used in the CSV. Default is '"'.
        trim_empty_rows (bool): Whether to trim leading/trailing empty rows in each block. Default is True.
        skip_errors (bool): Whether to skip documents with parsing errors. Default is True.
        split_index_meta_key (Optional[str]): Metadata key to store the split index. Default is 'csv_split_index'.
        detect_side_tables (bool): Whether to detect tables that sit side-by-side in different columns.
                                   If True, a more expensive BFS-based approach is used.
    """

    def __init__(
        self,
        split_threshold: int = 2,
        delimiter: str = ",",
        quotechar: str = '"',
        trim_empty_rows: bool = True,
        skip_errors: bool = True,
        split_index_meta_key: Optional[str] = "csv_split_index",
        detect_side_tables: bool = False,
    ):
        if split_threshold < 1:
            raise ValueError("split_threshold must be at least 1")
        self.split_threshold = split_threshold
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.trim_empty_rows = trim_empty_rows
        self.skip_errors = skip_errors
        self.split_index_meta_key = split_index_meta_key
        self.detect_side_tables = detect_side_tables

    def _split_csv_content(self, content: str) -> List[str]:
        """Splits CSV content into blocks of CSV text."""
        try:
            reader = csv.reader(
                io.StringIO(content), delimiter=self.delimiter, quotechar=self.quotechar, skipinitialspace=True
            )
            rows = list(reader)
        except csv.Error as e:
            raise ValueError(f"CSV parsing error: {str(e)}") from e

        if self.detect_side_tables:
            blocks_2d = self._split_rows_side_by_side(rows)
        else:
            blocks_2d = self._split_rows_into_blocks(rows)

        return [self._block_to_csv(b) for b in blocks_2d]

    def _split_rows_into_blocks(self, rows: List[List[str]]) -> List[List[List[str]]]:
        """
        "Classic" approach: identifies table blocks separated by consecutive empty rows.

        Controlled by `split_threshold`.
        """
        blocks, current_block, empty_count = [], [], 0

        for row in rows:
            # row is considered empty if all cells are blank
            if all(cell.strip() == "" for cell in row):
                empty_count += 1
            else:
                if empty_count >= self.split_threshold and current_block:
                    blocks.append(current_block)
                    current_block = []
                empty_count = 0
                current_block.append(row)

        if current_block:
            blocks.append(current_block)

        return self._clean_blocks(blocks)

    def _split_rows_side_by_side(self, rows: List[List[str]]) -> List[List[List[str]]]:
        """
        BFS to detect *all* distinct regions of non-empty cells, including side-by-side tables.

        Each connected group of cells is output as a separate block.
        By "connected," we mean non-empty cells that touch horizontally or vertically.
        """
        if not rows:
            return []

        # 1) Normalize row lengths so we have a 2D grid
        max_cols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < max_cols:
                r.append("")

        R, C = len(rows), max_cols
        visited = [[False] * C for _ in range(R)]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        blocks = []

        def bfs(start_r: int, start_c: int) -> List[Tuple[int, int]]:
            """Collect all connected (non-empty) cells using BFS starting from (start_r, start_c)."""
            queue = deque()
            queue.append((start_r, start_c))
            visited[start_r][start_c] = True
            connected_cells = [(start_r, start_c)]

            while queue:
                r0, c0 = queue.popleft()
                for dr, dc in directions:
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < R and 0 <= cc < C and not visited[rr][cc] and rows[rr][cc].strip() != "":
                        visited[rr][cc] = True
                        queue.append((rr, cc))
                        connected_cells.append((rr, cc))
            return connected_cells

        # 2) Find connected components of non-empty cells
        for r in range(R):
            for c in range(C):
                if not visited[r][c] and rows[r][c].strip() != "":
                    cells = bfs(r, c)
                    min_r = min(x[0] for x in cells)
                    max_r = max(x[0] for x in cells)
                    min_c = min(x[1] for x in cells)
                    max_c = max(x[1] for x in cells)
                    submatrix = []
                    for rr in range(min_r, max_r + 1):
                        row_slice = rows[rr][min_c : max_c + 1]
                        submatrix.append(row_slice)
                    blocks.append(submatrix)

        cleaned_blocks = []
        for b in blocks:
            cb = self._clean_single_block_2d(b)
            if cb:
                cleaned_blocks.append(cb)

        return cleaned_blocks

    def _clean_blocks(self, blocks: List[List[List[str]]]) -> List[List[List[str]]]:
        """
        Existing row-based cleaning (trim leading/trailing empty rows).
        """
        cleaned = []
        for block in blocks:
            if not self.trim_empty_rows:
                cleaned_block = block.copy()
            else:
                start, end = 0, len(block)
                while start < end and all(cell.strip() == "" for cell in block[start]):
                    start += 1
                while end > start and all(cell.strip() == "" for cell in block[end - 1]):
                    end -= 1
                cleaned_block = block[start:end]

            if cleaned_block:
                cleaned.append(cleaned_block)
        return cleaned

    def _clean_single_block_2d(self, block_2d: List[List[str]]) -> List[List[str]]:
        """
        For a 2D block remove empty top/bottom rows, and empty left/right columns
        """
        if not block_2d:
            return []

        if self.trim_empty_rows:
            start_r, end_r = 0, len(block_2d)
            while start_r < end_r and all(cell.strip() == "" for cell in block_2d[start_r]):
                start_r += 1
            while end_r > start_r and all(cell.strip() == "" for cell in block_2d[end_r - 1]):
                end_r -= 1
            block_2d = block_2d[start_r:end_r]

        if not block_2d:
            return []

        min_col, max_col = 0, len(block_2d[0])
        while min_col < max_col:
            if all(row[min_col].strip() == "" for row in block_2d):
                min_col += 1
            else:
                break
        while max_col > min_col:
            if all(row[max_col - 1].strip() == "" for row in block_2d):
                max_col -= 1
            else:
                break

        if min_col >= max_col:
            return []

        cleaned = [row[min_col:max_col] for row in block_2d]
        return cleaned

    def _block_to_csv(self, block: List[List[str]]) -> str:
        """Converts a 2D block of rows into CSV text."""
        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter, quotechar=self.quotechar, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(block)
        return output.getvalue().strip()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """Processes documents to split CSV content."""
        split_docs = []
        for doc in documents:
            if not doc.content:
                continue
            try:
                blocks = self._split_csv_content(doc.content)
            except Exception as e:
                if not self.skip_errors:
                    raise
                warnings.warn(f"Skipping document {doc.id}: {str(e)}", UserWarning)
                continue

            for idx, csv_text in enumerate(blocks):
                meta = doc.meta.copy()
                if self.split_index_meta_key:
                    meta[self.split_index_meta_key] = idx
                split_docs.append(Document(content=csv_text, meta=meta))

        return {"documents": split_docs}
