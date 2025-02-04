import csv
import io
import warnings
from collections import deque
from typing import Deque, Dict, List, Tuple

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class CSVDocumentSplitter:
    """
    A component to split a CSV document into multiple CSV documents.

    It supports splitting based on consecutive empty rows (classic approach)
    or detecting side-by-side tables using a BFS search for non-empty cells.
    """

    def __init__(
        self,
        split_threshold: int = 2,
        delimiter: str = ",",
        quotechar: str = '"',
        trim_empty_rows: bool = True,
        skip_errors: bool = True,
        detect_side_tables: bool = False,
    ) -> None:
        """
        Initialize the CSVDocumentSplitter.

        :param split_threshold: The number of consecutive empty rows required to split the CSV (must be at least 1).
        :param delimiter: The delimiter used in the CSV. Default is ','.
        :param quotechar: The quote character used in the CSV. Default is '"'.
        :param trim_empty_rows: Whether to trim leading/trailing empty rows in each block. Default is True.
        :param skip_errors: Whether to skip documents with parsing errors. Default is True.
        :param detect_side_tables: Whether to detect tables that sit side-by-side in different columns.
                                   If True, a more expensive BFS-based approach is used.
        """
        if split_threshold < 1:
            raise ValueError("split_threshold must be at least 1")
        self.split_threshold = split_threshold
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.trim_empty_rows = trim_empty_rows
        self.skip_errors = skip_errors
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
        blocks: List[List[List[str]]] = []
        current_block: List[List[str]] = []
        empty_count: int = 0

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
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        R, C = len(rows), max_cols
        visited = [[False] * C for _ in range(R)]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        blocks = []

        def bfs(start_r: int, start_c: int) -> List[Tuple[int, int]]:
            """Collect all connected (non-empty) cells using BFS starting from (start_r, start_c)."""
            queue: Deque[Tuple[int, int]] = deque()
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
        For a 2D block remove empty top/bottom rows, and empty left/right columns.
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
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Processes documents to split CSV content."""
        split_docs: List[Document] = []
        for doc in documents:
            if not doc.content:
                continue
            try:
                blocks = self._split_csv_content(doc.content)
            except Exception as e:
                if not self.skip_errors:
                    raise
                logger.warning(f"Skipping document {doc.id}: {str(e)}")
                continue

            for idx, csv_text in enumerate(blocks):
                meta = doc.meta.copy()
                meta["split_id"] = idx
                split_docs.append(Document(content=csv_text, meta=meta))

        return {"documents": split_docs}
