# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
from pathlib import Path
from typing import Any, Literal

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install pandas openpyxl'") as pandas_xlsx_import:
    import openpyxl
    import pandas as pd

with LazyImport("Run 'pip install tabulate'") as tabulate_import:
    from tabulate import tabulate  # noqa: F401 # the library is used but not directly referenced


def _get_position_token_parts(comment: Any) -> tuple[str, str]:
    """Return token parts that pandas will not treat as comment markers."""
    for prefix in ("__HAYSTACK_XLSX_CELL_POSITION__", "HAYSTACKXLSXCELLPOSITION", "cellposition"):
        if not isinstance(comment, str) or not comment or comment not in prefix:
            break
    for separator in ("_", ":", "~"):
        if not isinstance(comment, str) or not comment or comment not in separator:
            return prefix, separator
    return prefix, "~"


def _make_position_token(
    row_idx: int, col_idx: int, value: Any, comment: Any, token_prefix: str, token_separator: str
) -> str:
    """Create a cell value that preserves pandas comment semantics while identifying its source coordinate."""
    token = f"{token_prefix}{row_idx}{token_separator}{col_idx}"
    if isinstance(comment, str) and comment and isinstance(value, str) and comment in value:
        comment_idx = value.find(comment)
        if comment_idx == 0:
            return f"{comment}{token}"
        return f"{value[:comment_idx]}{token}{value[comment_idx:]}"
    return token


def _parse_position_token(value: Any, token_prefix: str, token_separator: str) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    token_idx = value.rfind(token_prefix)
    if token_idx == -1:
        return None

    row, separator, column = value[token_idx + len(token_prefix) :].partition(token_separator)
    if not separator:
        return None
    try:
        return int(row), int(column)
    except ValueError:
        return None


@component
class XLSXToDocument:
    """
    Converts XLSX (Excel) files into Documents.

    Supports reading data from specific sheets or all sheets in the Excel file. If all sheets are read, a Document is
    created for each sheet. The content of the Document is the table which can be saved in CSV or Markdown format.

    ### Usage example

    ```python
    from haystack.components.converters.xlsx import XLSXToDocument
    from datetime import datetime

    converter = XLSXToDocument()
    results = converter.run(
        sources=["test/test_files/xlsx/basic_tables_two_sheets.xlsx"], meta={"date_added": datetime.now().isoformat()}
    )
    documents = results["documents"]

    print(documents[0].content)
    # >> ",A,B\\n1,col_a,col_b\\n2,1.5,test\\n"
    ```
    """

    def __init__(
        self,
        table_format: Literal["csv", "markdown"] = "csv",
        sheet_name: str | int | list[str | int] | None = None,
        read_excel_kwargs: dict[str, Any] | None = None,
        table_format_kwargs: dict[str, Any] | None = None,
        *,
        link_format: Literal["markdown", "plain", "none"] = "none",
        store_full_path: bool = False,
    ) -> None:
        """
        Creates a XLSXToDocument component.

        :param table_format: The format to convert the Excel file to.
        :param sheet_name: The name of the sheet to read. If None, all sheets are read.
        :param read_excel_kwargs: Additional arguments to pass to `pandas.read_excel`.
            See https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas-read-excel
        :param table_format_kwargs: Additional keyword arguments to pass to the table format function.
            - If `table_format` is "csv", these arguments are passed to `pandas.DataFrame.to_csv`.
              See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas-dataframe-to-csv
            - If `table_format` is "markdown", these arguments are passed to `pandas.DataFrame.to_markdown`.
              See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html#pandas-dataframe-to-markdown
        :param link_format: The format for link output. Possible options:
            - `"markdown"`: `[text](url)`
            - `"plain"`: `text (url)`
            - `"none"`: Only the text is extracted, link addresses are ignored.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        pandas_xlsx_import.check()
        self.table_format = table_format
        if table_format not in ["csv", "markdown"]:
            raise ValueError(f"Unsupported export format: {table_format}. Choose either 'csv' or 'markdown'.")
        if link_format not in ("markdown", "plain", "none"):
            msg = f"Unknown link format '{link_format}'. Supported formats are: 'markdown', 'plain', 'none'"
            raise ValueError(msg)
        if table_format == "markdown":
            tabulate_import.check()
        self.link_format = link_format
        self.sheet_name = sheet_name
        self.read_excel_kwargs = read_excel_kwargs or {}
        self.table_format_kwargs = table_format_kwargs or {}
        self.store_full_path = store_full_path

    @component.output_types(documents=list[Document])
    def run(
        self, sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, list[Document]]:
        """
        Converts a XLSX file to a Document.

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

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            try:
                tables, tables_metadata = self._extract_tables(bytestream)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to a Document, skipping. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

            # Loop over tables and create a Document for each table
            for table, excel_metadata in zip(tables, tables_metadata, strict=True):
                merged_metadata = {**bytestream.meta, **metadata, **excel_metadata}

                if not self.store_full_path and "file_path" in bytestream.meta:
                    file_path = bytestream.meta["file_path"]
                    merged_metadata["file_path"] = os.path.basename(file_path)

                document = Document(content=table, meta=merged_metadata)
                documents.append(document)

        return {"documents": documents}

    @staticmethod
    def _generate_excel_column_names(n_cols: int) -> list[str]:
        result = []
        for i in range(n_cols):
            col_name = ""
            num = i
            while num >= 0:
                col_name = chr(num % 26 + 65) + col_name
                num = num // 26 - 1
            result.append(col_name)
        return result

    def _apply_hyperlinks(
        self,
        dataframe: pd.DataFrame,
        hyperlinks: dict[tuple[int, int], str],
        cell_positions: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        for (original_row_idx, original_col_idx), url in hyperlinks.items():
            position = cell_positions.get((original_row_idx, original_col_idx))
            if position is None:
                continue
            row_idx, col_idx = position
            if row_idx < len(dataframe) and col_idx < len(dataframe.columns):
                cell_value = dataframe.iat[row_idx, col_idx]
                text = str(cell_value) if pd.notna(cell_value) else ""
                # Hyperlink text must be assignable to numeric and other typed columns.
                column = dataframe.columns[col_idx]
                if dataframe[column].dtype != object:
                    dataframe[column] = dataframe[column].astype(object)
                if self.link_format == "markdown":
                    dataframe.iat[row_idx, col_idx] = f"[{text}]({url})"
                else:
                    dataframe.iat[row_idx, col_idx] = f"{text} ({url})"

    @staticmethod
    def _get_worksheet(workbook: Any, sheet_key: str | int | None) -> Any:
        if isinstance(sheet_key, int):
            return workbook.worksheets[sheet_key]
        if sheet_key is None:
            return workbook.active
        return workbook[sheet_key]

    def _get_cell_positions(
        self, workbook: Any, sheet_keys: list[str | int | None]
    ) -> dict[Any, dict[tuple[int, int], tuple[int, int]]]:
        """Map original zero-based cell coordinates to positions in the filtered DataFrame."""
        comment = self.read_excel_kwargs.get("comment")
        token_prefix, token_separator = _get_position_token_parts(comment)
        for sheet_key in sheet_keys:
            worksheet = self._get_worksheet(workbook, sheet_key)
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.value = _make_position_token(
                        cell.row - 1, cell.column - 1, cell.value, comment, token_prefix, token_separator
                    )

        # Use pandas itself to resolve skiprows, names, usecols, and comment. The token values identify which
        # original cells survived filtering without reimplementing pandas' option interactions.
        cell_map_bytes = io.BytesIO()
        workbook.save(cell_map_bytes)
        cell_map_bytes.seek(0)
        mapping_kwargs: dict[str, Any] = {
            key: self.read_excel_kwargs[key]
            for key in ("skiprows", "nrows", "skipfooter", "usecols", "names", "comment", "index_col")
            if key in self.read_excel_kwargs
        }
        mapping_kwargs.update({"sheet_name": self.sheet_name, "header": None, "engine": "openpyxl"})
        cells_by_sheet = pd.read_excel(io=cell_map_bytes, **mapping_kwargs)
        if isinstance(cells_by_sheet, pd.DataFrame):
            cells_by_sheet = {self.sheet_name: cells_by_sheet}

        cell_positions_by_sheet = {}
        for sheet_key, dataframe in cells_by_sheet.items():
            cell_positions = {}
            for row_idx, row in enumerate(dataframe.itertuples(index=False, name=None)):
                for col_idx, value in enumerate(row):
                    position = _parse_position_token(value, token_prefix, token_separator)
                    if position is not None:
                        cell_positions[position] = (row_idx, col_idx)
            cell_positions_by_sheet[sheet_key] = cell_positions
        return cell_positions_by_sheet

    def _extract_tables(self, bytestream: ByteStream) -> tuple[list[str], list[dict]]:
        """
        Extract tables from an Excel file.
        """
        file_bytes = io.BytesIO(bytestream.data)
        resolved_read_excel_kwargs = {
            **self.read_excel_kwargs,
            "sheet_name": self.sheet_name,
            "header": None,  # Don't assign any pandas column labels
            "engine": "openpyxl",  # Use openpyxl as the engine to read the Excel file
        }
        sheet_to_dataframe = pd.read_excel(io=file_bytes, **resolved_read_excel_kwargs)
        if isinstance(sheet_to_dataframe, pd.DataFrame):
            sheet_to_dataframe = {self.sheet_name: sheet_to_dataframe}

        # If link extraction is enabled, load the workbook with openpyxl to read hyperlinks
        hyperlinks_by_sheet: dict[str | int | None, dict[tuple[int, int], str]] = {}
        cell_positions_by_sheet: dict[str | int | None, dict[tuple[int, int], tuple[int, int]]] = {}
        if self.link_format != "none":
            file_bytes.seek(0)
            wb = openpyxl.load_workbook(file_bytes, data_only=True)
            for sheet_key in sheet_to_dataframe:
                ws = self._get_worksheet(wb, sheet_key)
                cell_links: dict[tuple[int, int], str] = {}
                for row in ws.iter_rows():
                    for cell in row:
                        if cell.hyperlink and cell.hyperlink.target:
                            # Store zero-based Excel coordinates; they are translated to filtered positions below.
                            cell_links[(cell.row - 1, cell.column - 1)] = cell.hyperlink.target
                hyperlinks_by_sheet[sheet_key] = cell_links
            cell_positions_by_sheet = self._get_cell_positions(wb, list(sheet_to_dataframe))
            wb.close()

        updated_sheet_to_dataframe = {}
        for key in sheet_to_dataframe:
            df = sheet_to_dataframe[key]
            # Row starts at 1 in Excel
            df.index = df.index + 1
            # Apply hyperlinks to cell values before replacing the original column indices with Excel column names.
            if key in hyperlinks_by_sheet:
                self._apply_hyperlinks(df, hyperlinks_by_sheet[key], cell_positions_by_sheet[key])
            # Excel column names are Alphabet Characters
            header = self._generate_excel_column_names(df.shape[1])
            df.columns = header

            updated_sheet_to_dataframe[key] = df

        tables = []
        metadata = []
        for key, value in updated_sheet_to_dataframe.items():
            if self.table_format == "csv":
                resolved_kwargs = {"index": True, "header": True, "lineterminator": "\n", **self.table_format_kwargs}
                tables.append(value.to_csv(**resolved_kwargs))
            else:
                resolved_kwargs = {
                    "index": True,
                    "headers": value.columns,
                    "tablefmt": "pipe",
                    "missingval": "",
                    **self.table_format_kwargs,
                }
                if resolved_kwargs["tablefmt"] == "pipe":
                    value = value.replace(
                        {
                            r"\r\n|\r|\n": " ",  # keep in-cell line breaks from creating extra Markdown rows
                            r"(\\*)\|": r"\1\1\\|",  # escape pipes but preserve any preceding literal backslashes
                        },
                        regex=True,
                    )

                # to_markdown uses tabulate, whose missingval only covers None: a NaN
                # reaches the formatter as a number and is written out as "nan". Replace
                # the empty cells with None so an empty cell reads as empty, the way
                # to_csv already writes it, and so missingval keeps working.
                filled = value.astype(object).where(value.notna(), None)
                tables.append(filled.to_markdown(**resolved_kwargs))
            # add sheet_name to metadata
            metadata.append({"xlsx": {"sheet_name": key}})
        return tables, metadata
