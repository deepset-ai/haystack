# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
from datetime import datetime
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
    from openpyxl.styles.numbers import is_datetime

with LazyImport("Run 'pip install tabulate'") as tabulate_import:
    from tabulate import tabulate  # noqa: F401 # the library is used but not directly referenced

# The `pandas.read_excel` options that decide which cells end up where in the DataFrame.
_POSITION_KWARGS = ("skiprows", "nrows", "skipfooter", "usecols", "names", "index_col")
# Placeholder written into non-empty cells so pandas itself can resolve where they land after the
# positional read_excel_kwargs (skiprows/names/usecols/...) have been applied.
_CELL_TAG = "__haystack_cell__"


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
            By default, the converter passes `keep_default_na=False` and `na_values=[""]`, so literal text like
            "NA" and "N/A" is kept and only empty cells are treated as missing.
            If you set either `keep_default_na` or `na_values`, these defaults aren't applied and pandas' own
            behavior is used: `na_values` alone adds to pandas' default NA strings, and `keep_default_na=False`
            alone doesn't treat empty cells as missing unless `""` is in `na_values`.
            Set `keep_default_na=True` to restore pandas' default NA parsing.
            Custom parsing options may cause conversion errors or unexpected output.
            For example, `skiprows` can produce incorrect row numbers, and `index_col`
            can cause values to change or the entire file to be skipped.
            Validate the output on representative inputs when using these options.
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

                if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
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

    @staticmethod
    def _get_worksheet(workbook: Any, sheet_key: str | int | None) -> Any:
        if isinstance(sheet_key, int):
            return workbook.worksheets[sheet_key]
        if sheet_key is None:
            return workbook.active
        return workbook[sheet_key]

    @staticmethod
    def _date_only_cells(worksheet: Any) -> dict[tuple[int, int], str]:
        """
        Find cells that hold a date and whose number format shows no time.

        pandas reads such a cell as a datetime at midnight, and because the converter reads with `header=None` every
        column has object dtype, so the cell would be written as "2025-03-14 00:00:00". Returns the ISO date text to
        write instead, keyed by 0-based Excel coordinates.
        """
        cells: dict[tuple[int, int], str] = {}
        for row in worksheet.iter_rows():
            for cell in row:
                value = cell.value
                if (
                    isinstance(value, datetime)
                    and value.time() == datetime.min.time()
                    and is_datetime(cell.number_format) == "date"
                ):
                    cells[(cell.row - 1, cell.column - 1)] = value.date().isoformat()
        return cells

    def _map_cells_to_dataframe(
        self, workbook: Any, cells_by_sheet: dict[str | int | None, dict[tuple[int, int], Any]]
    ) -> dict[str | int | None, dict[tuple[int, int], Any]]:
        """Re-key 0-based Excel cell coordinates to their positions in the filtered DataFrame."""
        for sheet_key in cells_by_sheet:
            for row in self._get_worksheet(workbook, sheet_key).iter_rows():
                for cell in row:
                    # Empty cells must stay empty: pandas sizes the sheet by its non-empty cells, and the placeholder
                    # cells of merged ranges are read-only.
                    if cell.value not in (None, ""):
                        cell.value = f"{_CELL_TAG}{cell.row - 1},{cell.column - 1}"
        tagged_bytes = io.BytesIO()
        workbook.save(tagged_bytes)
        position_kwargs = {
            key: self.read_excel_kwargs[key] for key in _POSITION_KWARGS if key in self.read_excel_kwargs
        }
        tagged_sheets = pd.read_excel(
            io=tagged_bytes, sheet_name=self.sheet_name, header=None, engine="openpyxl", **position_kwargs
        )
        if isinstance(tagged_sheets, pd.DataFrame):
            tagged_sheets = {self.sheet_name: tagged_sheets}

        mapped_cells = {}
        for sheet_key, cell_values in cells_by_sheet.items():
            positions = {}
            for row_idx, values in enumerate(tagged_sheets[sheet_key].itertuples(index=False, name=None)):
                for col_idx, value in enumerate(values):
                    if isinstance(value, str) and value.startswith(_CELL_TAG):
                        excel_row, excel_col = value.removeprefix(_CELL_TAG).split(",")
                        positions[(int(excel_row), int(excel_col))] = (row_idx, col_idx)
            mapped_cells[sheet_key] = {
                positions[cell]: cell_value for cell, cell_value in cell_values.items() if cell in positions
            }
        return mapped_cells

    def _collect_cell_overrides(
        self, file_bytes: io.BytesIO, sheet_to_dataframe: dict[str | int | None, pd.DataFrame]
    ) -> dict[str | int | None, dict[tuple[int, int], tuple[str | None, str | None]]]:
        """
        Find the cells whose text must differ from what pandas read.

        Returns, per sheet, DataFrame positions mapped to (date text or None, hyperlink target or None). The workbook
        is only loaded with openpyxl when it is needed: for hyperlinks, or for date cells whose number format has no
        time part (pandas gives those a midnight time).
        """
        has_midnight_dates = any(
            isinstance(value, datetime) and pd.notna(value) and value.time() == datetime.min.time()
            for df in sheet_to_dataframe.values()
            for value in df.to_numpy().ravel()
        )
        cell_overrides: dict[str | int | None, dict[tuple[int, int], tuple[str | None, str | None]]] = {}
        if self.link_format == "none" and not has_midnight_dates:
            return cell_overrides

        file_bytes.seek(0)
        wb = openpyxl.load_workbook(file_bytes, data_only=True)
        for sheet_key in sheet_to_dataframe:
            ws = self._get_worksheet(wb, sheet_key)
            overrides: dict[tuple[int, int], tuple[str | None, str | None]] = {}
            if has_midnight_dates:
                for coordinates, iso_date in self._date_only_cells(ws).items():
                    overrides[coordinates] = (iso_date, None)
            if self.link_format != "none":
                for row in ws.iter_rows():
                    for cell in row:
                        if cell.hyperlink and cell.hyperlink.target:
                            # Zero-based Excel coordinates; translated to filtered positions below.
                            link_cell = (cell.row - 1, cell.column - 1)
                            overrides[link_cell] = (overrides.get(link_cell, (None, None))[0], cell.hyperlink.target)
            cell_overrides[sheet_key] = overrides
        if any(cell_overrides.values()) and any(key in self.read_excel_kwargs for key in _POSITION_KWARGS):
            cell_overrides = self._map_cells_to_dataframe(wb, cell_overrides)
        wb.close()
        return cell_overrides

    def _apply_cell_overrides(
        self, df: pd.DataFrame, overrides: dict[tuple[int, int], tuple[str | None, str | None]]
    ) -> None:
        """Write date text and hyperlinks into the DataFrame cells at the given positions."""
        for (row_idx, col_idx), (date_text, url) in overrides.items():
            if row_idx >= len(df) or col_idx >= len(df.columns):
                continue
            # Text must be assignable to numeric and other typed columns.
            column = df.columns[col_idx]
            if df[column].dtype != object:
                df[column] = df[column].astype(object)
            if date_text is not None:
                df.iat[row_idx, col_idx] = date_text
            if url is not None:
                cell_value: Any = df.iat[row_idx, col_idx]
                text = str(cell_value) if pd.notna(cell_value) else ""
                if self.link_format == "markdown":
                    df.iat[row_idx, col_idx] = f"[{text}]({url})"
                else:
                    df.iat[row_idx, col_idx] = f"{text} ({url})"

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
        if "keep_default_na" not in self.read_excel_kwargs and "na_values" not in self.read_excel_kwargs:
            resolved_read_excel_kwargs.update(keep_default_na=False, na_values=[""])
        sheet_to_dataframe = pd.read_excel(io=file_bytes, **resolved_read_excel_kwargs)
        if isinstance(sheet_to_dataframe, pd.DataFrame):
            sheet_to_dataframe = {self.sheet_name: sheet_to_dataframe}

        cell_overrides = self._collect_cell_overrides(file_bytes, sheet_to_dataframe)

        updated_sheet_to_dataframe = {}
        for key in sheet_to_dataframe:
            df = sheet_to_dataframe[key]
            # Row starts at 1 in Excel
            df.index = df.index + 1
            # Excel column names are Alphabet Characters
            header = self._generate_excel_column_names(df.shape[1])
            df.columns = header

            self._apply_cell_overrides(df, cell_overrides.get(key, {}))
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
