# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install openpyxl'") as xlsx_import:
    import openpyxl  # pylint: disable=unused-import # the library is used but not directly referenced

with LazyImport("Run 'pip install tabulate'") as tabulate_import:
    from tabulate import tabulate  # pylint: disable=unused-import # the library is used but not directly referenced


@component
class XLSXToDocument:
    """
    Converts XLSX files to Documents.

    By default, it reads all work sheets into CSV format.

    ### Usage example

    ```python
    from haystack.components.converters.xlsx import XLSXToDocument

    converter = XLSXToDocument()
    results = converter.run(sources=["sample.xlsx"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'col1,col2\now1,row1\nrow2row2\n'
    ```
    """

    def __init__(
        self,
        table_format: Literal["csv", "markdown"] = "csv",
        sheet_name: Union[str, int, List[Union[str, int]], None] = None,
        read_excel_kwargs: Optional[Dict[str, Any]] = None,
        table_format_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a XLSXToDocument component.

        :param table_format: The format to convert the Excel file to.
        :param sheet_name: The name of the sheet to read. If None, all sheets are read.
        :param read_excel_kwargs: Additional arguments to pass to `pandas.read_excel`.
            See https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas-read-excel
        :param table_format_kwargs: Additional keyword arguments to pass to the table format function.
        """
        xlsx_import.check()
        self.table_format = table_format
        if table_format not in ["csv", "markdown"]:
            raise ValueError(f"Unsupported export format: {table_format}. Choose either 'csv' or 'markdown'.")
        if table_format == "markdown":
            tabulate_import.check()
        self.sheet_name = sheet_name
        self.read_excel_kwargs = read_excel_kwargs or {}
        self.table_format_kwargs = table_format_kwargs or {}

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
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

        for source, metadata in zip(sources, meta_list):
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
            for table, excel_metadata in zip(tables, tables_metadata):
                merged_metadata = {**bytestream.meta, **metadata, **excel_metadata}
                document = Document(content=table, meta=merged_metadata)
                documents.append(document)

        return {"documents": documents}

    @staticmethod
    def _generate_excel_column_names(n_cols: int) -> List[str]:
        result = []
        for i in range(n_cols):
            col_name = ""
            num = i
            while num >= 0:
                col_name = chr(num % 26 + 65) + col_name
                num = num // 26 - 1
            result.append(col_name)
        return result

    def _extract_tables(self, bytestream: ByteStream) -> Tuple[List[str], List[Dict]]:
        """
        Extract tables from a Excel file.
        """
        resolved_read_excel_kwargs = {
            **self.read_excel_kwargs,
            "sheet_name": self.sheet_name,
            "header": None,  # Don't assign any pandas column labels
            "engine": "openpyxl",  # Use openpyxl as the engine to read the Excel file
        }
        dict_or_df = pd.read_excel(io=io.BytesIO(bytestream.data), **resolved_read_excel_kwargs)
        if isinstance(dict_or_df, pd.DataFrame):
            dict_or_df = {self.sheet_name: dict_or_df}

        for key in dict_or_df:
            df = dict_or_df[key]
            # Row starts at 1 in Excel
            df.index = df.index + 1
            # Excel column names are Alphabet Characters
            header = self._generate_excel_column_names(df.shape[1])
            df.columns = header
            dict_or_df[key] = df

        tables = []
        metadata = []
        for key in dict_or_df:
            if self.table_format == "csv":
                resolved_kwargs = {"index": True, "header": True, "lineterminator": "\n", **self.table_format_kwargs}
                tables.append(dict_or_df[key].to_csv(**resolved_kwargs))
            else:
                resolved_kwargs = {
                    "index": True,
                    "headers": dict_or_df[key].columns,
                    "tablefmt": "pipe",
                    **self.table_format_kwargs,
                }
                # to_markdown uses tabulate
                tables.append(dict_or_df[key].to_markdown(**resolved_kwargs))
            # add sheet_name to metadata
            metadata.append({"xlsx": {"sheet_name": key}})
        return tables, metadata
