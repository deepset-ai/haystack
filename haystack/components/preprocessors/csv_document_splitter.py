# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args

from haystack import Document, component, logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pandas'") as pandas_import:
    import pandas as pd

logger = logging.getLogger(__name__)

SplitMode = Literal["threshold", "row-wise"]


@component
class CSVDocumentSplitter:
    """
    A component for splitting CSV documents into sub-tables based on split arguments.

    The splitter supports two modes of operation:
    - identify consecutive empty rows or columns that exceed a given threshold
    and uses them as delimiters to segment the document into smaller tables.
    - split each row into a separate sub-table, represented as a Document.

    """

    def __init__(
        self,
        row_split_threshold: Optional[int] = 2,
        column_split_threshold: Optional[int] = 2,
        read_csv_kwargs: Optional[Dict[str, Any]] = None,
        split_mode: SplitMode = "threshold",
    ) -> None:
        """
        Initializes the CSVDocumentSplitter component.

        :param row_split_threshold: The minimum number of consecutive empty rows required to trigger a split.
        :param column_split_threshold: The minimum number of consecutive empty columns required to trigger a split.
        :param read_csv_kwargs: Additional keyword arguments to pass to `pandas.read_csv`.
            By default, the component with options:
            - `header=None`
            - `skip_blank_lines=False` to preserve blank lines
            - `dtype=object` to prevent type inference (e.g., converting numbers to floats).
            See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for more information.
        :param split_mode:
            If `threshold`, the component will split the document based on the number of
            consecutive empty rows or columns that exceed the `row_split_threshold` or `column_split_threshold`.
            If `row-wise`, the component will split each row into a separate sub-table.
        """
        pandas_import.check()
        if split_mode not in get_args(SplitMode):
            raise ValueError(
                f"Split mode '{split_mode}' not recognized. Choose one among: {', '.join(get_args(SplitMode))}."
            )
        if row_split_threshold is not None and row_split_threshold < 1:
            raise ValueError("row_split_threshold must be greater than 0")

        if column_split_threshold is not None and column_split_threshold < 1:
            raise ValueError("column_split_threshold must be greater than 0")

        if row_split_threshold is None and column_split_threshold is None:
            raise ValueError("At least one of row_split_threshold or column_split_threshold must be specified.")

        self.row_split_threshold = row_split_threshold
        self.column_split_threshold = column_split_threshold
        self.read_csv_kwargs = read_csv_kwargs or {}
        self.split_mode = split_mode

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Processes and splits a list of CSV documents into multiple sub-tables.

        **Splitting Process:**
        1. Applies a row-based split if `row_split_threshold` is provided.
        2. Applies a column-based split if `column_split_threshold` is provided.
        3. If both thresholds are specified, performs a recursive split by rows first, then columns, ensuring
           further fragmentation of any sub-tables that still contain empty sections.
        4. Sorts the resulting sub-tables based on their original positions within the document.

        :param documents: A list of Documents containing CSV-formatted content.
            Each document is assumed to contain one or more tables separated by empty rows or columns.

        :return:
            A dictionary with a key `"documents"`, mapping to a list of new `Document` objects,
            each representing an extracted sub-table from the original CSV.
            The metadata of each document includes:
                - A field `source_id` to track the original document.
                - A field `row_idx_start` to indicate the starting row index of the sub-table in the original table.
                - A field `col_idx_start` to indicate the starting column index of the sub-table in the original table.
                - A field `split_id` to indicate the order of the split in the original document.
                - All other metadata copied from the original document.

        - If a document cannot be processed, it is returned unchanged.
        - The `meta` field from the original document is preserved in the split documents.
        """
        if len(documents) == 0:
            return {"documents": documents}

        resolved_read_csv_kwargs = {"header": None, "skip_blank_lines": False, "dtype": object, **self.read_csv_kwargs}

        split_documents = []
        split_dfs = []
        for document in documents:
            try:
                df = pd.read_csv(StringIO(document.content), **resolved_read_csv_kwargs)  # type: ignore
            except Exception as e:
                logger.error(f"Error processing document {document.id}. Keeping it, but skipping splitting. Error: {e}")
                split_documents.append(document)
                continue

            if self.split_mode == "row-wise":
                # each row is a separate sub-table
                split_dfs = self._split_by_row(df=df)

            elif self.split_mode == "threshold":
                if self.row_split_threshold is not None and self.column_split_threshold is None:
                    # split by rows
                    split_dfs = self._split_dataframe(df=df, split_threshold=self.row_split_threshold, axis="row")
                elif self.column_split_threshold is not None and self.row_split_threshold is None:
                    # split by columns
                    split_dfs = self._split_dataframe(df=df, split_threshold=self.column_split_threshold, axis="column")
                else:
                    # recursive split
                    split_dfs = self._recursive_split(
                        df=df,
                        row_split_threshold=self.row_split_threshold,  # type: ignore
                        column_split_threshold=self.column_split_threshold,  # type: ignore
                    )

            # check if no sub-tables were found
            if len(split_dfs) == 0:
                logger.warning(
                    "No sub-tables found while splitting CSV Document with id {doc_id}. Skipping document.",
                    doc_id=document.id,
                )
                continue

            # Sort split_dfs first by row index, then by column index
            split_dfs.sort(key=lambda dataframe: (dataframe.index[0], dataframe.columns[0]))

            for split_id, split_df in enumerate(split_dfs):
                split_documents.append(
                    Document(
                        content=split_df.to_csv(index=False, header=False, lineterminator="\n"),
                        meta={
                            **document.meta.copy(),
                            "source_id": document.id,
                            "row_idx_start": int(split_df.index[0]),
                            "col_idx_start": int(split_df.columns[0]),
                            "split_id": split_id,
                        },
                    )
                )

        return {"documents": split_documents}

    @staticmethod
    def _find_split_indices(
        df: "pd.DataFrame", split_threshold: int, axis: Literal["row", "column"]
    ) -> List[Tuple[int, int]]:
        """
        Finds the indices of consecutive empty rows or columns in a DataFrame.

        :param df: DataFrame to split.
        :param split_threshold: Minimum number of consecutive empty rows or columns to trigger a split.
        :param axis: Axis along which to find empty elements. Either "row" or "column".
        :return: List of indices where consecutive empty rows or columns start.
        """
        if axis == "row":
            empty_elements = df[df.isnull().all(axis=1)].index.tolist()
        else:
            empty_elements = df.columns[df.isnull().all(axis=0)].tolist()

        # If no empty elements found, return empty list
        if len(empty_elements) == 0:
            return []

        # Identify groups of consecutive empty elements
        split_indices = []
        consecutive_count = 1
        start_index = empty_elements[0]

        for i in range(1, len(empty_elements)):
            if empty_elements[i] == empty_elements[i - 1] + 1:
                consecutive_count += 1
            else:
                if consecutive_count >= split_threshold:
                    split_indices.append((start_index, empty_elements[i - 1]))
                consecutive_count = 1
                start_index = empty_elements[i]

        # Handle the last group of consecutive elements
        if consecutive_count >= split_threshold:
            split_indices.append((start_index, empty_elements[-1]))

        return split_indices

    def _split_dataframe(
        self, df: "pd.DataFrame", split_threshold: int, axis: Literal["row", "column"]
    ) -> List["pd.DataFrame"]:
        """
        Splits a DataFrame into sub-tables based on consecutive empty rows or columns exceeding `split_threshold`.

        :param df: DataFrame to split.
        :param split_threshold: Minimum number of consecutive empty rows or columns to trigger a split.
        :param axis: Axis along which to split. Either "row" or "column".
        :return: List of split DataFrames.
        """
        # Find indices of consecutive empty rows or columns
        split_indices = self._find_split_indices(df=df, split_threshold=split_threshold, axis=axis)

        # If no split_indices are found, return the original DataFrame
        if len(split_indices) == 0:
            return [df]

        # Split the DataFrame at identified indices
        sub_tables = []
        table_start_idx = 0
        df_length = df.shape[0] if axis == "row" else df.shape[1]
        for empty_start_idx, empty_end_idx in split_indices + [(df_length, df_length)]:
            # Avoid empty splits
            if empty_start_idx - table_start_idx >= 1:
                if axis == "row":
                    sub_table = df.iloc[table_start_idx:empty_start_idx]
                else:
                    sub_table = df.iloc[:, table_start_idx:empty_start_idx]
                if not sub_table.empty:
                    sub_tables.append(sub_table)
            table_start_idx = empty_end_idx + 1

        return sub_tables

    def _recursive_split(
        self, df: "pd.DataFrame", row_split_threshold: int, column_split_threshold: int
    ) -> List["pd.DataFrame"]:
        """
        Recursively splits a DataFrame.

        Recursively splits a DataFrame first by empty rows, then by empty columns, and repeats the process
        until no more splits are possible. Returns a list of DataFrames, each representing a fully separated sub-table.

        :param df: A Pandas DataFrame representing a table (or multiple tables) extracted from a CSV.
        :param row_split_threshold: The minimum number of consecutive empty rows required to trigger a split.
        :param column_split_threshold: The minimum number of consecutive empty columns to trigger a split.
        """

        # Step 1: Split by rows
        new_sub_tables = self._split_dataframe(df=df, split_threshold=row_split_threshold, axis="row")

        # Step 2: Split by columns
        final_tables = []
        for table in new_sub_tables:
            final_tables.extend(self._split_dataframe(df=table, split_threshold=column_split_threshold, axis="column"))

        # Step 3: Recursively reapply splitting checked by whether any new empty rows appear after column split
        result = []
        for table in final_tables:
            # Check if there are consecutive rows >= row_split_threshold now present
            if len(self._find_split_indices(df=table, split_threshold=row_split_threshold, axis="row")) > 0:
                result.extend(
                    self._recursive_split(
                        df=table, row_split_threshold=row_split_threshold, column_split_threshold=column_split_threshold
                    )
                )
            else:
                result.append(table)

        return result

    def _split_by_row(self, df: "pd.DataFrame") -> List["pd.DataFrame"]:
        """Split each CSV row into a separate subtable"""
        split_dfs = []
        for idx, row in enumerate(df.itertuples(index=False)):
            split_df = pd.DataFrame(row).T
            split_df.index = [idx]  # Set the index of the new DataFrame to idx
            split_dfs.append(split_df)
        return split_dfs
