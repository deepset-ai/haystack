# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from typing import Dict, List, Literal, Optional

from haystack import Document, component, logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pandas'") as pandas_import:
    import pandas as pd

logger = logging.getLogger(__name__)


@component
class CSVDocumentSplitter:
    """
    A component for splitting CSV documents
    """

    def __init__(self, row_split_threshold: Optional[int] = 2, column_split_threshold: Optional[int] = 2) -> None:
        """
        Initializes the CSVDocumentSplitter component.

        :param row_split_threshold:
            The minimum number of consecutive empty rows required to trigger a split.
            A higher threshold prevents excessive splitting, while a lower threshold may lead
            to more fragmented sub-tables.
        :param column_split_threshold:
            The minimum number of consecutive empty columns required to trigger a split.
            A higher threshold prevents excessive splitting, while a lower threshold may lead
            to more fragmented sub-tables.
        """
        pandas_import.check()
        if row_split_threshold is not None and row_split_threshold < 1:
            raise ValueError("row_split_threshold must be greater than 0")

        if column_split_threshold is not None and column_split_threshold < 1:
            raise ValueError("column_split_threshold must be greater than 0")

        if row_split_threshold is None and column_split_threshold is None:
            raise ValueError("At least one of row_split_threshold or column_split_threshold must be specified.")

        self.row_split_threshold = row_split_threshold
        self.column_split_threshold = column_split_threshold

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Processes and splits a list of CSV documents into multiple sub-tables.

        **Splitting Process:**
        1. Row Splitting: Detects empty rows and separates tables stacked vertically.
        2. Column Splitting: Detects empty columns and separates side-by-side tables.
        3. Recursive Row Check: After splitting by columns, it checks for new row splits
           introduced by the column split.

        :param documents: A list of Documents containing CSV-formatted content.
            Each document is assumed to contain one or more tables separated by empty rows or columns.

        :return:
            A dictionary with a key `"documents"`, mapping to a list of new `Document` objects,
            each representing an extracted sub-table from the original CSV.

        - If a document cannot be processed, it is returned unchanged.
        - The `meta` field from the original document is preserved in the split documents.
        """
        if len(documents) == 0:
            return {"documents": documents}

        split_documents = []
        for document in documents:
            try:
                df = pd.read_csv(StringIO(document.content), header=None, dtype=object)  # type: ignore
            except Exception as e:
                logger.error(f"Error processing document {document.id}. Keeping it, but skipping splitting. Error: {e}")
                split_documents.append(document)
                continue

            if self.row_split_threshold is not None:
                # split by rows
                split_dfs = self._split_dataframe(df=df, split_threshold=self.row_split_threshold, axis="row")
            elif self.column_split_threshold is not None:
                # split by columns
                split_dfs = self._split_dataframe(df=df, split_threshold=self.column_split_threshold, axis="column")
            else:
                # recursive split
                split_dfs = self._recursive_split(
                    df=df,
                    row_split_threshold=self.row_split_threshold,
                    column_split_threshold=self.column_split_threshold,
                )

            for split_df in split_dfs:
                split_documents.append(
                    Document(
                        content=split_df.to_csv(index=False, header=False, lineterminator="\n"),
                        meta=document.meta.copy(),
                    )
                )

        return {"documents": split_documents}

    @staticmethod
    def _find_split_indices(df: "pd.DataFrame", split_threshold: int, axis: Literal["row", "column"]) -> List[int]:
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

        # Identify groups of consecutive empty elements
        split_indices = []
        consecutive_count = 1
        for i in range(1, len(empty_elements)):
            if empty_elements[i] == empty_elements[i - 1] + 1:
                consecutive_count += 1
            else:
                if consecutive_count >= split_threshold:
                    split_indices.append(empty_elements[i - 1])
                consecutive_count = 1

        if consecutive_count >= split_threshold:
            split_indices.append(empty_elements[-1])

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

        # Split the DataFrame at identified indices
        sub_tables = []
        start_idx = 0
        df_length = df.shape[0] if axis == "row" else df.shape[1]
        for end_idx in split_indices + [df_length]:
            # Avoid empty splits
            if end_idx - start_idx > 1:
                if axis == "row":
                    # TODO Shouldn't drop all empty rows just the ones in the range
                    sub_table = df.iloc[start_idx:end_idx].dropna(how="all", axis=0)
                else:
                    # TODO Shouldn't drop all empty columns just the ones in the range
                    sub_table = df.iloc[:, start_idx:end_idx].dropna(how="all", axis=1)
                if not sub_table.empty:
                    sub_tables.append(sub_table)
            start_idx = end_idx + 1

        return sub_tables

    def _recursive_split(
        self, df: "pd.DataFrame", row_split_threshold: Optional[int], column_split_threshold: Optional[int]
    ) -> List["pd.DataFrame"]:
        """
        Recursively splits a DataFrame.

        Recursively splits a DataFrame first by empty rows, then by empty columns, and repeats the process
        until no more splits are possible. Returns a list of DataFrames, each representing a fully separated sub-table.

        :param df: A Pandas DataFrame representing a table (or multiple tables) extracted from a CSV.
        :param row_split_threshold: The minimum number of consecutive empty rows required to trigger a split.
        :param column_split_threshold: The minimum number of consecutive empty columns to trigger a split.

        **Splitting Process:**
        1. Row Splitting: Detects empty rows and separates tables stacked vertically.
        2. Column Splitting: Detects empty columns and separates side-by-side tables.
        3. Recursive Row Check: After splitting by columns, it checks for new row splits
           introduced by the column split.

        Termination Condition: If no further splits are detected, the recursion stops.
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
