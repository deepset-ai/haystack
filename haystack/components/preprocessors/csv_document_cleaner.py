# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from io import StringIO
from typing import Dict, List, Optional

from haystack import Document, component, logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pandas'") as pandas_import:
    import pandas as pd

logger = logging.getLogger(__name__)


@component
class CSVDocumentCleaner:
    """
    A component for cleaning CSV documents by removing empty rows and columns.

    This component processes CSV content stored in Documents, allowing
    for the optional ignoring of a specified number of rows and columns before performing
    the cleaning operation. Additionally, it provides options to keep document IDs and
    control whether empty rows and columns should be removed.
    """

    def __init__(
        self,
        *,
        ignore_rows: int = 0,
        ignore_columns: int = 0,
        remove_empty_rows: bool = True,
        remove_empty_columns: bool = True,
        keep_id: bool = False,
    ) -> None:
        """
        Initializes the CSVDocumentCleaner component.

        :param ignore_rows: Number of rows to ignore from the top of the CSV table before processing.
        :param ignore_columns: Number of columns to ignore from the left of the CSV table before processing.
        :param remove_empty_rows: Whether to remove rows that are entirely empty.
        :param remove_empty_columns: Whether to remove columns that are entirely empty.
        :param keep_id: Whether to retain the original document ID in the output document.

        Rows and columns ignored using these parameters are preserved in the final output, meaning
        they are not considered when removing empty rows and columns.
        """
        self.ignore_rows = ignore_rows
        self.ignore_columns = ignore_columns
        self.remove_empty_rows = remove_empty_rows
        self.remove_empty_columns = remove_empty_columns
        self.keep_id = keep_id
        pandas_import.check()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Cleans CSV documents by removing empty rows and columns while preserving specified ignored rows and columns.

        :param documents: List of Documents containing CSV-formatted content.
        :return: A dictionary with a list of cleaned Documents under the key "documents".

        Processing steps:
        1. Reads each document's content as a CSV table.
        2. Retains the specified number of `ignore_rows` from the top and `ignore_columns` from the left.
        3. Drops any rows and columns that are entirely empty (if enabled by `remove_empty_rows` and
            `remove_empty_columns`).
        4. Reattaches the ignored rows and columns to maintain their original positions.
        5. Returns the cleaned CSV content as a new `Document` object, with an option to retain the original
            document ID.
        """
        if len(documents) == 0:
            return {"documents": []}

        ignore_rows = self.ignore_rows
        ignore_columns = self.ignore_columns

        cleaned_documents = []
        for document in documents:
            try:
                df = pd.read_csv(StringIO(document.content), header=None, dtype=object)  # type: ignore
            except Exception as e:
                logger.error(
                    "Error processing document {id}. Keeping it, but skipping cleaning. Error: {error}",
                    id=document.id,
                    error=e,
                )
                cleaned_documents.append(document)
                continue

            if ignore_rows > df.shape[0] or ignore_columns > df.shape[1]:
                logger.warning(
                    "Document {id} has fewer rows {df_rows} or columns {df_cols} "
                    "than the number of rows {rows} or columns {cols} to ignore. "
                    "Keeping the entire document.",
                    id=document.id,
                    df_rows=df.shape[0],
                    df_cols=df.shape[1],
                    rows=ignore_rows,
                    cols=ignore_columns,
                )
                cleaned_documents.append(document)
                continue

            final_df = self._clean_df(df=df, ignore_rows=ignore_rows, ignore_columns=ignore_columns)

            clean_doc = Document(
                id=document.id if self.keep_id else "",
                content=final_df.to_csv(index=False, header=False, lineterminator="\n"),
                blob=document.blob,
                meta=deepcopy(document.meta),
                score=document.score,
                embedding=document.embedding,
                sparse_embedding=document.sparse_embedding,
            )
            cleaned_documents.append(clean_doc)
        return {"documents": cleaned_documents}

    def _clean_df(self, df: "pd.DataFrame", ignore_rows: int, ignore_columns: int) -> "pd.DataFrame":
        """
        Cleans a DataFrame by removing empty rows and columns while preserving ignored sections.

        :param df: The input DataFrame representing the CSV data.
        :param ignore_rows: Number of top rows to ignore.
        :param ignore_columns: Number of left columns to ignore.
        """
        # Get ignored rows and columns
        ignored_rows = self._get_ignored_rows(df=df, ignore_rows=ignore_rows)
        ignored_columns = self._get_ignored_columns(df=df, ignore_columns=ignore_columns)
        final_df = df.iloc[ignore_rows:, ignore_columns:]

        # Drop rows that are entirely empty
        if self.remove_empty_rows:
            final_df = final_df.dropna(axis=0, how="all")

        # Drop columns that are entirely empty
        if self.remove_empty_columns:
            final_df = final_df.dropna(axis=1, how="all")

        # Reattach ignored rows
        if ignore_rows > 0 and ignored_rows is not None:
            # Keep only relevant columns
            ignored_rows = ignored_rows.loc[:, final_df.columns]
            final_df = pd.concat([ignored_rows, final_df], axis=0)

        # Reattach ignored columns
        if ignore_columns > 0 and ignored_columns is not None:
            # Keep only relevant rows
            ignored_columns = ignored_columns.loc[final_df.index, :]
            final_df = pd.concat([ignored_columns, final_df], axis=1)

        return final_df

    @staticmethod
    def _get_ignored_rows(df: "pd.DataFrame", ignore_rows: int) -> Optional["pd.DataFrame"]:
        """
        Extracts the rows to be ignored from the DataFrame.

        :param df: The input DataFrame.
        :param ignore_rows: Number of rows to extract from the top.
        """
        if ignore_rows > 0:
            return df.iloc[:ignore_rows, :]
        return None

    @staticmethod
    def _get_ignored_columns(df: "pd.DataFrame", ignore_columns: int) -> Optional["pd.DataFrame"]:
        """
        Extracts the columns to be ignored from the DataFrame.

        :param df: The input DataFrame.
        :param ignore_columns: Number of columns to extract from the left.
        """
        if ignore_columns > 0:
            return df.iloc[:, :ignore_columns]
        return None
