# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from typing import Dict, List

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
    the cleaning operation.
    """

    def __init__(self, ignore_rows: int = 0, ignore_columns: int = 0) -> None:
        """
        Initializes the CSVDocumentCleaner component.

        :param ignore_rows: Number of rows to ignore from the top of the CSV table before processing.
        :param ignore_columns: Number of columns to ignore from the left of the CSV table before processing.

        Rows and columns ignored using these parameters are preserved in the final output, meaning
        they are not considered when removing empty rows and columns.
        """
        self.ignore_rows = ignore_rows
        self.ignore_columns = ignore_columns
        pandas_import.check()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Cleans CSV documents by removing empty rows and columns while preserving specified ignored rows and columns.

        :param documents: List of Documents containing CSV-formatted content.

        Processing steps:
        1. Reads each document's content as a CSV table.
        2. Retains the specified number of `ignore_rows` from the top and `ignore_columns` from the left.
        3. Drops any rows and columns that are entirely empty (all NaN values).
        4. Reattaches the ignored rows and columns to maintain their original positions.
        5. Returns the cleaned CSV content as a new `Document` object.
        """
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

            # Save ignored rows
            ignored_rows = None
            if ignore_rows > 0:
                ignored_rows = df.iloc[:ignore_rows, :]

            # Save ignored columns
            ignored_columns = None
            if ignore_columns > 0:
                ignored_columns = df.iloc[:, :ignore_columns]

            # Drop rows and columns that are entirely empty
            remaining_df = df.iloc[ignore_rows:, ignore_columns:]
            final_df = remaining_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

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

            cleaned_documents.append(
                Document(
                    content=final_df.to_csv(index=False, header=False, lineterminator="\n"), meta=document.meta.copy()
                )
            )
        return {"documents": cleaned_documents}
