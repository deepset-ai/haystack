"""read_csv_kwargs that sets a header must not break CSV splitting."""

from haystack import Document
from haystack.components.preprocessors.csv_document_splitter import CSVDocumentSplitter

CSV = "name,score\nAda,9\n\nBob,8"


def test_default_header_none_keeps_positional_columns():
    splitter = CSVDocumentSplitter(row_split_threshold=1, column_split_threshold=None)
    result = splitter.run([Document(content=CSV)])
    assert [
        (d.meta["row_idx_start"], d.meta["col_idx_start"]) for d in result["documents"]
    ] == [(0, 0), (3, 0)]


def test_read_csv_kwargs_header_reports_column_positions():
    """With a caller-supplied header the columns are labels, not positions."""
    splitter = CSVDocumentSplitter(
        row_split_threshold=1,
        column_split_threshold=None,
        read_csv_kwargs={"header": 0},
    )
    result = splitter.run([Document(content=CSV)])
    assert [
        (d.meta["row_idx_start"], d.meta["col_idx_start"]) for d in result["documents"]
    ] == [(0, 0), (2, 0)]
