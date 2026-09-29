"""read_csv_kwargs that sets a header must not break CSV splitting."""

from haystack import Document
from haystack.components.preprocessors.csv_document_splitter import CSVDocumentSplitter

ROWS = "name,score\nAda,9\n\nBob,8"
COLUMNS = "z,_,a\n1,,2\n3,,4\n"


def test_default_header_none_keeps_positional_columns():
    splitter = CSVDocumentSplitter(row_split_threshold=1, column_split_threshold=None)
    result = splitter.run([Document(content=ROWS)])
    assert [(d.meta["row_idx_start"], d.meta["col_idx_start"]) for d in result["documents"]] == [(0, 0), (3, 0)]


def test_read_csv_kwargs_header_reports_column_positions():
    """With a caller-supplied header the columns are labels, not positions."""
    splitter = CSVDocumentSplitter(row_split_threshold=1, column_split_threshold=None, read_csv_kwargs={"header": 0})
    result = splitter.run([Document(content=ROWS)])
    assert [(d.meta["row_idx_start"], d.meta["col_idx_start"]) for d in result["documents"]] == [(0, 0), (2, 0)]


def test_header_labels_do_not_reorder_sub_tables():
    """Sub-tables keep the original column order, not the alphabetical one."""
    splitter = CSVDocumentSplitter(row_split_threshold=None, column_split_threshold=1, read_csv_kwargs={"header": 0})
    result = splitter.run([Document(content=COLUMNS)])
    assert [(d.meta["col_idx_start"], d.meta["split_id"]) for d in result["documents"]] == [(0, 0), (2, 1)]
    assert [d.content.strip().replace("\n", "/") for d in result["documents"]] == ["1/3", "2/4"]
