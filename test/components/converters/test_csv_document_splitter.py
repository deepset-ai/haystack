import pytest
from haystack import Document
from haystack.components.preprocessors.csv_document_splitter import CSVDocumentSplitter


@pytest.fixture
def splitter() -> CSVDocumentSplitter:
    return CSVDocumentSplitter()


def test_single_table_no_split(splitter: CSVDocumentSplitter) -> None:
    csv_content = """A,B,C
1,2,3
4,5,6
"""
    doc = Document(content=csv_content)
    result = splitter.run([doc])["documents"]
    assert len(result) == 1
    assert result[0].content == csv_content


def test_row_split(splitter: CSVDocumentSplitter) -> None:
    csv_content = """A,B,C
1,2,3
,,
,,
X,Y,Z
7,8,9
"""
    doc = Document(content=csv_content)
    result = splitter.run([doc])["documents"]
    assert len(result) == 2
    expected_tables = ["A,B,C\n1,2,3\n", "X,Y,Z\n7,8,9\n"]
    for i, table in enumerate(result):
        assert table.content == expected_tables[i]


def test_column_split(splitter: CSVDocumentSplitter) -> None:
    csv_content = """A,B,,,X,Y
1,2,,,7,8
3,4,,,9,10
"""
    doc = Document(content=csv_content)
    result = splitter.run([doc])["documents"]
    assert len(result) == 2
    expected_tables = ["A,B\n1,2\n3,4\n", "X,Y\n7,8\n9,10\n"]
    for i, table in enumerate(result):
        assert table.content == expected_tables[i]


def test_recursive_split(splitter: CSVDocumentSplitter) -> None:
    csv_content = """A,B,,,X,Y
1,2,,,7,8
,,,,,
,,,,,
P,Q,,,M,N
3,4,,,9,10
"""
    doc = Document(content=csv_content)
    result = splitter.run([doc])["documents"]
    assert len(result) == 4
    expected_tables = ["A,B\n1,2\n", "X,Y\n7,8\n", "P,Q\n3,4\n", "M,N\n9,10\n"]
    for i, table in enumerate(result):
        assert table.content == expected_tables[i]


def test_threshold_no_effect() -> None:
    splitter = CSVDocumentSplitter(row_split_threshold=3)
    csv_content = """A,B,C
1,2,3
,,
,,
X,Y,Z
7,8,9
"""
    doc = Document(content=csv_content)
    result = splitter.run([doc])["documents"]
    assert len(result) == 1


def test_empty_input(splitter: CSVDocumentSplitter) -> None:
    csv_content = ""
    doc = Document(content=csv_content)
    result = splitter.run([doc])["documents"]
    assert len(result) == 1
    assert result[0].content == csv_content
