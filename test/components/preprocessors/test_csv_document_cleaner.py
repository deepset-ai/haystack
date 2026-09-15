# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.components.preprocessors.csv_document_cleaner import CSVDocumentCleaner


def test_empty_column() -> None:
    csv_content = """,A,B,C
,1,2,3
,4,5,6
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "A,B,C\n1,2,3\n4,5,6\n"


@pytest.mark.parametrize("empty_row", [",,\n", "\n"])
def test_empty_row(empty_row: str) -> None:
    csv_content = f"A,B,C\n1,2,3\n{empty_row}4,5,6\n"
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "A,B,C\n1,2,3\n4,5,6\n"


def test_empty_column_and_row() -> None:
    csv_content = """,A,B,C
,1,2,3
,,,
,4,5,6
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "A,B,C\n1,2,3\n4,5,6\n"


def test_ignore_rows() -> None:
    csv_content = """,,
A,B,C
4,5,6
7,8,9
"""
    csv_document = Document(content=csv_content, meta={"name": "test.csv"})
    csv_document_cleaner = CSVDocumentCleaner(ignore_rows=1)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",,\nA,B,C\n4,5,6\n7,8,9\n"
    assert cleaned_document.meta == {"name": "test.csv"}


def test_ignore_rows_2() -> None:
    csv_content = """A,B,C
,,
4,5,6
7,8,9
"""
    csv_document = Document(content=csv_content, meta={"name": "test.csv"})
    csv_document_cleaner = CSVDocumentCleaner(ignore_rows=1)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "A,B,C\n4,5,6\n7,8,9\n"
    assert cleaned_document.meta == {"name": "test.csv"}


def test_ignore_rows_3() -> None:
    csv_content = """A,B,C
4,,6
7,,9
"""
    csv_document = Document(content=csv_content, meta={"name": "test.csv"})
    csv_document_cleaner = CSVDocumentCleaner(ignore_rows=1)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "A,C\n4,6\n7,9\n"
    assert cleaned_document.meta == {"name": "test.csv"}


def test_ignore_columns() -> None:
    csv_content = """,,A,B
,2,3,4
,7,8,9
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(ignore_columns=1)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",,A,B\n,2,3,4\n,7,8,9\n"


def test_too_many_ignore_rows() -> None:
    csv_content = """,,
A,B,C
4,5,6
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(ignore_rows=4)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",,\nA,B,C\n4,5,6\n"


def test_too_many_ignore_columns() -> None:
    csv_content = """,,
A,B,C
4,5,6
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(ignore_columns=4)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",,\nA,B,C\n4,5,6\n"


def test_ignore_rows_and_columns() -> None:
    csv_content = """,A,B,C
1,item,s,
2,item2,fd,
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(ignore_columns=1, ignore_rows=1)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",A,B\n1,item,s\n2,item2,fd\n"


def test_zero_ignore_rows_and_columns() -> None:
    csv_content = """,A,B,C
1,item,s,
2,item2,fd,
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(ignore_columns=0, ignore_rows=0)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",A,B,C\n1,item,s,\n2,item2,fd,\n"


def test_empty_document() -> None:
    csv_document = Document(content="")
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ""
    assert cleaned_document.meta == {}


def test_empty_documents() -> None:
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([])
    assert result["documents"] == []


def test_keep_id() -> None:
    csv_content = """,A,B,C
1,item,s,
"""
    csv_document = Document(id="123", content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(keep_id=True)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.id == "123"
    assert cleaned_document.content == ",A,B,C\n1,item,s,\n"


def test_id_not_none() -> None:
    csv_content = """,A,B,C
1,item,s,
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.id != ""
    assert cleaned_document.content == ",A,B,C\n1,item,s,\n"


@pytest.mark.parametrize("empty_row", [",,\n", "\n"])
def test_remove_empty_rows_false(empty_row: str) -> None:
    csv_content = f",B,C\n{empty_row},5,6\n"
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(remove_empty_rows=False)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "B,C\n,\n5,6\n"


def test_ignore_rows_preserves_blank_rows() -> None:
    csv_document = Document(content="A,B\n\n\n1,2\n")
    cleaner = CSVDocumentCleaner(ignore_rows=2)

    result = cleaner.run(documents=[csv_document])

    assert result["documents"][0].content == "A,B\n,\n1,2\n"


def test_remove_empty_rows_false_preserves_trailing_blank_row() -> None:
    csv_document = Document(content="A,B\n1,2\n\n")
    cleaner = CSVDocumentCleaner(remove_empty_rows=False)

    result = cleaner.run(documents=[csv_document])

    assert result["documents"][0].content == "A,B\n1,2\n,\n"


@pytest.mark.parametrize("remove_empty_rows", [True, False])
def test_leading_blank_rows(remove_empty_rows: bool) -> None:
    csv_document = Document(content="\n,B,C\n,5,6\n")
    cleaner = CSVDocumentCleaner(remove_empty_rows=remove_empty_rows)

    result = cleaner.run(documents=[csv_document])

    expected = "B,C\n5,6\n" if remove_empty_rows else ",\nB,C\n5,6\n"
    assert result["documents"][0].content == expected


def test_ignore_rows_preserves_leading_blank_row() -> None:
    csv_document = Document(content="\nA,B\n\n1,2\n")
    cleaner = CSVDocumentCleaner(ignore_rows=1)

    result = cleaner.run(documents=[csv_document])

    assert result["documents"][0].content == ",\nA,B\n1,2\n"


def test_blank_lines_inside_quoted_field_are_preserved() -> None:
    csv_content = 'A,B\n"first\n\nlast",2\n'
    cleaner = CSVDocumentCleaner(remove_empty_rows=False)

    result = cleaner.run(documents=[Document(content=csv_content)])

    assert result["documents"][0].content == csv_content


def test_remove_empty_columns_false() -> None:
    csv_content = """,B,C
,,
,,4
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(remove_empty_columns=False)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",B,C\n,,4\n"


def test_remove_empty_rows_and_columns_false() -> None:
    csv_content = """,B,C
,,4
,,
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(remove_empty_rows=False, remove_empty_columns=False)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == ",B,C\n,,4\n,,\n"
