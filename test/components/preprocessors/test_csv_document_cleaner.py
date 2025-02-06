# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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


def test_empty_row() -> None:
    csv_content = """A,B,C
1,2,3
,,
4,5,6
"""
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
