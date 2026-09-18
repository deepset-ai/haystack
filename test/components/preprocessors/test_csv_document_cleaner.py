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


def test_remove_empty_rows_false() -> None:
    csv_content = """,B,C
,,
,5,6
"""
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner(remove_empty_rows=False)
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "B,C\n,\n5,6\n"


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


def test_literal_na_strings_preserved_in_cells() -> None:
    """Regression for #12784: pandas' default NA strings ('N/A', 'NA', 'n/a',
    'NULL', 'None', 'NaN', 'nan') used to be coerced to NaN before the cleaner
    could see them, so cells that held those literal strings came back empty."""
    csv_content = "A,B,C\nfoo,N/A,bar\nbaz,NA,qux\n"
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "A,B,C\nfoo,N/A,bar\nbaz,NA,qux\n"


def test_row_of_na_strings_not_treated_as_empty() -> None:
    """Regression for #12784: a row whose cells all say 'N/A' used to be dropped
    by dropna(how='all') because pandas had already coerced every cell to NaN.
    With keep_default_na=False the literal string is preserved and the row
    stays in the output."""
    csv_content = "A,B,C\n1,2,3\nN/A,N/A,N/A\n4,5,6\n"
    csv_document = Document(content=csv_content)
    csv_document_cleaner = CSVDocumentCleaner()
    result = csv_document_cleaner.run([csv_document])
    cleaned_document = result["documents"][0]
    assert cleaned_document.content == "A,B,C\n1,2,3\nN/A,N/A,N/A\n4,5,6\n"
