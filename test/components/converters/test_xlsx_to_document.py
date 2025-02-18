import logging
from typing import Union

import pytest

from haystack.components.converters.xlsx import XLSXToDocument


class TestXLSXToDocument:
    def test_init(self) -> None:
        converter = XLSXToDocument()
        assert converter.sheet_name is None
        assert converter.read_excel_kwargs == {}
        assert converter.table_format == "csv"
        assert converter.table_format_kwargs == {}

    def test_run_basic_tables(self, test_files_path) -> None:
        converter = XLSXToDocument(store_full_path=True)
        paths = [test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"]
        results = converter.run(sources=paths, meta={"date_added": "2022-01-01T00:00:00"})
        documents = results["documents"]
        assert len(documents) == 2
        assert documents[0].content == ",A,B\n1,col_a,col_b\n2,1.5,test\n"
        assert documents[0].meta == {
            "date_added": "2022-01-01T00:00:00",
            "file_path": str(test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"),
            "xlsx": {"sheet_name": "Basic Table"},
        }
        assert documents[1].content == ",A,B\n1,col_c,col_d\n2,True,\n"
        assert documents[1].meta == {
            "date_added": "2022-01-01T00:00:00",
            "file_path": str(test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"),
            "xlsx": {"sheet_name": "Table Missing Value"},
        }

    def test_run_table_empty_rows_and_columns(self, test_files_path) -> None:
        converter = XLSXToDocument(store_full_path=False)
        paths = [test_files_path / "xlsx" / "table_empty_rows_and_columns.xlsx"]
        results = converter.run(sources=paths, meta={"date_added": "2022-01-01T00:00:00"})
        documents = results["documents"]
        assert len(documents) == 1
        assert documents[0].content == ",A,B,C\n1,,,\n2,,,\n3,,,\n4,,col_a,col_b\n5,,1.5,test\n"
        assert documents[0].meta == {
            "date_added": "2022-01-01T00:00:00",
            "file_path": "table_empty_rows_and_columns.xlsx",
            "xlsx": {"sheet_name": "Sheet1"},
        }

    def test_run_multiple_tables_in_one_sheet(self, test_files_path) -> None:
        converter = XLSXToDocument(store_full_path=True)
        paths = [test_files_path / "xlsx" / "multiple_tables.xlsx"]
        results = converter.run(sources=paths, meta={"date_added": "2022-01-01T00:00:00"})
        documents = results["documents"]
        assert len(documents) == 1
        assert (
            documents[0].content
            == ",A,B,C,D,E,F\n1,,,,,,\n2,,,,,,\n3,,col_a,col_b,,,\n4,,1.5,test,,col_c,col_d\n5,,,,,3,True\n"
        )
        assert documents[0].meta == {
            "date_added": "2022-01-01T00:00:00",
            "file_path": str(test_files_path / "xlsx" / "multiple_tables.xlsx"),
            "xlsx": {"sheet_name": "Sheet1"},
        }

    def test_run_markdown(self, test_files_path) -> None:
        converter = XLSXToDocument(table_format="markdown", store_full_path=True)
        paths = [test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"]
        results = converter.run(sources=paths, meta={"date_added": "2022-01-01T00:00:00"})
        documents = results["documents"]
        assert len(documents) == 2
        assert (
            documents[0].content
            == "|    | A     | B     |\n|---:|:------|:------|\n|  1 | col_a | col_b |\n|  2 | 1.5   | test  |"
        )
        assert documents[0].meta == {
            "date_added": "2022-01-01T00:00:00",
            "file_path": str(test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"),
            "xlsx": {"sheet_name": "Basic Table"},
        }
        assert (
            documents[1].content
            == "|    | A     | B     |\n|---:|:------|:------|\n|  1 | col_c | col_d |\n|  2 | True  | nan   |"
        )
        assert documents[1].meta == {
            "date_added": "2022-01-01T00:00:00",
            "file_path": str(test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"),
            "xlsx": {"sheet_name": "Table Missing Value"},
        }

    @pytest.mark.parametrize(
        "sheet_name, expected_sheet_name, expected_content",
        [
            ("Basic Table", "Basic Table", ",A,B\n1,col_a,col_b\n2,1.5,test\n"),
            ("Table Missing Value", "Table Missing Value", ",A,B\n1,col_c,col_d\n2,True,\n"),
            (0, 0, ",A,B\n1,col_a,col_b\n2,1.5,test\n"),
            (1, 1, ",A,B\n1,col_c,col_d\n2,True,\n"),
        ],
    )
    def test_run_sheet_name(
        self, sheet_name: Union[int, str], expected_sheet_name: str, expected_content: str, test_files_path
    ) -> None:
        converter = XLSXToDocument(sheet_name=sheet_name, store_full_path=True)
        paths = [test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"]
        results = converter.run(sources=paths)
        documents = results["documents"]
        assert len(documents) == 1
        assert documents[0].content == expected_content
        assert documents[0].meta == {
            "file_path": str(test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"),
            "xlsx": {"sheet_name": expected_sheet_name},
        }

    def test_run_with_read_excel_kwargs(self, test_files_path) -> None:
        converter = XLSXToDocument(sheet_name="Basic Table", read_excel_kwargs={"skiprows": 1}, store_full_path=True)
        paths = [test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"]
        results = converter.run(sources=paths, meta={"date_added": "2022-01-01T00:00:00"})
        documents = results["documents"]
        assert len(documents) == 1
        assert documents[0].content == ",A,B\n1,1.5,test\n"
        assert documents[0].meta == {
            "date_added": "2022-01-01T00:00:00",
            "file_path": str(test_files_path / "xlsx" / "basic_tables_two_sheets.xlsx"),
            "xlsx": {"sheet_name": "Basic Table"},
        }

    def test_run_error_wrong_file_type(self, caplog: pytest.LogCaptureFixture, test_files_path) -> None:
        converter = XLSXToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "sample_pdf_1.pdf and convert it" in caplog.text
            assert results["documents"] == []

    def test_run_error_non_existent_file(self, caplog: pytest.LogCaptureFixture) -> None:
        converter = XLSXToDocument()
        paths = ["non_existing_file.docx"]
        with caplog.at_level(logging.WARNING):
            converter.run(sources=paths)
            assert "Could not read non_existing_file.docx" in caplog.text
