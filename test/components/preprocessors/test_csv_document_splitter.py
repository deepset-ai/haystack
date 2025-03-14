# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
from pandas import read_csv
from io import StringIO
from haystack import Document, Pipeline
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.components.preprocessors.csv_document_splitter import CSVDocumentSplitter


@pytest.fixture
def splitter() -> CSVDocumentSplitter:
    return CSVDocumentSplitter()


@pytest.fixture
def csv_with_four_rows() -> str:
    return """A,B,C
1,2,3
X,Y,Z
7,8,9
"""


@pytest.fixture
def two_tables_sep_by_two_empty_rows() -> str:
    return """A,B,C
1,2,3
,,
,,
X,Y,Z
7,8,9
"""


@pytest.fixture
def three_tables_sep_by_empty_rows() -> str:
    return """A,B,C
,,
1,2,3
,,
,,
X,Y,Z
7,8,9
"""


@pytest.fixture
def two_tables_sep_by_two_empty_columns() -> str:
    return """A,B,,,X,Y
1,2,,,7,8
3,4,,,9,10
"""


class TestFindSplitIndices:
    def test_find_split_indices_row_two_tables(
        self, splitter: CSVDocumentSplitter, two_tables_sep_by_two_empty_rows: str
    ) -> None:
        df = read_csv(StringIO(two_tables_sep_by_two_empty_rows), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="row")
        assert result == [(2, 3)]

    def test_find_split_indices_row_two_tables_with_empty_row(
        self, splitter: CSVDocumentSplitter, three_tables_sep_by_empty_rows: str
    ) -> None:
        df = read_csv(StringIO(three_tables_sep_by_empty_rows), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="row")
        assert result == [(3, 4)]

    def test_find_split_indices_row_three_tables(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,C
1,2,3
,,
,,
X,Y,Z
7,8,9
,,
,,
P,Q,R
"""
        df = read_csv(StringIO(csv_content), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="row")
        assert result == [(2, 3), (6, 7)]

    def test_find_split_indices_column_two_tables(
        self, splitter: CSVDocumentSplitter, two_tables_sep_by_two_empty_columns: str
    ) -> None:
        df = read_csv(StringIO(two_tables_sep_by_two_empty_columns), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=1, axis="column")
        assert result == [(2, 3)]

    def test_find_split_indices_column_two_tables_with_empty_column(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,,B,,,X,Y
1,,2,,,7,8
3,,4,,,9,10
"""
        df = read_csv(StringIO(csv_content), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="column")
        assert result == [(3, 4)]

    def test_find_split_indices_column_three_tables(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,,,X,Y,,,P,Q
1,2,,,7,8,,,11,12
3,4,,,9,10,,,13,14
"""
        df = read_csv(StringIO(csv_content), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="column")
        assert result == [(2, 3), (6, 7)]


class TestInit:
    def test_row_split_threshold_raises_error(self) -> None:
        with pytest.raises(ValueError, match="row_split_threshold must be greater than 0"):
            CSVDocumentSplitter(row_split_threshold=-1)

    def test_column_split_threshold_raises_error(self) -> None:
        with pytest.raises(ValueError, match="column_split_threshold must be greater than 0"):
            CSVDocumentSplitter(column_split_threshold=-1)

    def test_row_split_threshold_and_row_column_threshold_none(self) -> None:
        with pytest.raises(
            ValueError, match="At least one of row_split_threshold or column_split_threshold must be specified."
        ):
            CSVDocumentSplitter(row_split_threshold=None, column_split_threshold=None)


class TestCSVDocumentSplitter:
    def test_single_table_no_split(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,C
1,2,3
4,5,6
"""
        doc = Document(content=csv_content, id="test_id")
        result = splitter.run([doc])["documents"]
        assert len(result) == 1
        assert result[0].content == csv_content
        assert result[0].meta == {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 0, "split_id": 0}

    def test_row_split(self, splitter: CSVDocumentSplitter, two_tables_sep_by_two_empty_rows: str) -> None:
        doc = Document(content=two_tables_sep_by_two_empty_rows, id="test_id")
        result = splitter.run([doc])["documents"]
        assert len(result) == 2
        expected_tables = ["A,B,C\n1,2,3\n", "X,Y,Z\n7,8,9\n"]
        expected_meta = [
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 0, "split_id": 0},
            {"source_id": "test_id", "row_idx_start": 4, "col_idx_start": 0, "split_id": 1},
        ]
        for i, table in enumerate(result):
            assert table.content == expected_tables[i]
            assert table.meta == expected_meta[i]

    def test_column_split(self, splitter: CSVDocumentSplitter, two_tables_sep_by_two_empty_columns: str) -> None:
        doc = Document(content=two_tables_sep_by_two_empty_columns, id="test_id")
        result = splitter.run([doc])["documents"]
        assert len(result) == 2
        expected_tables = ["A,B\n1,2\n3,4\n", "X,Y\n7,8\n9,10\n"]
        expected_meta = [
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 0, "split_id": 0},
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 4, "split_id": 1},
        ]
        for i, table in enumerate(result):
            assert table.content == expected_tables[i]
            assert table.meta == expected_meta[i]

    def test_recursive_split_one_level(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,,,X,Y
1,2,,,7,8
,,,,,
,,,,,
P,Q,,,M,N
3,4,,,9,10
"""
        doc = Document(content=csv_content, id="test_id")
        result = splitter.run([doc])["documents"]
        assert len(result) == 4
        expected_tables = ["A,B\n1,2\n", "X,Y\n7,8\n", "P,Q\n3,4\n", "M,N\n9,10\n"]
        expected_meta = [
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 0, "split_id": 0},
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 4, "split_id": 1},
            {"source_id": "test_id", "row_idx_start": 4, "col_idx_start": 0, "split_id": 2},
            {"source_id": "test_id", "row_idx_start": 4, "col_idx_start": 4, "split_id": 3},
        ]
        for i, table in enumerate(result):
            assert table.content == expected_tables[i]
            assert table.meta == expected_meta[i]

    def test_recursive_split_two_levels(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,,,X,Y
1,2,,,7,8
,,,,M,N
,,,,9,10
P,Q,,,,
3,4,,,,
"""
        doc = Document(content=csv_content, id="test_id")
        result = splitter.run([doc])["documents"]
        assert len(result) == 3
        expected_tables = ["A,B\n1,2\n", "X,Y\n7,8\nM,N\n9,10\n", "P,Q\n3,4\n"]
        expected_meta = [
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 0, "split_id": 0},
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 4, "split_id": 1},
            {"source_id": "test_id", "row_idx_start": 4, "col_idx_start": 0, "split_id": 2},
        ]
        for i, table in enumerate(result):
            assert table.content == expected_tables[i]
            assert table.meta == expected_meta[i]

    def test_csv_with_blank_lines(self, splitter: CSVDocumentSplitter) -> None:
        csv_data = """ID,LeftVal,,,RightVal,Extra
1,Hello,,,World,Joined
2,StillLeft,,,StillRight,Bridge

A,B,,,C,D
E,F,,,G,H
"""
        splitter = CSVDocumentSplitter(row_split_threshold=1, column_split_threshold=1)
        result = splitter.run([Document(content=csv_data, id="test_id")])
        docs = result["documents"]
        assert len(docs) == 4
        expected_tables = [
            "ID,LeftVal\n1,Hello\n2,StillLeft\n",
            "RightVal,Extra\nWorld,Joined\nStillRight,Bridge\n",
            "A,B\nE,F\n",
            "C,D\nG,H\n",
        ]
        expected_meta = [
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 0, "split_id": 0},
            {"source_id": "test_id", "row_idx_start": 0, "col_idx_start": 4, "split_id": 1},
            {"source_id": "test_id", "row_idx_start": 4, "col_idx_start": 0, "split_id": 2},
            {"source_id": "test_id", "row_idx_start": 4, "col_idx_start": 4, "split_id": 3},
        ]
        for i, table in enumerate(docs):
            assert table.content == expected_tables[i]
            assert table.meta == expected_meta[i]

    def test_sub_table_with_one_row(self):
        splitter = CSVDocumentSplitter(row_split_threshold=1)
        doc = Document(content="""A,B,C\n1,2,3\n,,\n4,5,6""")
        split_result = splitter.run([doc])
        assert len(split_result["documents"]) == 2

    def test_threshold_no_effect(self, two_tables_sep_by_two_empty_rows: str) -> None:
        splitter = CSVDocumentSplitter(row_split_threshold=3)
        doc = Document(content=two_tables_sep_by_two_empty_rows)
        result = splitter.run([doc])["documents"]
        assert len(result) == 1

    def test_empty_input(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = ""
        doc = Document(content=csv_content)
        result = splitter.run([doc])["documents"]
        assert len(result) == 1
        assert result[0].content == csv_content

    def test_empty_documents(self, splitter: CSVDocumentSplitter) -> None:
        result = splitter.run([])["documents"]
        assert len(result) == 0

    def test_to_dict_with_defaults(self) -> None:
        splitter = CSVDocumentSplitter()
        config_serialized = component_to_dict(splitter, name="CSVDocumentSplitter")
        config = {
            "type": "haystack.components.preprocessors.csv_document_splitter.CSVDocumentSplitter",
            "init_parameters": {
                "row_split_threshold": 2,
                "column_split_threshold": 2,
                "read_csv_kwargs": {},
                "split_mode": "threshold",
            },
        }
        assert config_serialized == config

    def test_to_dict_non_defaults(self) -> None:
        splitter = CSVDocumentSplitter(row_split_threshold=1, column_split_threshold=None, read_csv_kwargs={"sep": ";"})
        config_serialized = component_to_dict(splitter, name="CSVDocumentSplitter")
        config = {
            "type": "haystack.components.preprocessors.csv_document_splitter.CSVDocumentSplitter",
            "init_parameters": {
                "row_split_threshold": 1,
                "column_split_threshold": None,
                "read_csv_kwargs": {"sep": ";"},
                "split_mode": "threshold",
            },
        }
        assert config_serialized == config

    def test_from_dict_defaults(self) -> None:
        splitter = component_from_dict(
            CSVDocumentSplitter,
            data={
                "type": "haystack.components.preprocessors.csv_document_splitter.CSVDocumentSplitter",
                "init_parameters": {},
            },
            name="CSVDocumentSplitter",
        )
        assert splitter.row_split_threshold == 2
        assert splitter.column_split_threshold == 2
        assert splitter.read_csv_kwargs == {}

    def test_from_dict_non_defaults(self) -> None:
        splitter = component_from_dict(
            CSVDocumentSplitter,
            data={
                "type": "haystack.components.preprocessors.csv_document_splitter.CSVDocumentSplitter",
                "init_parameters": {
                    "row_split_threshold": 1,
                    "column_split_threshold": None,
                    "read_csv_kwargs": {"sep": ";"},
                    "split_mode": "threshold",
                },
            },
            name="CSVDocumentSplitter",
        )
        assert splitter.row_split_threshold == 1
        assert splitter.column_split_threshold is None
        assert splitter.read_csv_kwargs == {"sep": ";"}

    def test_split_by_row(self, csv_with_four_rows: str) -> None:
        splitter = CSVDocumentSplitter(split_mode="row-wise")
        doc = Document(content=csv_with_four_rows)
        result = splitter.run([doc])["documents"]
        assert len(result) == 4
        assert result[0].content == "A,B,C\n"
        assert result[1].content == "1,2,3\n"
        assert result[2].content == "X,Y,Z\n"

    def test_split_by_row_with_empty_rows(self, caplog) -> None:
        splitter = CSVDocumentSplitter(split_mode="row-wise", row_split_threshold=2)
        doc = Document(content="")
        with caplog.at_level(logging.ERROR):
            result = splitter.run([doc])["documents"]
            assert len(result) == 1
            assert result[0].content == ""
