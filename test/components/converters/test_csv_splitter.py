import pytest
from haystack import Document
from haystack.components.converters.csv import CSVSplitter


class TestCSVSplitter:
    def test_empty_file(self) -> None:
        """
        Completely empty CSV content should produce zero documents.
        """
        splitter = CSVSplitter(detect_side_tables=True)
        result = splitter.run([Document(content="")])
        docs = result["documents"]
        assert len(docs) == 0

    def test_minimal_single_row(self) -> None:
        """
        Single row CSV to confirm we get exactly one block.
        """
        csv_data = "OnlyCol1,OnlyCol2,OnlyCol3"
        splitter = CSVSplitter(detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 1
        assert docs[0].content.strip() == csv_data

    def test_side_tables_false_no_split_on_side(self) -> None:
        """
        detect_side_tables=False => everything in a single row remains one block.
        """
        csv_data = """LeftCol1,LeftCol2,,,RightCol1,RightCol2
L1A,L1B,,,R1A,R1B
L2A,L2B,,,R2A,R2B
"""
        splitter = CSVSplitter(detect_side_tables=False)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 1
        assert "LeftCol1" in docs[0].content
        assert "RightCol1" in docs[0].content

    def test_side_tables_false_with_empty_rows(self) -> None:
        """
        detect_side_tables=False with multiple empty rows => classic row-based splits only.
        """
        csv_data = """A1,A2,,,B1,B2
C1,C2,,,D1,D2


X1,X2,,,Y1,Y2
"""
        splitter = CSVSplitter(split_threshold=2, detect_side_tables=False)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 2
        block_texts = [doc.content for doc in docs]
        assert any("A1,A2,,,B1,B2" in block for block in block_texts)
        assert any("X1,X2,,,Y1,Y2" in block for block in block_texts)

    def test_side_by_side_basic(self) -> None:
        """
        Simple case of left and right tables with detect_side_tables=True => split horizontally.
        """
        csv_data = """LeftCol1,LeftCol2,,,RightCol1,RightCol2
L1A,L1B,,,R1A,R1B
L2A,L2B,,,R2A,R2B
"""
        splitter = CSVSplitter(detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 2
        left_table = docs[0].content
        right_table = docs[1].content
        assert "LeftCol1" in left_table
        assert "RightCol1" in right_table

    def test_empty_rows_and_side_tables(self) -> None:
        """
        Multiple blocks (side-by-side columns + empty lines) => 4 total splits.
        """
        csv_data = """A1,A2,,,B1,B2
A3,A4,,,B3,B4


X1,X2,,,Y1,Y2
X3,X4,,,Y3,Y4
"""
        splitter = CSVSplitter(split_threshold=2, detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 4
        block_texts = [d.content for d in docs]
        assert any("A1,A2" in text for text in block_texts)
        assert any("B3,B4" in text for text in block_texts)
        assert any("X1,X2" in text for text in block_texts)
        assert any("Y3,Y4" in text for text in block_texts)

    def test_complex_bridging(self) -> None:
        """
        Rows bridging from left to right => BFS splits each row into left block & right block.
        """
        csv_data = """ID,LeftVal,,,RightVal,Extra
1,Hello,,,World,Joined
2,StillLeft,,,StillRight,Bridge

A,B,,,C,D
E,F,,,G,H
"""
        splitter = CSVSplitter(split_threshold=1, detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 4
        block_texts = [doc.content for doc in docs]
        assert any("ID,LeftVal" in text for text in block_texts)
        assert any("Hello" in text for text in block_texts)
        assert any("World,Joined" in text for text in block_texts)
        assert any("StillLeft" in text for text in block_texts)
        assert any("StillRight,Bridge" in text for text in block_texts)
        assert any("A,B" in text for text in block_texts)
        assert any("C,D" in text for text in block_texts)
        assert any("E,F" in text for text in block_texts)
        assert any("G,H" in text for text in block_texts)

    def test_multiline_fields(self) -> None:
        """
        CSV with quoted multiline fields to ensure BFS or row-based approach parses them safely.
        """
        csv_data = """ID,Description
1,"This is a
multiline description"
2,"Side-by-side
But still single
field",,RightCol,Yes
"""
        splitter = CSVSplitter(detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) >= 1
        combined_csv = "\n".join(d.content for d in docs)
        assert "multiline description" in combined_csv
        assert "Side-by-side\nBut still single\nfield" in combined_csv

    def test_large_empty_section(self) -> None:
        """
        Large stretch of empty rows => multiple blocks if over split_threshold.
        """
        csv_data = """ColA,ColB
1,2
3,4





5,6
7,8
"""
        splitter = CSVSplitter(split_threshold=2, detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 2

    def test_edge_case_column_padding(self) -> None:
        """
        BFS splits row '7,,,,8,9' into two clusters: left (7) and right (8,9).
        """
        csv_data = """A,B,C
1,2
3,4,5,6
7,,,,8,9
10
"""
        splitter = CSVSplitter(detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) >= 1
        combined = "\n".join(d.content for d in docs)
        assert "A,B,C" in combined
        assert "7,,," in combined or "7,," in combined
        assert "8,9" in combined

    def test_single_column_many_rows(self) -> None:
        """
        Single column with blank lines => multiple vertical blocks by BFS.
        Checks that Val3 and Val4 appear together.
        """
        csv_data = """OnlyCol
Val1

Val2

Val3
Val4
"""
        splitter = CSVSplitter(split_threshold=1, detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) == 3
        block_texts = [d.content for d in docs]
        assert any("Val1" in txt for txt in block_texts)
        assert any("Val2" in txt for txt in block_texts)
        normalized_blocks = [b.replace("\r", "") for b in block_texts]
        assert any("Val3\nVal4" in b for b in normalized_blocks)

    def test_multiple_side_tables_and_skipped_errors(self) -> None:
        """
        Default Python CSV won't raise error on "Broken,Row => BFS sees a multiline cell, not an error.
        """
        csv_data = """Left1,Left2,,,Right1,Right2
A,B,,,C,D
"Broken,Row
X,Y,,,Z,W
"""
        splitter = CSVSplitter(skip_errors=True, detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) > 0

    def test_multiple_side_tables_and_no_skipped_errors(self) -> None:
        """
        Same broken CSV row, skip_errors=False => no error is raised unless we set strict=True.
        """
        csv_data = """Left1,Left2,,,Right1,Right2
A,B,,,C,D
"Broken,Row
X,Y,,,Z,W
"""
        splitter = CSVSplitter(skip_errors=False, detect_side_tables=True)
        result = splitter.run([Document(content=csv_data)])
        docs = result["documents"]
        assert len(docs) > 0


if __name__ == "__main__":
    pytest.main([__file__])
