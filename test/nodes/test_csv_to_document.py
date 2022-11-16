import csv
from pathlib import Path
from typing import List

import pytest

from haystack import Document
from haystack.nodes.file_converter import CsvToDocuments


def write_as_csv(data: List[List[str]], file_path: Path):
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


@pytest.mark.integration
def test_csv_to_document_with_qa_headers(tmp_path):
    node = CsvToDocuments()
    csv_path = tmp_path / "csv_qa_with_headers.csv"
    rows = [
        ["question", "answer"],
        ["What is Haystack ?", "Haystack is an NLP Framework to use transformers in your Applications."],
    ]
    write_as_csv(rows, csv_path)

    output, edge = node.run(file_paths=csv_path)
    assert edge == "output_1"
    assert "documents" in output
    assert len(output["documents"]) == 1

    doc = output["documents"][0]
    assert isinstance(doc, Document)
    assert doc.content == "What is Haystack ?"
    assert doc.meta["answer"] == "Haystack is an NLP Framework to use transformers in your Applications."
