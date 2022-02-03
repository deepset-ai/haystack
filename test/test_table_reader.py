import pandas as pd
import pytest

from haystack.schema import Document
from haystack.pipelines.base import Pipeline


def test_table_reader(table_reader):
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["58", "47", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    table = pd.DataFrame(data)

    query = "When was Di Caprio born?"
    prediction = table_reader.predict(query=query, documents=[Document(content=table, content_type="table")])
    assert prediction["answers"][0].answer == "11 november 1974"
    assert prediction["answers"][0].offsets_in_context[0].start == 7
    assert prediction["answers"][0].offsets_in_context[0].end == 8


def test_table_reader_in_pipeline(table_reader):
    pipeline = Pipeline()
    pipeline.add_node(table_reader, "TableReader", ["Query"])
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["58", "47", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }

    table = pd.DataFrame(data)
    query = "When was Di Caprio born?"

    prediction = pipeline.run(query=query, documents=[Document(content=table, content_type="table")])

    assert prediction["answers"][0].answer == "11 november 1974"
    assert prediction["answers"][0].offsets_in_context[0].start == 7
    assert prediction["answers"][0].offsets_in_context[0].end == 8


@pytest.mark.parametrize("table_reader", ["tapas"], indirect=True)
def test_table_reader_aggregation(table_reader):
    data = {
        "Mountain": ["Mount Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu"],
        "Height": ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"],
    }
    table = pd.DataFrame(data)

    query = "How tall are all mountains on average?"
    prediction = table_reader.predict(query=query, documents=[Document(content=table, content_type="table")])
    assert prediction["answers"][0].answer == "8609.2 m"
    assert prediction["answers"][0].meta["aggregation_operator"] == "AVERAGE"
    assert prediction["answers"][0].meta["answer_cells"] == ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"]

    query = "How tall are all mountains together?"
    prediction = table_reader.predict(query=query, documents=[Document(content=table, content_type="table")])
    assert prediction["answers"][0].answer == "43046.0 m"
    assert prediction["answers"][0].meta["aggregation_operator"] == "SUM"
    assert prediction["answers"][0].meta["answer_cells"] == ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"]
