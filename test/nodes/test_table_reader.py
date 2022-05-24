import logging

import pandas as pd
import pytest

from haystack.schema import Document, Answer
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


def test_table_reader_batch_single_query_single_doc_list(table_reader):
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["58", "47", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    table = pd.DataFrame(data)

    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(queries=[query], documents=[Document(content=table, content_type="table")])
    # Expected output: List of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 1  # Predictions for 5 docs


def test_table_reader_batch_single_query_multiple_doc_lists(table_reader):
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["58", "47", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    table = pd.DataFrame(data)

    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(
        queries=[query], documents=[[Document(content=table, content_type="table")]]
    )
    # Expected output: List of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 1  # Predictions for 1 collection of docs


def test_table_reader_batch_multiple_queries_single_doc_list(table_reader):
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["58", "47", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    table = pd.DataFrame(data)

    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(
        queries=[query, query], documents=[Document(content=table, content_type="table")]
    )
    # Expected output: List of lists of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], list)
    assert isinstance(prediction["answers"][0][0][0], Answer)
    assert len(prediction["answers"]) == 2  # Predictions for 2 queries


def test_table_reader_batch_multiple_queries_multiple_doc_lists(table_reader):
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["58", "47", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    table = pd.DataFrame(data)

    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(
        queries=[query, query],
        documents=[[Document(content=table, content_type="table")], [Document(content=table, content_type="table")]],
    )
    # Expected output: List of lists answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 2  # Predictions for 2 collections of documents


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


def test_table_without_rows(caplog, table_reader):
    # empty DataFrame
    table = pd.DataFrame()
    document = Document(content=table, content_type="table", id="no_rows")
    with caplog.at_level(logging.WARNING):
        predictions = table_reader.predict(query="test", documents=[document])
        assert "Skipping document with id 'no_rows'" in caplog.text
        assert len(predictions["answers"]) == 0


def test_text_document(caplog, table_reader):
    document = Document(content="text", id="text_doc")
    with caplog.at_level(logging.WARNING):
        predictions = table_reader.predict(query="test", documents=[document])
        assert "Skipping document with id 'text_doc'" in caplog.text
        assert len(predictions["answers"]) == 0
