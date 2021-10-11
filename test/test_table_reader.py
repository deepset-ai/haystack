import pandas as pd

from haystack import Document, Pipeline
from haystack.reader.transformers import TableReader


def test_table_reader():
    table_reader = TableReader("google/tapas-base-finetuned-wtq")
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["57", "46", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
    }

    table = pd.DataFrame(data)
    query = "When was DiCaprio born?"

    prediction = table_reader.predict(query=query, documents=[Document(content=table, content_type="table")])
    assert prediction["answers"][0].answer == "10 june 1996"
    assert prediction["answers"][0].offsets_in_context[0].start == 7
    assert prediction["answers"][0].offsets_in_context[0].end == 8

    # test aggregation
    query = "How old are DiCaprio and Pitt on average?"
    prediction = table_reader.predict(query=query, documents=[Document(content=table, content_type="table")])
    assert prediction["answers"][0].answer == 51.5
    assert prediction["answers"][0].meta["answer_cells"] == ["57", "46"]
    assert prediction["answers"][0].meta["aggregation_operator"] == "AVERAGE"
    assert prediction["answers"][0].offsets_in_context[0].start == 1
    assert prediction["answers"][0].offsets_in_context[0].end == 2
    assert prediction["answers"][0].offsets_in_context[1].start == 5
    assert prediction["answers"][0].offsets_in_context[1].end == 6


def test_table_reader_in_pipeline():
    table_reader = TableReader("google/tapas-base-finetuned-wtq")
    pipeline = Pipeline()
    pipeline.add_node(table_reader, "TableReader", ["Query"])
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["57", "46", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
    }

    table = pd.DataFrame(data)
    query = "Which actors played in more than 60 movies?"

    prediction = pipeline.run(query=query, documents=[Document(content=table, content_type="table")])

    assert prediction["answers"][0].answer == "brad pitt, george clooney"
    assert prediction["answers"][0].meta["aggregation_operator"] == "NONE"
    assert prediction["answers"][0].offsets_in_context[0].start == 0
    assert prediction["answers"][0].offsets_in_context[0].end == 1
    assert prediction["answers"][0].offsets_in_context[1].start == 8
    assert prediction["answers"][0].offsets_in_context[1].end == 9
