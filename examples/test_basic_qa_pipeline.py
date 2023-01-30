from examples.basic_qa_pipeline import basic_qa_pipeline

from haystack.schema import Answer, Document


def test_basic_qa_pipeline():
    prediction = basic_qa_pipeline()

    assert prediction is not None
    assert prediction["query"] == "Who is the father of Arya Stark?"

    assert len(prediction["answers"]) == 5  # top-k of Reader
    assert type(prediction["answers"][0]) == Answer
    assert prediction["answers"][0].answer == "Ned"
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].meta["name"] == "43_Arya_Stark.txt"

    assert len(prediction["documents"]) == 10  # top-k of Retriever
    assert type(prediction["documents"][0]) == Document
    assert prediction["documents"][0].score <= 1
    assert prediction["documents"][0].score >= 0
    assert prediction["documents"][0].meta["name"] == "450_Baelor.txt"
