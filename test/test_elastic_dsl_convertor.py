import pytest

from rest_api.controller.request import Question
from rest_api.controller.response import Answer, AnswersToIndividualQuestion


def test_query_dsl_with_without_valid_query_field():
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"title": "Search"}},
                    {"match": {"content": "Elasticsearch"}}
                ],
                "filter": [
                    {"term": {"status": "published"}},
                    {"range": {"publish_date": {"gte": "2015-01-01"}}}
                ]
            }
        }
    }
    with pytest.raises(Exception):
        Question.from_elastic_query_dsl(query)


def test_query_dsl_with_without_multiple_query_field():
    query = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"name.first": {"query": "shay", "_name": "first"}}},
                    {"match": {"name.last": {"query": "banon", "_name": "last"}}}
                ],
                "filter": {
                    "terms": {
                        "name.last": ["banon", "kimchy"],
                        "_name": "test"
                    }
                }
            }
        }
    }
    with pytest.raises(Exception):
        Question.from_elastic_query_dsl(query)


def test_query_dsl_with_single_query():
    query = {
        "query": {
            "match": {
                "message": {
                    "query": "this is a test"
                }
            }
        }
    }
    question = Question.from_elastic_query_dsl(query)
    assert 1 == len(question.questions)
    assert question.questions.__contains__("this is a test")
    assert question.filters is None


def test_query_dsl_with_filter():
    query = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"name.first": {"query": "shay", "_name": "first"}}}
                ],
                "filter": {
                    "terms": {
                        "name.last": ["banon", "kimchy"],
                        "_name": "test"
                    }
                }
            }
        }
    }
    question = Question.from_elastic_query_dsl(query)
    assert 1 == len(question.questions)
    assert question.questions.__contains__("shay")
    assert len(question.filters) == 1
    assert question.filters["_name"] == "test"


def test_query_dsl_with_complex_query():
    query = {
        "size": 17,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": "I am test1",
                            "type": "most_fields",
                            "fields": ["text", "title"]
                        }
                    }
                ],
                "filter": [
                    {
                        "terms": {
                            "year": "2020"
                        }
                    },
                    {
                        "terms": {
                            "quarter": "1"
                        }
                    },
                    {
                        "range": {
                            "date": {
                                "gte": "12-12-12"
                            }
                        }
                    }
                ]
            }
        }
    }
    top_k_reader = 7
    question = Question.from_elastic_query_dsl(query, top_k_reader)
    assert 1 == len(question.questions)
    assert question.questions.__contains__("I am test1")
    assert 2 == len(question.filters)
    assert question.filters["year"] == "2020"
    assert question.filters["quarter"] == "1"
    assert 17 == question.top_k_retriever
    assert 7 == question.top_k_reader


def test_response_dsl_with_empty_answers():
    sample_answer = AnswersToIndividualQuestion(question="test question", answers=[])
    response = AnswersToIndividualQuestion.to_elastic_response_dsl(sample_answer.__dict__)
    assert 0 == response['hits']['total']['value']
    assert 0 == len(response['hits']['hits'])


def test_response_dsl_with_answers():
    full_answer = Answer(
        answer="answer",
        question="question",
        score=0.1234,
        probability=0.5678,
        context="context",
        offset_start=200,
        offset_end=300,
        offset_start_in_doc=2000,
        offset_end_in_doc=2100,
        document_id="id_1",
        meta={
            "meta1": "meta_value"
        }
    )
    empty_answer = Answer(
        answer=None,
        question=None,
        score=None,
        probability=None,
        context=None,
        offset_start=250,
        offset_end=350,
        offset_start_in_doc=None,
        offset_end_in_doc=None,
        document_id=None,
        meta=None
    )
    sample_answer = AnswersToIndividualQuestion(question="test question", answers=[full_answer, empty_answer])
    response = AnswersToIndividualQuestion.to_elastic_response_dsl(sample_answer.__dict__)

    # Test number of returned answers
    assert response['hits']['total']['value'] == 2

    # Test converted answers
    hits = response['hits']['hits']
    assert len(hits) == 2
    # Test full answer record
    assert hits[0]["_score"] == 0.1234
    assert hits[0]["_id"] == "id_1"
    assert hits[0]["_source"]["answer"] == "answer"
    assert hits[0]["_source"]["question"] == "question"
    assert hits[0]["_source"]["context"] == "context"
    assert hits[0]["_source"]["probability"] == 0.5678
    assert hits[0]["_source"]["offset_start"] == 200
    assert hits[0]["_source"]["offset_end"] == 300
    assert hits[0]["_source"]["offset_start_in_doc"] == 2000
    assert hits[0]["_source"]["offset_end_in_doc"] == 2100
    assert hits[0]["_source"]["meta"] == {"meta1": "meta_value"}
    # Test empty answer record
    assert hits[1]["_score"] is None
    assert hits[1]["_id"] is None
    assert hits[1]["_source"]["answer"] is None
    assert hits[1]["_source"]["question"] is None
    assert hits[1]["_source"]["context"] is None
    assert hits[1]["_source"]["probability"] is None
    assert hits[1]["_source"]["offset_start"] == 250
    assert hits[1]["_source"]["offset_end"] == 350
    assert hits[1]["_source"]["offset_start_in_doc"] is None
    assert hits[1]["_source"]["offset_end_in_doc"] is None
    assert hits[1]["_source"]["meta"] is None
