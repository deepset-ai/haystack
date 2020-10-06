import pytest

from rest_api.controller.search import Question


def test_query_dsl_with_invalid_query():
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
    assert len(question.questions) == 1
    assert question.questions.__contains__("this is a test")
    assert question.filters is None


def test_query_dsl_with_filter():
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
    question = Question.from_elastic_query_dsl(query)
    assert len(question.questions) == 2
    assert question.questions.__contains__("shay")
    assert question.questions.__contains__("banon")
    assert len(question.filters) == 1
    assert question.filters["_name"] == "test"


def test_query_dsl_with_complex_query():
    query = {
        "size": 10,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": "I am test1",
                            "type": "most_fields",
                            "fields": ["text", "title"]
                        }
                    },
                    {
                        "multi_match": {
                            "query": "I am test2",
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
    question = Question.from_elastic_query_dsl(query)
    assert len(question.questions) == 2
    assert question.questions.__contains__("I am test1")
    assert question.questions.__contains__("I am test2")
    assert len(question.filters) == 2
    assert question.filters["year"] == "2020"
    assert question.filters["quarter"] == "1"
    assert question.top_k_retriever == 10

