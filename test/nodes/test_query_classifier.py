from haystack.nodes.query_classifier.base import BaseQueryClassifier


def test_transformers_query_classifier(transformers_query_classifier):
    assert isinstance(transformers_query_classifier, BaseQueryClassifier)

    output = transformers_query_classifier.run(query="morse code")
    assert output == ({}, "output_2")

    output = transformers_query_classifier.run(query="How old is John?")
    assert output == ({}, "output_1")


def test_transformers_query_classifier_batch(transformers_query_classifier):
    assert isinstance(transformers_query_classifier, BaseQueryClassifier)

    queries = ["morse code", "How old is John?"]
    output = transformers_query_classifier.run_batch(queries=queries)

    assert output[0] == {"output_2": {"queries": ["morse code"]}, "output_1": {"queries": ["How old is John?"]}}


def test_zero_shot_transformers_query_classifier(zero_shot_transformers_query_classifier):
    assert isinstance(zero_shot_transformers_query_classifier, BaseQueryClassifier)

    output = zero_shot_transformers_query_classifier.run(query="What's the answer?")
    assert output == ({}, "output_3")

    output = zero_shot_transformers_query_classifier.run(query="Would you be so kind to tell me the answer?")
    assert output == ({}, "output_1")

    output = zero_shot_transformers_query_classifier.run(query="Can you give me the right answer for once??")
    assert output == ({}, "output_2")


def test_zero_shot_transformers_query_classifier_batch(zero_shot_transformers_query_classifier):
    assert isinstance(zero_shot_transformers_query_classifier, BaseQueryClassifier)

    queries = [
        "What's the answer?",
        "Would you be so kind to tell me the answer?",
        "Can you give me the right answer for once??",
    ]

    output = zero_shot_transformers_query_classifier.run_batch(queries=queries)

    assert output[0] == {
        "output_3": {"queries": ["What's the answer?"]},
        "output_1": {"queries": ["Would you be so kind to tell me the answer?"]},
        "output_2": {"queries": ["Can you give me the right answer for once??"]},
    }
