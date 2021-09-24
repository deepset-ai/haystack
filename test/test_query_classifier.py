from haystack.pipeline import (
    Pipeline,
    RootNode,
)
from haystack.classifier import SklearnQueryClassifier, TransformersQueryClassifier


def test_query_keyword_statement_classifier():
    class KeywordOutput(RootNode):
        outgoing_edges = 2

        def run(self, **kwargs):
            kwargs["output"] = "keyword"
            return kwargs, "output_1"

    class QuestionOutput(RootNode):
        outgoing_edges = 2

        def run(self, **kwargs):
            kwargs["output"] = "question"
            return kwargs, "output_2"

    pipeline = Pipeline()
    pipeline.add_node(
        name="SkQueryKeywordQuestionClassifier",
        component=SklearnQueryClassifier(),
        inputs=["Query"],
    )
    pipeline.add_node(
        name="KeywordNode",
        component=KeywordOutput(),
        inputs=["SkQueryKeywordQuestionClassifier.output_2"],
    )
    pipeline.add_node(
        name="QuestionNode",
        component=QuestionOutput(),
        inputs=["SkQueryKeywordQuestionClassifier.output_1"],
    )
    output = pipeline.run(query="morse code")
    assert output["output"] == "keyword"

    output = pipeline.run(query="How old is John?")
    assert output["output"] == "question"

    pipeline = Pipeline()
    pipeline.add_node(
        name="TfQueryKeywordQuestionClassifier",
        component=TransformersQueryClassifier(),
        inputs=["Query"],
    )
    pipeline.add_node(
        name="KeywordNode",
        component=KeywordOutput(),
        inputs=["TfQueryKeywordQuestionClassifier.output_2"],
    )
    pipeline.add_node(
        name="QuestionNode",
        component=QuestionOutput(),
        inputs=["TfQueryKeywordQuestionClassifier.output_1"],
    )
    output = pipeline.run(query="morse code")
    assert output["output"] == "keyword"

    output = pipeline.run(query="How old is John?")
    assert output["output"] == "question"
