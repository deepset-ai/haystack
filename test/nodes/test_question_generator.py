from typing import List

import pytest

from haystack.pipelines import (
    QuestionAnswerGenerationPipeline,
    QuestionGenerationPipeline,
    RetrieverQuestionGenerationPipeline,
)
from haystack.nodes.question_generator import QuestionGenerator
from haystack.schema import Document


text = (
    "The Living End are an Australian punk rockabilly band from Melbourne, formed in 1994. Since 2002, "
    "the line-up consists of Chris Cheney (vocals, guitar), Scott Owen (double bass, vocals), and Andy "
    "Strachan (drums). The band rose to fame in 1997 after the release of their EP Second Solution / Prisoner "
    "of Society, which peaked at No. 4 on the Australian ARIA Singles Chart. They have released eight studio "
    "albums, two of which reached the No. 1 spot on the ARIA Albums Chart: The Living End (October 1998) and "
    "State of Emergency (February 2006). They have also achieved chart success in the U.S. and the United "
    "Kingdom. The Band was nominated 27 times and won five awards at the Australian ARIA Music Awards "
    'ceremonies: "Highest Selling Single" for Second Solution / Prisoner of Society (1998), "Breakthrough '
    'Artist – Album" and "Best Group" for The Living End (1999), as well as "Best Rock Album" for White '
    "Noise (2008) and The Ending Is Just the Beginning Repeating (2011). In October 2010, their debut album "
    'was listed in the book "100 Best Australian Albums". Australian musicologist Ian McFarlane described '
    'the group as "one of Australia’s premier rock acts. By blending a range of styles (punk, rockabilly '
    "and flat out rock) with great success, The Living End has managed to produce anthemic choruses and "
    'memorable songs in abundance".'
)
document = Document(content=text)
query = "Living End"
keywords = [
    "Australian",
    "punk",
    "drummer",
    "Living",
    "band",
    "Band",
    "Second",
    "album",
    "albums",
    "dialect",
    "music",
    "book",
    "group",
    "produce",
    "Music",
    "Awards",
    "year",
    "released",
]
text_2 = (
    "Berlin straddles the banks of the Spree, which flows into the Havel (a tributary of the Elbe) in the "
    "western borough of Spandau. Among the city's main topographical features are the many lakes in the western "
    "and southeastern boroughs formed by the Spree, Havel and Dahme, the largest of which is Lake Müggelsee. "
    "Due to its location in the European Plain, Berlin is influenced by a temperate seasonal climate. About "
    "one-third of the city's area is composed of forests, parks, gardens, rivers, canals and lakes. The city "
    "lies in the Central German dialect area, the Berlin dialect being a variant of the Lusatian-New Marchian "
    "dialects."
)
document_2 = Document(content=text_2)
keywords_2 = [
    "Berlin",
    "Elbe",
    "Spandau",
    "Spree",
    "boroughs",
    "lakes",
    "largest",
    "seasonal",
    "climate",
    "city",
    "dialect",
    "German",
]


def test_qg_pipeline(question_generator):
    p = QuestionGenerationPipeline(question_generator)
    result = p.run(documents=[document])
    keys = list(result)
    assert "generated_questions" in keys
    assert len(result["generated_questions"][0]["questions"]) > 0


def test_qg_pipeline_non_default_params():
    question_generator = QuestionGenerator(model_name_or_path="valhalla/t5-small-e2e-qg", num_queries_per_doc=2)
    p = QuestionGenerationPipeline(question_generator)
    result = p.run(documents=[document, document_2])
    assert isinstance(result, dict)
    assert "generated_questions" in result
    assert "documents" in result
    assert isinstance(result["generated_questions"], list)
    assert isinstance(result["documents"], list)
    assert len(result["generated_questions"]) == 2
    assert len(result["documents"]) == 2
    assert len(result["generated_questions"][0]["questions"]) == 26
    assert len(result["generated_questions"][1]["questions"]) == 12

    # first list of questions should be about Australian punk band
    verify_questions(result["generated_questions"][0]["questions"], keywords)
    # second list of questions should be about Berlin
    verify_questions(result["generated_questions"][1]["questions"], keywords_2)


@pytest.mark.parametrize("split_length, num_queries_per_doc", [(50, 1), (50, 2), (50, 3), (100, 1), (100, 2), (100, 3)])
def test_qa_generator_non_default_params(split_length, num_queries_per_doc):
    question_generator = QuestionGenerator(
        model_name_or_path="valhalla/t5-small-e2e-qg",
        split_length=split_length,
        num_queries_per_doc=num_queries_per_doc,
    )
    questions = question_generator.generate_batch(texts=[document.content, document_2.content])
    assert isinstance(questions, list)
    assert len(questions) == 2
    assert isinstance(questions[0], list)
    assert isinstance(questions[1], list)
    assert len(questions[0]) > 0
    assert len(questions[1]) > 0

    # first list of questions should be about Australian punk band
    verify_questions(questions[0], keywords)
    # second list of questions should be about Berlin
    verify_questions(questions[1], keywords_2)


@pytest.mark.parametrize("retriever,document_store", [("tfidf", "memory")], indirect=True)
def test_rqg_pipeline(question_generator, retriever):
    retriever.document_store.write_documents([document])
    p = RetrieverQuestionGenerationPipeline(retriever, question_generator)
    result = p.run(query)
    keys = list(result)
    assert "generated_questions" in keys
    assert len(result["generated_questions"][0]["questions"]) > 0


@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_qag_pipeline(question_generator, reader):
    p = QuestionAnswerGenerationPipeline(question_generator, reader)
    results = p.run(documents=[document])
    assert "queries" in results
    assert "answers" in results
    assert len(results["queries"]) == len(results["answers"])
    assert len(results["answers"]) > 0
    assert results["answers"][0][0].answer is not None


def verify_questions(questions: List[str], question_keywords: List[str]):
    for q in questions:
        assert any(word in q for word in question_keywords)
