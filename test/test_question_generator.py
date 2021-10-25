from haystack.pipeline import QuestionAnswerGenerationPipeline, QuestionGenerationPipeline, RetrieverQuestionGenerationPipeline
from haystack.schema import Document
import pytest


text = 'The Living End are an Australian punk rockabilly band from Melbourne, formed in 1994. Since 2002, the line-up consists of Chris Cheney (vocals, guitar), Scott Owen (double bass, vocals), and Andy Strachan (drums). The band rose to fame in 1997 after the release of their EP Second Solution / Prisoner of Society, which peaked at No. 4 on the Australian ARIA Singles Chart. They have released eight studio albums, two of which reached the No. 1 spot on the ARIA Albums Chart: The Living End (October 1998) and State of Emergency (February 2006). They have also achieved chart success in the U.S. and the United Kingdom. The Band was nominated 27 times and won five awards at the Australian ARIA Music Awards ceremonies: "Highest Selling Single" for Second Solution / Prisoner of Society (1998), "Breakthrough Artist – Album" and "Best Group" for The Living End (1999), as well as "Best Rock Album" for White Noise (2008) and The Ending Is Just the Beginning Repeating (2011). In October 2010, their debut album was listed in the book "100 Best Australian Albums". Australian musicologist Ian McFarlane described the group as "one of Australia’s premier rock acts. By blending a range of styles (punk, rockabilly and flat out rock) with great success, The Living End has managed to produce anthemic choruses and memorable songs in abundance".'
document = Document(content=text)
query = "Living End"


def test_qg_pipeline(question_generator):
    p = QuestionGenerationPipeline(question_generator)
    result = p.run(documents=[document])
    keys = list(result)
    assert "generated_questions" in keys
    assert len(result["generated_questions"][0]["questions"]) > 0


@pytest.mark.parametrize("retriever,document_store", [("tfidf", "memory")], indirect=True)
def test_rqg_pipeline(question_generator, retriever):
    retriever.document_store.write_documents([document])
    retriever.fit()
    p = RetrieverQuestionGenerationPipeline(retriever, question_generator)
    result = p.run(query)
    keys = list(result)
    assert "generated_questions" in keys
    assert len(result["generated_questions"][0]["questions"]) > 0


@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_qag_pipeline(question_generator, reader):
    p = QuestionAnswerGenerationPipeline(question_generator, reader)
    results = p.run(documents=[document])["results"]
    assert len(results) > 0
    assert results[0]["query"]
    assert len(results[0]["answers"]) > 0
    assert results[0]["answers"][0].answer is not None

