from examples.basic_faq_pipeline import basic_faq_pipeline

from haystack.schema import Answer, Document


def test_basic_faq_pipeline():
    prediction = basic_faq_pipeline()

    assert prediction is not None
    assert prediction["query"] == "How is the virus spreading?"

    assert len(prediction["answers"]) == 10  # top-k of Retriever
    assert type(prediction["answers"][0]) == Answer
    assert (
        prediction["answers"][0].answer
        == """This virus was first detected in Wuhan City, Hubei Province, China. The first infections were linked to a live animal market, but the virus is now spreading from person-to-person. It’s important to note that person-to-person spread can happen on a continuum. Some viruses are highly contagious (like measles), while other viruses are less so.\n\nThe virus that causes COVID-19 seems to be spreading easily and sustainably in the community (“community spread”) in some affected geographic areas. Community spread means people have been infected with the virus in an area, including some who are not sure how or where they became infected.\n\nLearn what is known about the spread of newly emerged coronaviruses."""
    )
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
