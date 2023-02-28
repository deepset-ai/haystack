from haystack.nodes import TopPSampler
from haystack.schema import Document

docs = [
    Document(
        content="""Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from""",
        meta={"name": "0"},
        id="1",
    ),
    Document(
        content="""Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with""",
        id="2",
    ),
    Document(
        content="""Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are""",
        meta={"name": "1"},
        id="3",
    ),
    Document(
        content="""The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio""",
        meta={"name": "2"},
        id="4",
    ),
    Document(
        content="""The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin""",
        meta={},
        id="5",
    ),
]


def test_top_p_sampling(top_p_sampler):
    query = "What is the most important religious learning from the Bible?"
    results = top_p_sampler.predict(query=query, documents=docs, top_p=0.98)
    assert len(results) == 2
    assert results[0].id == "1"
    assert results[1].id == "5"


def test_top_p_sampling_at_least_one_result(top_p_sampler):
    query = "What is the most important building in King's Landing that has a religious background?"
    results = top_p_sampler.predict(query=query, documents=docs, top_p=0.9)

    # The top_p sampler should always return at least one result if strict is False (default)
    # in this case the result with the highest prob score is returned even if top p is too low
    assert len(results) == 1
    assert results[0].id == "5"

    # unless strict is set to True
    sampler = TopPSampler(strict=True)
    results = sampler.predict(query=query, documents=docs, top_p=0.9)
    assert len(results) == 0
