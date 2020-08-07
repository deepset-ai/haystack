import pytest

from haystack.retriever.dense import DensePassageRetriever


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
def test_dpr_inmemory_retrieval(document_store):
    documents = [
        {'name': '0', 'text': """Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from"""},
        {'name': '1', 'text': """Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are"""},
        {'name': '2', 'text': """Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with"""},
    ]

    retriever = DensePassageRetriever(document_store=document_store, embedding_model="dpr-bert-base-nq", use_gpu=False)

    embedded = []
    for doc in documents:
        embedding = retriever.embed_passages([doc['text']])[0]
        doc['embedding'] = embedding
        embedded.append(doc)

        assert (embedding.shape[0] == 768)
        assert (embedding[0] - 0.52872 < 0.001)

    document_store.write_documents(embedded)

    res = retriever.retrieve(query="Which philosopher attacked Schopenhauer?")
    assert res[0].meta["name"] == "1"
