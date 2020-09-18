import pytest
import time

from haystack.retriever.dense import DensePassageRetriever
from haystack import Document
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
def test_dpr_inmemory_retrieval(document_store):

    documents = [
        Document(
            text="""Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from""",
            meta={"name": "0"}
        ),
        Document(
            text="""Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with""",
        ),
        Document(
            text="""Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are""",
            meta={"name": "1"}
        ),
        Document(
            text="""The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio""",
            meta={"name": "2"}
        ),
        Document(
            text="""The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin""",
            meta={}
        )
    ]

    document_store.delete_all_documents(index="test_dpr")
    document_store.write_documents(documents, index="test_dpr")
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      use_gpu=True, embed_title=True,
                                      remove_sep_tok_from_untitled_passages=True)
    document_store.update_embeddings(retriever=retriever, index="test_dpr")
    time.sleep(2)

    docs_with_emb = document_store.get_all_documents(index="test_dpr")

    # FAISSDocumentStore doesn't return embeddings, so these tests only work with ElasticsearchDocumentStore
    if isinstance(document_store, ElasticsearchDocumentStore):
        assert (len(docs_with_emb[0].embedding) == 768)
        assert (abs(docs_with_emb[0].embedding[0] - (-0.30634)) < 0.001)
        assert (abs(docs_with_emb[1].embedding[0] - (-0.37449)) < 0.001)
        assert (abs(docs_with_emb[2].embedding[0] - (-0.24695)) < 0.001)
        assert (abs(docs_with_emb[3].embedding[0] - (-0.08017)) < 0.001)
        assert (abs(docs_with_emb[4].embedding[0] - (-0.01534)) < 0.001)
    res = retriever.retrieve(query="Which philosopher attacked Schopenhauer?", index="test_dpr")
    assert res[0].meta["name"] == "1"

    # clean up
    document_store.delete_all_documents(index="test_dpr")
