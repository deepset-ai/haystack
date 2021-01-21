import pytest
import time
import numpy as np

from haystack import Document
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever

from transformers import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast

@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("return_embedding", [True, False])
def test_dpr_retrieval(document_store, retriever, return_embedding):

    documents = [
        Document(
            text="""Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from""",
            meta={"name": "0"},
            id="1",
        ),
        Document(
            text="""Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with""",
            id="2",
        ),
        Document(
            text="""Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are""",
            meta={"name": "1"},
            id="3",
        ),
        Document(
            text="""The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio""",
            meta={"name": "2"},
            id="4",
        ),
        Document(
            text="""The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin""",
            meta={},
            id="5",
        )
    ]

    document_store.return_embedding = return_embedding
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever=retriever)
    time.sleep(1)

    if return_embedding is True:
        doc_1 = document_store.get_document_by_id("1")
        assert (len(doc_1.embedding) == 768)
        assert (abs(doc_1.embedding[0] - (-0.3063)) < 0.001)
        doc_2 = document_store.get_document_by_id("2")
        assert (abs(doc_2.embedding[0] - (-0.3914)) < 0.001)
        doc_3 = document_store.get_document_by_id("3")
        assert (abs(doc_3.embedding[0] - (-0.2470)) < 0.001)
        doc_4 = document_store.get_document_by_id("4")
        assert (abs(doc_4.embedding[0] - (-0.0802)) < 0.001)
        doc_5 = document_store.get_document_by_id("5")
        assert (abs(doc_5.embedding[0] - (-0.0551)) < 0.001)

    res = retriever.retrieve(query="Which philosopher attacked Schopenhauer?")

    assert res[0].meta["name"] == "1"

    # test embedding
    if return_embedding is True:
        assert res[0].embedding is not None
    else:
        assert res[0].embedding is None

    # test filtering
    if not isinstance(document_store, FAISSDocumentStore):
        res = retriever.retrieve(query="Which philosopher attacked Schopenhauer?", filters={"name": ["0", "2"]})
        assert len(res) == 2
        for r in res:
            assert r.meta["name"] in ["0", "2"]


@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_dpr_saving_and_loading(retriever, document_store):
    retriever.save("test_dpr_save")
    def sum_params(model):
        s = []
        for p in model.parameters():
            n = p.cpu().data.numpy()
            s.append(np.sum(n))
        return sum(s)
    original_sum_query = sum_params(retriever.query_encoder)
    original_sum_passage = sum_params(retriever.passage_encoder)
    del retriever

    loaded_retriever = DensePassageRetriever.load("test_dpr_save", document_store)

    loaded_sum_query = sum_params(loaded_retriever.query_encoder)
    loaded_sum_passage = sum_params(loaded_retriever.passage_encoder)

    assert abs(original_sum_query - loaded_sum_query) < 0.1
    assert abs(original_sum_passage - loaded_sum_passage) < 0.1

    # comparison of weights (RAM intense!)
    # for p1, p2 in zip(retriever.query_encoder.parameters(), loaded_retriever.query_encoder.parameters()):
    #     assert (p1.data.ne(p2.data).sum() == 0)
    #
    # for p1, p2 in zip(retriever.passage_encoder.parameters(), loaded_retriever.passage_encoder.parameters()):
    #     assert (p1.data.ne(p2.data).sum() == 0)

    # attributes
    assert loaded_retriever.embed_title == True
    assert loaded_retriever.batch_size == 16
    assert loaded_retriever.max_seq_len_passage == 256
    assert loaded_retriever.max_seq_len_query == 64

    # Tokenizer
    assert isinstance(loaded_retriever.passage_tokenizer, DPRContextEncoderTokenizerFast)
    assert isinstance(loaded_retriever.query_tokenizer, DPRQuestionEncoderTokenizerFast)
    assert loaded_retriever.passage_tokenizer.do_lower_case == True
    assert loaded_retriever.query_tokenizer.do_lower_case == True
    assert loaded_retriever.passage_tokenizer.vocab_size == 30522
    assert loaded_retriever.query_tokenizer.vocab_size == 30522
    assert loaded_retriever.passage_tokenizer.model_max_length == 512
    assert loaded_retriever.query_tokenizer.model_max_length == 512

