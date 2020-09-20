import faiss
import numpy as np
import pytest

from haystack import Document
from haystack.document_store.faiss import FAISSDocumentStore, FaissIndexStore

from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.dense import EmbeddingRetriever
from haystack import Finder

DOCUMENTS = [
    {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
    {"name": "name_4", "text": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_5", "text": "text_5", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_6", "text": "text_6", "embedding": np.random.rand(768).astype(np.float64)},
]


def check_data_correctness(documents_indexed, documents_inserted):
    # test if correct vector_ids are assigned
    for i, doc in enumerate(documents_indexed):
        assert doc.meta["vector_id"] == str(i)

    # test if number of documents is correct
    assert len(documents_indexed) == len(documents_inserted)

    # test if two docs have same vector_is assigned
    vector_ids = set()
    for i, doc in enumerate(documents_indexed):
        vector_ids.add(doc.meta["vector_id"])
    assert len(vector_ids) == len(documents_inserted)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_index_save_and_load(document_store):
    document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save("haystack_test_faiss")

    # clear existing faiss_index
    document_store.faiss_index.reset()

    # test faiss index is cleared
    assert document_store.faiss_index.ntotal == 0

    # test loading the index
    new_document_store = document_store.load(sql_url="sqlite:///haystack_test.db", faiss_file_path="haystack_test_faiss")

    # check faiss index is restored
    assert new_document_store.faiss_index.ntotal == len(DOCUMENTS)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
@pytest.mark.parametrize("batch_size", [2])
def test_faiss_write_docs(document_store, index_buffer_size, batch_size):
    document_store.index_buffer_size = index_buffer_size

    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i: i + batch_size])

    documents_indexed = document_store.get_all_documents()

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        # we currently don't get the embeddings back when we call document_store.get_all_documents()
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        stored_emb = document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(original_doc["embedding"], stored_emb[:-1], rtol=0.01)

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
def test_faiss_update_docs(document_store, index_buffer_size):
    # adjust buffer size
    document_store.index_buffer_size = index_buffer_size

    # initial write
    document_store.write_documents(DOCUMENTS)

    # do the update
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      use_gpu=False, embed_title=True,
                                      remove_sep_tok_from_untitled_passages=True)

    document_store.update_embeddings(retriever=retriever)
    documents_indexed = document_store.get_all_documents()

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        updated_embedding = retriever.embed_passages([Document.from_dict(original_doc)])
        stored_emb = document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(updated_embedding, stored_emb[:-1], rtol=0.01)

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_retrieving(document_store):
    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",
                                   use_gpu=False)
    result = retriever.retrieve(query="How to test this?")
    assert len(result) == len(DOCUMENTS)
    assert type(result[0]) == Document


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_finding(document_store):
    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",
                                   use_gpu=False)
    finder = Finder(reader=None, retriever=retriever)

    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert len(prediction.get('answers', [])) == 1


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_indexing_with_different_params(document_store):

    document_store.write_documents(DOCUMENTS, index_factory='IVF4,Flat',
                                   convert_l2_to_ip=False, metric_type=faiss.METRIC_L2, allow_training=True)
    documents_indexed = document_store.get_all_documents()

    # test if number of documents is correct
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if correct vector_ids are assigned
    for i, doc in enumerate(documents_indexed):
        assert doc.meta["vector_id"] == str(i)

    # test insertion of documents in an existing index should pass
    document_store.write_documents(DOCUMENTS, index_factory='IVF4,Flat',
                                   convert_l2_to_ip=False, metric_type=faiss.METRIC_L2, allow_training=True)
    documents_indexed = document_store.get_all_documents()

    # test if number of documents is correct
    assert len(documents_indexed) == 2 * len(DOCUMENTS)

    # test if correct vector_ids are assigned
    for i, doc in enumerate(documents_indexed):
        assert doc.meta["vector_id"] == str(i)

    # test saving the index
    document_store.save("haystack_test_faiss")

    # test loading the index
    document_store.load(sql_url="sqlite:///haystack_test.db", faiss_file_path="haystack_test_faiss")


def test_faiss_write_docs_with_small_buffer_size():
    documents = [
        {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float32)},
    ]

    document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db",
                                        index_buffer_size=2, index="small_buffer_size")
    document_store.delete_all_documents()
    document_store.write_documents(documents)
    documents_indexed = document_store.get_all_documents()

    # test if number of documents is correct
    assert len(documents_indexed) == len(documents)

    # test if two docs have same vector_is assigned
    vector_ids = set()
    for i, doc in enumerate(documents_indexed):
        vector_ids.add(doc.meta["vector_id"])
    assert len(vector_ids) == len(documents)


def test_faiss_write_and_update_docs_with_small_buffer_size():
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

    document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db",
                                        index_buffer_size=2, index="small_buffer_size")
    document_store.delete_all_documents()
    document_store.write_documents(documents)

    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      use_gpu=True, embed_title=True,
                                      remove_sep_tok_from_untitled_passages=True)

    document_store.update_embeddings(retriever=retriever)
    documents_indexed = document_store.get_all_documents()

    # test if number of documents is correct
    assert len(documents_indexed) == len(documents)

    # test if two docs have same vector_is assigned
    vector_ids = set()
    for i, doc in enumerate(documents_indexed):
        vector_ids.add(doc.meta["vector_id"])
    assert len(vector_ids) == len(documents)


def test_faiss_passing_index_from_outside():

    faiss_index = faiss.index_factory(768, 'IDMap,Flat', faiss.METRIC_L2)
    index_store = FaissIndexStore(faiss_index=faiss_index, convert_l2_to_ip=False)

    document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db", custom_index_store=index_store,
                                        index="index_from_outside")
    document_store.delete_all_documents()
    document_store.write_documents(DOCUMENTS)
    documents_indexed = document_store.get_all_documents()

    # test if number of documents is correct
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if two docs have same vector_is assigned
    vector_ids = set()
    for i, doc in enumerate(documents_indexed):
        vector_ids.add(doc.meta["vector_id"])
    assert len(vector_ids) == len(DOCUMENTS)
