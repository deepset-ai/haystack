import uuid
import faiss
import math
import numpy as np
import pytest
import sys

from haystack.schema import Document
from haystack.pipelines import DocumentSearchPipeline
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.document_stores.weaviate import WeaviateDocumentStore

from haystack.pipelines import Pipeline
from haystack.nodes.retriever.dense import EmbeddingRetriever

DOCUMENTS = [
    {"meta": {"name": "name_1", "year": "2020", "month": "01"}, "content": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
    {"meta": {"name": "name_2", "year": "2020", "month": "02"}, "content": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
    {"meta": {"name": "name_3", "year": "2020", "month": "03"}, "content": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
    {"meta": {"name": "name_4", "year": "2021", "month": "01"}, "content": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    {"meta": {"name": "name_5", "year": "2021", "month": "02"}, "content": "text_5", "embedding": np.random.rand(768).astype(np.float32)},
    {"meta": {"name": "name_6", "year": "2021", "month": "03"}, "content": "text_6", "embedding": np.random.rand(768).astype(np.float64)},
]


@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason="Test with tmp_path not working on windows runner")
def test_faiss_index_save_and_load(tmp_path):
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'haystack_test.db'}",
        index="haystack_test",
        progress_bar=False  # Just to check if the init parameters are kept
    )
    document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save(tmp_path / "haystack_test_faiss")

    # clear existing faiss_index
    document_store.faiss_indexes[document_store.index].reset()

    # test faiss index is cleared
    assert document_store.faiss_indexes[document_store.index].ntotal == 0

    # test loading the index
    new_document_store = FAISSDocumentStore.load(tmp_path / "haystack_test_faiss")

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar

    # test loading the index via init
    new_document_store = FAISSDocumentStore(faiss_index_path=tmp_path / "haystack_test_faiss")

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar


@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason="Test with tmp_path not working on windows runner")
def test_faiss_index_save_and_load_custom_path(tmp_path):
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'haystack_test.db'}",
        index="haystack_test",
        progress_bar=False  # Just to check if the init parameters are kept
    )
    document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save(index_path=tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json")

    # clear existing faiss_index
    document_store.faiss_indexes[document_store.index].reset()

    # test faiss index is cleared
    assert document_store.faiss_indexes[document_store.index].ntotal == 0

    # test loading the index
    new_document_store = FAISSDocumentStore.load(index_path=tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json")

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar

    # test loading the index via init
    new_document_store = FAISSDocumentStore(faiss_index_path=tmp_path / "haystack_test_faiss", faiss_config_path=tmp_path / "custom_path.json")

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar


@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason="Test with tmp_path not working on windows runner")
def test_faiss_index_mutual_exclusive_args(tmp_path):
    with pytest.raises(ValueError):
        FAISSDocumentStore(
            sql_url=f"sqlite:////{tmp_path/'haystack_test.db'}",
            faiss_index_path=f"{tmp_path/'haystack_test'}"
        )

    with pytest.raises(ValueError):
        FAISSDocumentStore(
            f"sqlite:////{tmp_path/'haystack_test.db'}",
            faiss_index_path=f"{tmp_path/'haystack_test'}"
        )


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
@pytest.mark.parametrize("batch_size", [2])
def test_faiss_write_docs(document_store, index_buffer_size, batch_size):
    document_store.index_buffer_size = index_buffer_size

    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i: i + batch_size])

    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        # we currently don't get the embeddings back when we call document_store.get_all_documents()
        original_doc = [d for d in DOCUMENTS if d["content"] == doc.content][0]
        stored_emb = document_store.faiss_indexes[document_store.index].reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(original_doc["embedding"], stored_emb, rtol=0.01)
        

@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
@pytest.mark.parametrize("batch_size", [4, 6])
def test_update_docs(document_store, retriever, batch_size):
    # initial write
    document_store.write_documents(DOCUMENTS)

    document_store.update_embeddings(retriever=retriever, batch_size=batch_size)
    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if correct vectors are associated with docs
    for doc in documents_indexed:
        original_doc = [d for d in DOCUMENTS if d["content"] == doc.content][0]
        updated_embedding = retriever.embed_documents([Document.from_dict(original_doc)])
        stored_doc = document_store.get_all_documents(filters={"name": [doc.meta["name"]]})[0]
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(updated_embedding, stored_doc.embedding, rtol=0.01)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["milvus", "faiss"], indirect=True)
def test_update_existing_docs(document_store, retriever):
    document_store.duplicate_documents = "overwrite"
    old_document = Document(content="text_1")
    # initial write
    document_store.write_documents([old_document])
    document_store.update_embeddings(retriever=retriever)
    old_documents_indexed = document_store.get_all_documents()
    assert len(old_documents_indexed) == 1

    # Update document data
    new_document = Document(content="text_2")
    new_document.id = old_document.id
    document_store.write_documents([new_document])
    document_store.update_embeddings(retriever=retriever)
    new_documents_indexed = document_store.get_all_documents()
    assert len(new_documents_indexed) == 1

    assert old_documents_indexed[0].id == new_documents_indexed[0].id
    assert old_documents_indexed[0].content == "text_1"
    assert new_documents_indexed[0].content == "text_2"
    assert not np.allclose(old_documents_indexed[0].embedding, new_documents_indexed[0].embedding, rtol=0.01)


@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_update_with_empty_store(document_store, retriever):
    # Call update with empty doc store
    document_store.update_embeddings(retriever=retriever)

    # initial write
    document_store.write_documents(DOCUMENTS)

    documents_indexed = document_store.get_all_documents()

    assert len(documents_indexed) == len(DOCUMENTS)


@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason="Test with tmp_path not working on windows runner")
@pytest.mark.parametrize("index_factory", ["Flat", "HNSW", "IVF1,Flat"])
def test_faiss_retrieving(index_factory, tmp_path):
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'test_faiss_retrieving.db'}", faiss_index_factory_str=index_factory
    )

    document_store.delete_all_documents(index="document")
    if "ivf" in index_factory.lower():
        document_store.train_index(DOCUMENTS)
    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="deepset/sentence_bert",
        use_gpu=False
    )
    result = retriever.retrieve(query="How to test this?")

    assert len(result) == len(DOCUMENTS)
    assert type(result[0]) == Document

    # Cleanup
    document_store.faiss_indexes[document_store.index].reset()


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_finding(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    pipe = DocumentSearchPipeline(retriever=retriever)

    prediction = pipe.run(query="How to test this?", params={"Retriever": {"top_k": 1}})

    assert len(prediction.get('documents', [])) == 1


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_delete_docs_with_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    document_store.delete_documents(filters={"name": ["name_1", "name_2", "name_3", "name_4"]})

    documents = document_store.get_all_documents()
    assert len(documents) == 2
    assert document_store.get_embedding_count() == 2
    assert {doc.meta["name"] for doc in documents} == {"name_5", "name_6"}


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_delete_docs_with_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    document_store.delete_documents(filters={"year": ["2020"]})

    documents = document_store.get_all_documents()
    assert len(documents) == 3
    assert document_store.get_embedding_count() == 3
    assert all("2021" == doc.meta["year"] for doc in documents)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_delete_docs_with_many_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    document_store.delete_documents(filters={"month": ["01"], "year": ["2020"]})

    documents = document_store.get_all_documents()
    assert len(documents) == 5
    assert document_store.get_embedding_count() == 5
    assert "name_1" not in {doc.meta["name"] for doc in documents}


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_delete_docs_by_id(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6
    doc_ids = [doc.id for doc in document_store.get_all_documents()]
    ids_to_delete = doc_ids[0:3]

    document_store.delete_documents(ids=ids_to_delete)

    documents = document_store.get_all_documents()
    assert len(documents) == len(doc_ids) - len(ids_to_delete)
    assert document_store.get_embedding_count() == len(doc_ids) - len(ids_to_delete)

    remaining_ids = [doc.id for doc in documents]
    assert all(doc_id not in remaining_ids for doc_id in ids_to_delete)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_delete_docs_by_id_with_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    ids_to_delete = [doc.id for doc in document_store.get_all_documents(filters={"name": ["name_1", "name_2"]})]
    ids_not_to_delete = [doc.id for doc in document_store.get_all_documents(filters={"name": ["name_3", "name_4", "name_5", "name_6"]})]

    document_store.delete_documents(ids=ids_to_delete, filters={"name": ["name_1", "name_2", "name_3", "name_4"]})

    documents = document_store.get_all_documents()
    assert len(documents) == len(DOCUMENTS) - len(ids_to_delete)
    assert document_store.get_embedding_count() == len(DOCUMENTS) - len(ids_to_delete)

    assert all(doc.meta["name"] != "name_1" for doc in documents)
    assert all(doc.meta["name"] != "name_2" for doc in documents)

    all_ids_left = [doc.id for doc in documents]
    assert all(doc_id in all_ids_left for doc_id in ids_not_to_delete)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_get_docs_with_filters_one_value(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    documents =  document_store.get_all_documents(filters={"year": ["2020"]})

    assert len(documents) == 3
    assert all("2020" == doc.meta["year"] for doc in documents)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_get_docs_with_filters_many_values(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    documents = document_store.get_all_documents(filters={"name": ["name_5", "name_6"]})

    assert len(documents) == 2
    assert {doc.meta["name"] for doc in documents} == {"name_5", "name_6"}


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_get_docs_with_many_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    documents = document_store.get_all_documents(filters={"month": ["01"], "year": ["2020"]})

    assert len(documents) == 1
    assert "name_1" == documents[0].meta["name"]
    assert "01" == documents[0].meta["month"]
    assert "2020" == documents[0].meta["year"]



@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_pipeline(document_store, retriever):
    documents = [
        {"name": "name_1", "content": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_2", "content": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_3", "content": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
        {"name": "name_4", "content": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    ]
    document_store.write_documents(documents)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="FAISS", inputs=["Query"])
    output = pipeline.run(query="How to test this?", params={"FAISS": {"top_k": 3}})
    assert len(output["documents"]) == 3


@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason="Test with tmp_path not working on windows runner")
def test_faiss_passing_index_from_outside(tmp_path):
    d = 768
    nlist = 2
    quantizer = faiss.IndexFlatIP(d)
    index = "haystack_test_1"
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)
    faiss_index.nprobe = 2
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'haystack_test_faiss.db'}", faiss_index=faiss_index, index=index
    )

    document_store.delete_documents()
    # as it is a IVF index we need to train it before adding docs
    document_store.train_index(DOCUMENTS)

    document_store.write_documents(documents=DOCUMENTS)
    documents_indexed = document_store.get_all_documents()

    # test if vectors ids are associated with docs
    for doc in documents_indexed:
        assert 0 <= int(doc.meta["vector_id"]) <= 7

def ensure_ids_are_correct_uuids(docs:list,document_store:object)->None:
    # Weaviate currently only supports UUIDs
    if type(document_store)==WeaviateDocumentStore:
        for d in docs:
            d["id"] = str(uuid.uuid4())

def test_cosine_similarity(document_store_cosine):
    # below we will write documents to the store and then query it to see if vectors were normalized

    ensure_ids_are_correct_uuids(docs=DOCUMENTS,document_store=document_store_cosine)
    document_store_cosine.write_documents(documents=DOCUMENTS)

    # note that the same query will be used later when querying after updating the embeddings
    query = np.random.rand(768).astype(np.float32)

    query_results = document_store_cosine.query_by_embedding(query_emb=query, top_k=len(DOCUMENTS), return_embedding=True)

    # check if search with cosine similarity returns the correct number of results
    assert len(query_results) == len(DOCUMENTS)
    indexed_docs = {}
    for doc in DOCUMENTS:
        indexed_docs[doc["content"]] = doc["embedding"]

    for doc in query_results:
        result_emb = doc.embedding
        original_emb = np.array([indexed_docs[doc.content]], dtype="float32")
        document_store_cosine.normalize_embedding(original_emb[0])

        # check if the stored embedding was normalized
        assert np.allclose(original_emb[0], result_emb, rtol=0.01)
        
        # check if the score is plausible for cosine similarity
        assert 0 <= doc.score <= 1.0

    # now check if vectors are normalized when updating embeddings
    class MockRetriever():
        def embed_documents(self, docs):
            return [np.random.rand(768).astype(np.float32) for doc in docs]

    retriever = MockRetriever()
    document_store_cosine.update_embeddings(retriever=retriever)
    query_results = document_store_cosine.query_by_embedding(query_emb=query, top_k=len(DOCUMENTS), return_embedding=True)

    for doc in query_results:
        original_emb = np.array([indexed_docs[doc.content]], dtype="float32")
        document_store_cosine.normalize_embedding(original_emb[0])
        # check if the original embedding has changed after updating the embeddings
        assert not np.allclose(original_emb[0], doc.embedding, rtol=0.01)


def test_normalize_embeddings_diff_shapes(document_store_cosine_small):
    VEC_1 = np.array([.1, .2, .3], dtype="float32")
    document_store_cosine_small.normalize_embedding(VEC_1)
    assert np.linalg.norm(VEC_1) - 1 < 0.01

    VEC_1 = np.array([.1, .2, .3], dtype="float32").reshape(1, -1)
    document_store_cosine_small.normalize_embedding(VEC_1)
    assert np.linalg.norm(VEC_1) - 1 < 0.01


def test_cosine_sanity_check(document_store_cosine_small):
    VEC_1 = np.array([.1, .2, .3], dtype="float32")
    VEC_2 = np.array([.4, .5, .6], dtype="float32")

    # This is the cosine similarity of VEC_1 and VEC_2 calculated using sklearn.metrics.pairwise.cosine_similarity
    # The score is normalized to yield a value between 0 and 1.
    KNOWN_COSINE = (0.9746317 + 1) / 2

    docs = [{"name": "vec_1", "text": "vec_1", "content": "vec_1", "embedding": VEC_1}]
    ensure_ids_are_correct_uuids(docs=docs,document_store=document_store_cosine_small)
    document_store_cosine_small.write_documents(documents=docs)

    query_results = document_store_cosine_small.query_by_embedding(query_emb=VEC_2, top_k=1, return_embedding=True)

    # check if faiss returns the same cosine similarity. Manual testing with faiss yielded 0.9746318
    assert math.isclose(query_results[0].score, KNOWN_COSINE, abs_tol=0.00002)
