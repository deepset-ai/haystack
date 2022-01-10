import uuid
import faiss
import math
import numpy as np
import pytest
import sys
import subprocess
import logging
from time import sleep
from sqlalchemy import create_engine, text
import psycopg

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


# @pytest.fixture
# def sql_url():

#     # status = subprocess.run(["docker run --name postgres_test -d -e POSTGRES_HOST_AUTH_METHOD=trust -p 5432:5432 postgres"], shell=True)
#     # if status.returncode:
#     #     logging.warning("Tried to start PostgreSQL through Docker but this failed. It is likely that there is already an existing instance running.")
#     # else:
#     #     sleep(5)

#     engine = create_engine(
#         'postgresql://postgres:postgres@127.0.0.1/postgres',
#         isolation_level='AUTOCOMMIT')

#     # with engine.connect() as connection:
#     connection = engine.connect()

#     try:
#         connection.execute('DROP SCHEMA public CASCADE;')
#     except Exception as e:
#         logging.error(e)
#     try:
#         connection.execute('CREATE SCHEMA public;')  
#         connection.execute('SET SESSION idle_in_transaction_session_timeout = "1s";')   

#         yield "postgresql://postgres:postgres@127.0.0.1/postgres"

#     finally:
#         connection.execute('DROP SCHEMA public CASCADE;')
#         connection.close()

#     logging.error(" -----------------------> Done")

    # sleep(1)

    # status = subprocess.run(["docker stop postgres_test"], shell=True)
    # if status.returncode:
    #     logging.warning("Tried to start PostgreSQL through Docker but this failed. It is likely that there is already an existing instance running.")

    # status = subprocess.run(["docker rm postgres_test"], shell=True)
    # if status.returncode:
    #     logging.warning("Tried to start PostgreSQL through Docker but this failed. It is likely that there is already an existing instance running.")



# @pytest.fixture
# def sql_url(tmp_path):
#     return f"sqlite:////{tmp_path/'haystack_test.db'}"



@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason="Test with tmp_path not working on windows runner")
def test_faiss_index_save_and_load(tmp_path): #, sql_url):

    engine = create_engine(
        'postgresql://postgres:postgres@127.0.0.1/postgres',
        isolation_level='AUTOCOMMIT',
        future=True)

    with engine.connect() as connection:
        #connection = engine.connect()

        try:
            connection.execute(text('DROP SCHEMA public CASCADE'))
            connection.commit()
        except psycopg.errors.ProgrammingError as pe:
            logging.error(pe)
        except Exception as e:
            logging.error(e)
        
        connection.execute(text('CREATE SCHEMA public'))
        connection.commit()
        #connection.execute('SET SESSION idle_in_transaction_session_timeout = "1s";')  

        sql_url = "postgresql://postgres:postgres@127.0.0.1/postgres"




        document_store = FAISSDocumentStore(
            sql_url=sql_url,
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

        # test saving and loading the loaded faiss index
        new_document_store.save(tmp_path / "haystack_test_faiss")
        reloaded_document_store = FAISSDocumentStore.load(tmp_path / "haystack_test_faiss")

        # check faiss index is restored
        assert reloaded_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
        # check if documents are restored
        assert len(reloaded_document_store.get_all_documents()) == len(DOCUMENTS)
        # Check if the init parameters are kept
        assert not reloaded_document_store.progress_bar

        # test loading the index via init
        new_document_store = FAISSDocumentStore(faiss_index_path=tmp_path / "haystack_test_faiss")

        # check faiss index is restored
        assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
        # check if documents are restored
        assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
        # Check if the init parameters are kept
        assert not new_document_store.progress_bar



        connection.execute(text('DROP SCHEMA public CASCADE'))
        connection.commit()
        # connection.close()

    logging.error(" -----------------------> Done")
