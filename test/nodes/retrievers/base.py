from typing import List

import logging
from math import isclose
from time import sleep
import subprocess
from abc import ABC, abstractmethod

import numpy as np
import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import InMemoryDocumentStore


def _init_container(container_name: str, setup_command: str, boot_wait: int = 0):
    """
    Utility to make the test suite start and stop containers at need.
    """
    # If the container is already running this is probably the CI. Don't touch it.
    process = subprocess.run([f"docker ps -f name={container_name} -q"], shell=True, check=True)
    if process.stdout:
        logging.info("The test container (%s) is already running. PyTest won't manage it.", container_name)
        yield

    else:
        # Otherwise, start the container and remove it once done.
        logging.warning(
            "Setting up %s.%s",
            container_name,
            f" This may add up to {boot_wait} seconds to your first test." if boot_wait else "",
        )
        process = subprocess.run(
            [f"docker start {container_name} || {setup_command} --name {container_name}"], shell=True, check=True
        )
        sleep(boot_wait)
        yield
        subprocess.run([f"docker stop {container_name}"], shell=True, check=True)
        subprocess.run([f"docker rm {container_name}"], shell=True, check=True)


class ABC_TestBaseRetriever(ABC):

    QUESTION = "Who lives in Berlin?"
    ANSWER = "My name is Carla and I live in Berlin"

    @pytest.fixture(scope="session")
    def init_elasticsearch(self):
        yield _init_container(
            container_name="haystack_tests_elasticsearch",
            setup_command='docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2',
            boot_wait=15,
        )

    @pytest.fixture
    def docstore(self, docs: List[Document]):
        docstore = InMemoryDocumentStore()
        docstore.write_documents(docs)
        return docstore

    @abstractmethod
    @pytest.fixture
    def retriever(self, docstore):
        raise NotImplementedError("Abstract method, use a subclass")

    #
    # End2End tests
    #

    def test_retrieval(self, retriever: BaseRetriever):
        res = retriever.retrieve(query="Who lives in Berlin?")
        assert len(res) > 0
        assert res[0].content == "My name is Carla and I live in Berlin"

    def test_retrieval_with_single_filter(self, retriever: BaseRetriever):
        result = retriever.retrieve(query="Who lives in Berlin?", filters={"name": ["filename1"]}, top_k=5)
        assert len(result) == 1
        assert result[0].content == "My name is Carla and I live in Berlin"
        assert result[0].meta["name"] == "filename1"

    def test_retrieval_with_many_filter(self, retriever: BaseRetriever):
        result = retriever.retrieve(
            query="Who lives in Berlin?",
            filters={"name": ["filename5", "filename1"], "meta_field": ["test1", "test2"]},
            top_k=5,
        )
        assert len(result) == 1
        assert result[0].meta["name"] == "filename1"

    def test_retrieval_with_many_filter_no_match(self, retriever: BaseRetriever):
        result = retriever.retrieve(
            query="Who lives in Berlin?", filters={"name": ["filename1"], "meta_field": ["test2", "test3"]}, top_k=5
        )
        assert len(result) == 0

    def test_batch_retrieval(self, retriever: BaseRetriever):
        res = retriever.retrieve_batch(queries=["Who lives in Berlin?", "Who lives in Madrid?"], top_k=5)

        assert len(res) == 2  # 2 queries
        assert len(res[0]) == 5  # top_k = 5
        assert res[0][0].content == "My name is Carla and I live in Berlin"
        assert res[1][0].content == "My name is Camila and I live in Madrid"
