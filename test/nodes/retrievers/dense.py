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

from test.nodes.retrievers.base import ABC_TestBaseRetriever


@pytest.mark.integration
class ABC_TestDenseRetrievers(ABC_TestBaseRetriever):
    @pytest.fixture
    def docstore(self, docs_with_ids: List[Document]):
        docstore = InMemoryDocumentStore(return_embedding=True)
        docstore.write_documents(docs_with_ids)
        return docstore
