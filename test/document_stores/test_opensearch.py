import pytest

from haystack.document_stores import OpenSearchDocumentStore


@pytest.mark.elasticsearch
def test_init_opensearch_client():
    OpenSearchDocumentStore(index="test_index", port=9201)
