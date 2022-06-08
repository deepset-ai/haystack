import sys

import pytest

from haystack.document_stores import OpenSearchDocumentStore

pytestmark = pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Opensearch not running on Windows CI")


@pytest.mark.elasticsearch
def test_init_opensearch_client():
    OpenSearchDocumentStore(index="test_index", port=9201)
