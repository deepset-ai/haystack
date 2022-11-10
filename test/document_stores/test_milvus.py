from unittest import mock

import pytest

from packaging import version

import haystack
from haystack.document_stores.milvus import Milvus2DocumentStore

fail_in_v1_13 = pytest.mark.xfail(
    version.parse(haystack.__version__) > version.parse("1.12"),
    reason="feature should be removed in v1.13, as it was deprecated in v1.11",
)


@fail_in_v1_13
@mock.patch("haystack.document_stores.milvus.connections")
@mock.patch("haystack.document_stores.milvus.MilvusDocumentStore._create_collection_and_index")
def test_Milvus2DocumentStore_deprecated(_create_collection_and_index, connections):
    print(version.parse(haystack.__version__), version.parse("1.10"))
    with pytest.warns(FutureWarning):
        Milvus2DocumentStore()
