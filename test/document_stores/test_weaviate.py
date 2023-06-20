import json
import uuid
from unittest import mock

import numpy as np
import pytest
import weaviate

from haystack.document_stores.weaviate import WeaviateDocumentStore
from haystack.schema import Document
from haystack.testing import DocumentStoreBaseTestAbstract

embedding_dim = 768


def get_uuid():
    return str(uuid.uuid4())


class TestWeaviateDocumentStore(DocumentStoreBaseTestAbstract):
    # Constants

    index_name = "DocumentsTest"

    @pytest.fixture
    def ds(self):
        return WeaviateDocumentStore(index=self.index_name, recreate_index=True, return_embedding=True)

    @pytest.fixture(scope="class")
    def documents(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    id=get_uuid(),
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "year": "2020", "month": "01", "numbers": [2.0, 4.0]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    id=get_uuid(),
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "year": "2021", "month": "02", "numbers": [-2.0, -4.0]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    id=get_uuid(),
                    content=f"A Baz Document {i}",
                    meta={"name": f"name_{i}", "month": "03"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

        return documents

    @pytest.fixture()
    def mocked_ds(self):
        """
        This fixture provides an instance of the WeaviateDocumentStore equipped with a mocked Weaviate client.
        """
        with mock.patch("haystack.document_stores.weaviate.client") as mocked_client:
            mocked_client.Client().is_ready.return_value = True
            mocked_client.Client().schema.contains.return_value = False
            yield WeaviateDocumentStore()

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_write_labels(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels_by_id(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels_by_filter(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_get_label_count(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_write_labels_duplicate(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_write_get_all_labels(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_labels_with_long_texts(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel_no_answer(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel_filter_aggregations(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel_meta_aggregations(self):
        pass

    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        """
        Weaviate doesn't include documents if the field is missing,
        so we customize this test
        """
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$ne": "2020"}})
        assert len(result) == 3

    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        """
        Weaviate doesn't include documents if the field is missing,
        so we customize this test
        """
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$nin": ["2020", "2021", "n.a."]}})
        assert len(result) == 0

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, this doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0

    @pytest.mark.integration
    def test_query_by_embedding(self, ds, documents):
        ds.write_documents(documents)

        docs = ds.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32))
        assert len(docs) == 9

        docs = ds.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32), top_k=1)
        assert len(docs) == 1

        docs = ds.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32), filters={"name": ["name_1"]})
        assert len(docs) == 3

    @pytest.mark.integration
    def test_query(self, ds, documents):
        ds.write_documents(documents)

        query_text = "Foo"
        docs = ds.query(query_text)
        assert len(docs) == 3

        # BM25 retrieval WITH filters is not yet supported as of Weaviate v1.14.1
        # Should be from 1.18: https://github.com/semi-technologies/weaviate/issues/2393
        # docs = ds.query(query_text, filters={"name": ["name_1"]})
        # assert len(docs) == 1

        docs = ds.query(query=None, filters={"name": ["name_0"]})
        assert len(docs) == 3

        docs = ds.query(query=None, filters={"content": [query_text.lower()]})
        assert len(docs) == 3

        docs = ds.query(query=None, filters={"content": ["baz"]})
        assert len(docs) == 3

    @pytest.mark.integration
    def test_get_all_documents_unaffected_by_QUERY_MAXIMUM_RESULTS(self, ds, documents, monkeypatch):
        """
        Ensure `get_all_documents` works no matter the value of QUERY_MAXIMUM_RESULTS
        see https://github.com/deepset-ai/haystack/issues/2517
        """
        ds.write_documents(documents)
        monkeypatch.setattr(ds, "get_document_count", lambda **kwargs: 13_000)
        docs = ds.get_all_documents()
        assert len(docs) == 9

    @pytest.mark.integration
    def test_deleting_by_id_or_by_filters(self, ds, documents):
        ds.write_documents(documents)
        # This test verifies that deleting an object by its ID does not first require fetching all documents. This fixes
        # a bug, as described in https://github.com/deepset-ai/haystack/issues/2898
        ds.get_all_documents = mock.MagicMock(wraps=ds.get_all_documents)

        assert ds.get_document_count() == 9

        # Delete a document by its ID. This should bypass the get_all_documents() call
        ds.delete_documents(ids=[documents[0].id])
        ds.get_all_documents.assert_not_called()
        assert ds.get_document_count() == 8

        ds.get_all_documents.reset_mock()
        # Delete a document with filters. Prove that using the filters will go through get_all_documents()
        ds.delete_documents(filters={"name": ["name_0"]})
        ds.get_all_documents.assert_called()
        assert ds.get_document_count() == 6

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["cosine", "l2", "dot_product"])
    def test_similarity_existing_index(self, similarity):
        """Testing non-matching similarity"""
        # create the document_store
        document_store = WeaviateDocumentStore(
            similarity=similarity, index=f"test_similarity_existing_index_{similarity}", recreate_index=True
        )

        # try to connect to the same document store but using the wrong similarity
        non_matching_similarity = "l2" if similarity == "cosine" else "cosine"
        with pytest.raises(ValueError, match=r"This index already exists in Weaviate with similarity .*"):
            document_store2 = WeaviateDocumentStore(
                similarity=non_matching_similarity,
                index=f"test_similarity_existing_index_{similarity}",
                recreate_index=False,
            )

    @pytest.mark.integration
    def test_cant_write_id_in_meta(self, ds):
        with pytest.raises(ValueError, match='"meta" info contains duplicate key "id"'):
            ds.write_documents([Document(content="test", meta={"id": "test-id"})])

    @pytest.mark.integration
    def test_cant_write_top_level_fields_in_meta(self, ds):
        with pytest.raises(ValueError, match='"meta" info contains duplicate key "content"'):
            ds.write_documents([Document(content="test", meta={"content": "test-id"})])

    @pytest.mark.integration
    def test_get_embedding_count(self, ds, documents):
        """
        We expect 9 docs with embeddings because all documents in the documents fixture for this class contain
        embeddings.
        """
        ds.write_documents(documents)
        assert ds.get_embedding_count() == 9

    @pytest.mark.unit
    def test__get_auth_secret(self):
        # Test with username and password
        secret = WeaviateDocumentStore._get_auth_secret("user", "pass", scope="some_scope")
        assert isinstance(secret, weaviate.AuthClientPassword)

        # Test with api key
        secret = WeaviateDocumentStore._get_auth_secret(api_key="wcs_api_key")
        assert isinstance(secret, weaviate.AuthApiKey)

        # Test with no authentication method
        secret = WeaviateDocumentStore._get_auth_secret()
        assert secret is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "embedded_options, expected_options",
        [
            (None, weaviate.EmbeddedOptions()),
            (
                {"hostname": "http://localhost", "port": "8080"},
                weaviate.EmbeddedOptions(hostname="http://localhost", port="8080"),
            ),
        ],
    )
    def test__get_embedded_options(self, embedded_options, expected_options):
        options = WeaviateDocumentStore._get_embedded_options(embedded_options)
        assert options == expected_options

    @pytest.mark.unit
    def test__get_current_properties(self, mocked_ds):
        mocked_ds.weaviate_client.schema.get.return_value = json.loads(
            """
{
  "classes": [{
    "class": "Document",
    "properties": [
        {
        "name": "hasWritten",
        "dataType": ["Article"]
        },
        {
        "name": "hitCounter",
        "dataType": ["int"]
        }
    ]
  }]
} """
        )
        # Ensure we dropped the cross-reference property
        assert mocked_ds._get_current_properties() == ["hitCounter"]

    @pytest.mark.unit
    def test_dict_metadata(self, mocked_ds):
        """
        Tests that metadata of type dict is converted to JSON string when writing to Weaviate and converted
        back to dict when reading from Weaviate.
        """
        doc = Document(content="test", meta={"dict_field": {"key": "value"}})
        # Test writing as JSON string
        mocked_ds.write_documents([doc])
        mocked_ds.weaviate_client.batch.add_data_object.assert_called_with(
            data_object={
                "content": json.dumps(doc.content),
                "content_type": doc.content_type,
                "id_hash_keys": doc.id_hash_keys,
                "dict_field": json.dumps(doc.meta["dict_field"]),
            },
            class_name=mock.ANY,
            uuid=mock.ANY,
            vector=mock.ANY,
        )

        # Test retrieving as dict
        mocked_ds.weaviate_client.query.get().do.return_value = json.loads(
            """
            {
              "data": {
                "Get": {
                  "Document": [
                    {
                      "content": "\\"test\\"",
                      "content_type": "text",
                      "dict_field": "{\\"key\\": \\"value\\"}",
                      "id_hash_keys": ["content"]
                    }
                  ]
                }
              }
            }
            """
        )
        mocked_ds.get_document_count = mock.MagicMock(return_value=1)
        mocked_ds.weaviate_client.schema.get.return_value = json.loads(
            """
            {
              "classes": [
                {
                  "class": "Document",
                  "description": "Haystack index, it's a class in Weaviate",
                  "properties": [
                    {
                      "dataType": ["text"],
                      "description": "Document Content (e.g. the text)",
                      "name": "content",
                      "tokenization": "word"
                    },
                    {
                      "dataType": ["text"],
                      "description": "JSON dynamic property dict_field",
                      "name": "dict_field",
                      "tokenization": "whitespace"
                    }
                  ]
                }
              ]
            }
            """
        )
        retrieved_docs = mocked_ds.get_all_documents()
        assert retrieved_docs[0].meta["dict_field"] == {"key": "value"}

    @pytest.mark.unit
    def test_list_of_dict_metadata(self, mocked_ds):
        """
        Tests that metadata of type list of dict is converted to list of JSON string when writing to Weaviate and
        converted back to list of dict when reading from Weaviate.
        """
        doc = Document(content="test", meta={"list_dict_field": [{"key": "value"}, {"key": "value"}]})
        # Test writing as list of JSON strings
        mocked_ds.write_documents([doc])
        mocked_ds.weaviate_client.batch.add_data_object.assert_called_with(
            data_object={
                "content": json.dumps(doc.content),
                "content_type": doc.content_type,
                "id_hash_keys": doc.id_hash_keys,
                "list_dict_field": [json.dumps(item) for item in doc.meta["list_dict_field"]],
            },
            class_name=mock.ANY,
            uuid=mock.ANY,
            vector=mock.ANY,
        )

        # Test retrieving as list of dict
        mocked_ds.weaviate_client.query.get().do.return_value = json.loads(
            """
            {
              "data": {
                "Get": {
                  "Document": [
                    {
                      "content": "\\"test\\"",
                      "content_type": "text",
                      "list_dict_field": ["{\\"key\\": \\"value\\"}", "{\\"key\\": \\"value\\"}"],
                      "id_hash_keys": ["content"]
                    }
                  ]
                }
              }
            }
            """
        )
        mocked_ds.get_document_count = mock.MagicMock(return_value=1)
        mocked_ds.weaviate_client.schema.get.return_value = json.loads(
            """
            {
              "classes": [
                {
                  "class": "Document",
                  "description": "Haystack index, it's a class in Weaviate",
                  "properties": [
                    {
                      "dataType": ["text"],
                      "description": "Document Content (e.g. the text)",
                      "name": "content",
                      "tokenization": "word"
                    },
                    {
                      "dataType": ["text[]"],
                      "description": "JSON dynamic property dict_field",
                      "name": "list_dict_field",
                      "tokenization": "whitespace"
                    }
                  ]
                }
              ]
            }
            """
        )
        retrieved_docs = mocked_ds.get_all_documents()
        assert retrieved_docs[0].meta["list_dict_field"] == [{"key": "value"}, {"key": "value"}]

    @pytest.mark.unit
    def test_write_documents_req_for_each_batch(self, mocked_ds, documents):
        mocked_ds.batch_size = 2
        mocked_ds.write_documents(documents)
        assert mocked_ds.weaviate_client.batch.create_objects.call_count == 5
