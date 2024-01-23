from unittest.mock import MagicMock, patch

import pymongo
import pytest
from numpy import float32, random

from haystack.document_stores import mongodb_atlas
from haystack.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class TestMongoDBDocumentStore:
    @pytest.fixture
    def mocked_ds(self):
        class DSMock(MongoDBAtlasDocumentStore):
            # We mock a subclass to avoid messing up the actual class object
            pass

        mongodb_atlas._validate_mongo_connection_string = MagicMock()
        mongodb_atlas._validate_database_name = MagicMock()
        mongodb_atlas._validate_collection_name = MagicMock()
        mongodb_atlas._get_collection = MagicMock()
        pymongo.MongoClient = MagicMock()

        mocked_ds = DSMock(
            mongo_connection_string="mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}",
            database_name="test_db",
            collection_name="test_collection",
            embedding_dim=1536,
        )

        return mocked_ds

    @pytest.mark.unit
    def test_error_is_raised_if_vector_index_name_is_not_set_for_vector_search(self, mocked_ds):
        with pytest.raises(ValueError):
            mocked_ds.vector_search_index = None
            mocked_ds.query_by_embedding(query_emb=random.rand(768))

    @pytest.mark.unit
    def test_vector_index_name_is_set_for_vector_search(self, mocked_ds):
        mocked_ds.vector_search_index = "vector_search_index"
        with patch.object(mocked_ds, "_get_collection", return_value=MagicMock()) as mock_get_collection:
            query_emb = random.rand(768)
            mocked_ds.query_by_embedding(query_emb=query_emb)
            expected_emb_in_call = query_emb.astype(float32)
            mocked_ds.normalize_embedding(expected_emb_in_call)
            # check that the correct arguments are passed to collection.aggregate()
            mock_get_collection().aggregate.assert_called()
            mock_get_collection().aggregate.assert_called_once_with(
                [
                    {
                        "$vectorSearch": {
                            "index": "vector_search_index",
                            "queryVector": expected_emb_in_call.tolist(),
                            "path": "embedding",
                            "numCandidates": 100,
                            "limit": 10,
                        }
                    },
                    {"$match": {}},
                    {"$project": {"embedding": False}},
                    {"$set": {"score": {"$meta": "vectorSearchScore"}}},
                ]
            )
