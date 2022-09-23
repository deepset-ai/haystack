from typing import List, Any, Union, Optional

import pytest

from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore, BaseDocumentStore
from haystack.nodes import BM25Retriever

from test.nodes.retrievers.sparse import ABC_TestSparseRetrievers


@pytest.mark.elasticsearch
class TestBM25Retriever(ABC_TestSparseRetrievers):
    @pytest.fixture(autouse=True, scope="session")
    def init_docstore(self, init_elasticsearch):
        pass

    @pytest.fixture
    def docstore(self, docs: List[Document]):
        docstore = ElasticsearchDocumentStore(
            index="haystack_test", return_embedding=True, embedding_dim=768, similarity="cosine", recreate_index=True
        )
        docstore.write_documents(docs)
        yield docstore
        docstore.delete_documents()

    @pytest.fixture
    def retriever(self, docstore: BaseDocumentStore):
        yield BM25Retriever(document_store=docstore)

    def test_elasticsearch_custom_query_with_terms(self, retriever: BM25Retriever):
        retriever.custom_query = """
            {
                "size": 10,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": ${query}, 
                                    "type": "most_fields", 
                                    "fields": ["content"]
                                }
                            }
                        ],
                        "filter": [
                            {
                                "terms": {
                                    "meta_field": ${meta_fields}
                                }
                            }
                        ]
                    }
                }
            }
        """
        results = retriever.retrieve(query="live", filters={"meta_fields": ["test1", "test5"]})
        assert len(results) == 2

    def test_elasticsearch_custom_query_with_term(self, retriever: BM25Retriever):
        # test custom "term" query
        retriever.custom_query = """
            {
                "size": 10,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": ${query}, 
                                    "type": "most_fields", 
                                    "fields": ["content"]
                                }
                            }
                        ],
                        "filter": [
                            {
                                "term": {
                                    "meta_field": ${meta_fields}
                                }
                            }
                        ]
                    }
                }
            }
        """
        results = retriever.retrieve(query="live", filters={"meta_fields": "test5"})
        assert len(results) == 1

    def test_highlight_content_and_name(self, retriever: BM25Retriever, docs: List[Document]):

        # Modify the index to add "name" to the search fields (requires re-indexing)
        retriever.document_store.search_fields = ["content", "name"]
        retriever.document_store._delete_index(retriever.document_store.index)
        retriever.document_store._create_document_index(retriever.document_store.index)
        retriever.document_store.write_documents(docs)

        # Enabled highlighting on "title"&"content" field only using custom query
        retriever.custom_query = """
            {
                "size": 20,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": ${query},
                                    "fields": [
                                        "content^3",
                                        "name^5"
                                    ]
                                }
                            }
                        ]
                    }
                },
                "highlight": {
                    "pre_tags": [
                        "**"
                    ],
                    "post_tags": [
                        "**"
                    ],
                    "number_of_fragments": 3,
                    "fragment_size": 5,
                    "fields": {
                        "content": {},
                        "name": {}
                    }
                }
            }
        """
        results = retriever.retrieve(query="Who's from Paris, according to filename3?")

        assert results[0].meta["highlighted"] == {"name": ["**filename3**"], "content": ["live in **Paris**"]}

    def test_highlight_name_only(self, retriever: BM25Retriever, docs: List[Document]):

        # Modify the index to have only "name" in the search fields (requires re-indexing)
        retriever.document_store.search_fields = ["name"]
        retriever.document_store._delete_index(retriever.document_store.index)
        retriever.document_store._create_document_index(retriever.document_store.index)
        retriever.document_store.write_documents(docs)

        # Mapping the content and title field as "text" perform search on these both fields.
        # FIXME Use search_fields rather
        retriever.document_store.custom_mapping = {"mappings": {"properties": {"name": {"type": "text"}}}}

        # Enabled highlighting on "title" field only using custom query
        retriever.custom_query = """{
                "size": 20,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": ${query},
                                    "fields": [
                                        "content^3",
                                        "name^5"
                                    ]
                                }
                            }
                        ]
                    }
                },
                "highlight": {
                    "pre_tags": [
                        "**"
                    ],
                    "post_tags": [
                        "**"
                    ],
                    "number_of_fragments": 3,
                    "fragment_size": 5,
                    "fields": {
                        "name": {}
                    }
                }
            }
        """
        results = retriever.retrieve(query="Who lives in Paris, according to filename3?")

        assert results[0].meta["highlighted"] == {"name": ["**filename3**"]}

    def test_filter_must_not_increase_results(self, retriever: BM25Retriever):
        # https://github.com/deepset-ai/haystack/pull/2359
        results_wo_filter = retriever.retrieve(query="Paris")
        results_w_filter = retriever.retrieve(query="Paris", filters={"odd_field": [0, 1]})
        assert results_w_filter == results_wo_filter

    def test_all_terms_must_match(self, retriever: BM25Retriever, docs: List[Document]):
        results = retriever.retrieve(query="live in Paris")
        assert len(results) == len(docs)

        retriever.all_terms_must_match = True
        results = retriever.retrieve(query="live in Paris")
        assert len(results) == 1
