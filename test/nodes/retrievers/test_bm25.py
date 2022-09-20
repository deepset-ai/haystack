from typing import List

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever.sparse import FilterRetriever

from test.nodes.retrievers.sparse import TestSparseRetrievers


class TestBM25Retriever(TestSparseRetrievers):
    @pytest.fixture()
    def test_retriever(self, docstore):
        pass

    @pytest.mark.elasticsearch
    def test_elasticsearch_custom_query(self):
        client = Elasticsearch()
        client.indices.delete(index="haystack_test_custom", ignore=[404])
        document_store = ElasticsearchDocumentStore(
            index="haystack_test_custom", content_field="custom_text_field", embedding_field="custom_embedding_field"
        )
        documents = [
            {"content": "test_1", "meta": {"year": "2019"}},
            {"content": "test_2", "meta": {"year": "2020"}},
            {"content": "test_3", "meta": {"year": "2021"}},
            {"content": "test_4", "meta": {"year": "2021"}},
            {"content": "test_5", "meta": {"year": "2021"}},
        ]
        document_store.write_documents(documents)

        # test custom "terms" query
        retriever = BM25Retriever(
            document_store=document_store,
            custom_query="""
                {
                    "size": 10,
                    "query": {
                        "bool": {
                            "should": [{
                                "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["content"]}}],
                                "filter": [{"terms": {"year": ${years}}}]}}}""",
        )
        results = retriever.retrieve(query="test", filters={"years": ["2020", "2021"]})
        assert len(results) == 4

        # test custom "term" query
        retriever = BM25Retriever(
            document_store=document_store,
            custom_query="""
                    {
                        "size": 10,
                        "query": {
                            "bool": {
                                "should": [{
                                    "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["content"]}}],
                                    "filter": [{"term": {"year": ${years}}}]}}}""",
        )
        results = retriever.retrieve(query="test", filters={"years": "2021"})
        assert len(results) == 3

    @pytest.mark.elasticsearch
    def test_elasticsearch_highlight(self):
        client = Elasticsearch()
        client.indices.delete(index="haystack_hl_test", ignore=[404])

        # Mapping the content and title field as "text" perform search on these both fields.
        document_store = ElasticsearchDocumentStore(
            index="haystack_hl_test",
            content_field="title",
            custom_mapping={"mappings": {"properties": {"content": {"type": "text"}, "title": {"type": "text"}}}},
        )
        documents = [
            {
                "title": "Green tea components",
                "meta": {
                    "content": "The green tea plant contains a range of healthy compounds that make it into the final drink"
                },
                "id": "1",
            },
            {
                "title": "Green tea catechin",
                "meta": {"content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG)."},
                "id": "2",
            },
            {
                "title": "Minerals in Green tea",
                "meta": {"content": "Green tea also has small amounts of minerals that can benefit your health."},
                "id": "3",
            },
            {
                "title": "Green tea Benefits",
                "meta": {
                    "content": "Green tea does more than just keep you alert, it may also help boost brain function."
                },
                "id": "4",
            },
        ]
        document_store.write_documents(documents)

        # Enabled highlighting on "title"&"content" field only using custom query
        retriever_1 = BM25Retriever(
            document_store=document_store,
            custom_query="""{
                "size": 20,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": ${query},
                                    "fields": [
                                        "content^3",
                                        "title^5"
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
                        "title": {}
                    }
                }
            }""",
        )
        results = retriever_1.retrieve(query="is green tea healthy")

        assert len(results[0].meta["highlighted"]) == 2
        assert results[0].meta["highlighted"]["title"] == ["**Green**", "**tea** components"]
        assert results[0].meta["highlighted"]["content"] == ["The **green**", "**tea** plant", "range of **healthy**"]

        # Enabled highlighting on "title" field only using custom query
        retriever_2 = BM25Retriever(
            document_store=document_store,
            custom_query="""{
                "size": 20,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": ${query},
                                    "fields": [
                                        "content^3",
                                        "title^5"
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
                        "title": {}
                    }
                }
            }""",
        )
        results = retriever_2.retrieve(query="is green tea healthy")

        assert len(results[0].meta["highlighted"]) == 1
        assert results[0].meta["highlighted"]["title"] == ["**Green**", "**tea** components"]

    def test_elasticsearch_filter_must_not_increase_results(self):
        index = "filter_must_not_increase_results"
        client = Elasticsearch()
        client.indices.delete(index=index, ignore=[404])
        documents = [
            {
                "content": "The green tea plant contains a range of healthy compounds that make it into the final drink",
                "meta": {"content_type": "text"},
                "id": "1",
            },
            {
                "content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG).",
                "meta": {"content_type": "text"},
                "id": "2",
            },
            {
                "content": "Green tea also has small amounts of minerals that can benefit your health.",
                "meta": {"content_type": "text"},
                "id": "3",
            },
            {
                "content": "Green tea does more than just keep you alert, it may also help boost brain function.",
                "meta": {"content_type": "text"},
                "id": "4",
            },
        ]
        doc_store = ElasticsearchDocumentStore(index=index)
        doc_store.write_documents(documents)
        results_wo_filter = doc_store.query(query="drink")
        assert len(results_wo_filter) == 1
        results_w_filter = doc_store.query(query="drink", filters={"content_type": "text"})
        assert len(results_w_filter) == 1
        doc_store.delete_index(index)

    def test_elasticsearch_all_terms_must_match(self):
        index = "all_terms_must_match"
        client = Elasticsearch()
        client.indices.delete(index=index, ignore=[404])
        documents = [
            {
                "content": "The green tea plant contains a range of healthy compounds that make it into the final drink",
                "meta": {"content_type": "text"},
                "id": "1",
            },
            {
                "content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG).",
                "meta": {"content_type": "text"},
                "id": "2",
            },
            {
                "content": "Green tea also has small amounts of minerals that can benefit your health.",
                "meta": {"content_type": "text"},
                "id": "3",
            },
            {
                "content": "Green tea does more than just keep you alert, it may also help boost brain function.",
                "meta": {"content_type": "text"},
                "id": "4",
            },
        ]
        doc_store = ElasticsearchDocumentStore(index=index)
        doc_store.write_documents(documents)
        results_wo_all_terms_must_match = doc_store.query(query="drink green tea")
        assert len(results_wo_all_terms_must_match) == 4
        results_w_all_terms_must_match = doc_store.query(query="drink green tea", all_terms_must_match=True)
        assert len(results_w_all_terms_must_match) == 1
        doc_store.delete_index(index)
