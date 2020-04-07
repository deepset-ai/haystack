from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from haystack.database.base import BaseDocumentStore

import logging
logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        host="localhost",
        username="",
        password="",
        index="document",
        search_fields="text",
        text_field="text",
        name_field="name",
        doc_id_field="document_id",
        tag_fields=None,
        embedding_field=None,
        embedding_dim=None,
        custom_mapping=None,
        excluded_meta_data=None,
        scheme="http",
        ca_certs=False,
        verify_certs=True,
        create_index=True
    ):
        self.client = Elasticsearch(hosts=[{"host": host}], http_auth=(username, password),
                                    scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs)

        # if no custom_mapping is supplied, use the default mapping
        if not custom_mapping:
            custom_mapping = {
                "mappings": {
                    "properties": {
                        name_field: {"type": "text"},
                        text_field: {"type": "text"},
                        doc_id_field: {"type": "text"},
                    }
                }
            }
            if embedding_field:
                custom_mapping["mappings"]["properties"][embedding_field] = {"type": "dense_vector",
                                                                             "dims": embedding_dim}
        # create an index if not exists
        if create_index:
            self.client.indices.create(index=index, ignore=400, body=custom_mapping)
        self.index = index

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        #TODO we should implement a more flexible interal mapping here that simplifies the usage of additional,
        # custom fields (e.g. meta data you want to return)
        self.search_fields = search_fields
        self.text_field = text_field
        self.name_field = name_field
        self.tag_fields = tag_fields
        self.doc_id_field = doc_id_field
        self.embedding_field = embedding_field
        self.excluded_meta_data = excluded_meta_data

    def get_document_by_id(self, id):
        query = {"filter": {"term": {"_id": id}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        if result:
            document = {
                "id": result[self.doc_id_field],
                "name": result[self.name_field],
                "text": result[self.text_field],
            }
        else:
            document = None
        return document

    def get_document_by_name(self, name):
        query = {"filter": {"term": {self.name_field: name}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        if result:
            document = {
                "id": result[self.doc_id_field],
                "name": result[self.name_field],
                "text": result[self.text_field],
            }
        else:
            document = None
        return document

    def get_document_ids_by_tags(self, tags):
        term_queries = [{"terms": {key: value}} for key, value in tags.items()]
        query = {"query": {"bool": {"must": term_queries}}}
        logger.debug(f"Tag filter query: {query}")
        result = self.client.search(index=self.index, body=query, size=10000)["hits"]["hits"]
        doc_ids = []
        for hit in result:
            doc_ids.append(hit["_id"])
        return doc_ids

    def write_documents(self, documents):
        for d in documents:
            try:
                self.client.index(index=self.index, body=d)
            except Exception as e:
                logger.error(f"Failed to index doc ({e}): {d}")

    def get_document_count(self):
        result = self.client.count()
        count = result["count"]
        return count

    def get_all_documents(self):
        result = scan(self.client, query={"query": {"match_all": {}}}, index=self.index)
        documents = []
        for hit in result:
            documents.append(
                {
                    "id": hit["_source"][self.doc_id_field],
                    "name": hit["_source"][self.name_field],
                    "text": hit["_source"][self.text_field],
                }
            )
        return documents

    def query(self, query, top_k=10, candidate_doc_ids=None, direct_filters=None, custom_query=None):
        # TODO:
        # for now: we keep the current structure of candidate_doc_ids for compatibility with SQL documentstores
        # midterm: get rid of it and do filtering with tags directly in this query

        # if a custom search query is provided then use it
        if custom_query:
            body = {
                "size": top_k,
                "query": {
                    "bool": custom_query
                }
            }
        # else use standard search query for provided search fields
        else:
            body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]
                    }
                },
            }

        # use other filters directly with query, if provided
        if direct_filters:
            # filter types are must, should, etc.
            for filter_type, filter_dict in direct_filters.items():
                body["query"]["bool"][filter_type] = filter_dict

        if candidate_doc_ids:
            body["query"]["bool"]["filter"] = [{"terms": {"_id": candidate_doc_ids}}]

        if self.excluded_meta_data:
            body["_source"] = {"excludes": self.excluded_meta_data}

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=self.index, body=body)["hits"]["hits"]
        paragraphs = []
        meta_data = []
        for hit in result:
            # add the text paragraph
            paragraphs.append(hit["_source"].pop(self.text_field))

            # add & rename some standard fields
            cur_meta = {
                "paragraph_id": hit["_id"],
                "document_id": hit["_source"].pop(self.doc_id_field),
                "document_name": hit["_source"].pop(self.name_field),
                "score": hit["_score"]
            }
            # add all the rest with original name
            cur_meta.update(hit["_source"])
            meta_data.append(cur_meta)
        return paragraphs, meta_data

    def query_by_embedding(self, query_emb, top_k=10, candidate_doc_ids=None):
        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
        else:
            # +1 in cosine similarity to avoid negative numbers
            body= {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector,doc['question_emb']) + 1.0",
                            "params": {
                                "query_vector": query_emb
                            }
                        }
                    }
                }
            }

            if candidate_doc_ids:
                body["query"]["script_score"]["query"] = {
                    "bool": {
                        "should": [{"match_all": {}}],
                        "filter": [{"terms": {"_id": candidate_doc_ids}}]
                }}

            if self.excluded_meta_data:
                body["_source"] = {"excludes": self.excluded_meta_data}

            logger.debug(f"Retriever query: {body}")
            result = self.client.search(index=self.index, body=body)["hits"]["hits"]
            paragraphs = []
            meta_data = []
            for hit in result:
                # add the text paragraph
                paragraphs.append(hit["_source"].pop(self.text_field))

                # add & rename some standard fields
                cur_meta = {
                        "paragraph_id": hit["_id"],
                        "document_id": hit["_source"].pop(self.doc_id_field),
                        "document_name": hit["_source"].pop(self.name_field),
                        "score": hit["_score"] -1 # -1 because we added +1 in the ES query
                    }
                # add all the rest with original name
                cur_meta.update(hit["_source"])
                meta_data.append(cur_meta)

            return paragraphs, meta_data
