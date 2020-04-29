import json
import logging
from string import Template

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

from haystack.database.base import BaseDocumentStore, Document

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
        external_source_id_field="external_source_id",
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
                        external_source_id_field: {"type": "text"},
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
        self.external_source_id_field = external_source_id_field
        self.embedding_field = embedding_field
        self.excluded_meta_data = excluded_meta_data

    def get_document_by_id(self, id: str) -> Document:
        query = {"query": {"ids": {"values": [id]}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]

        document = self._convert_es_hit_to_document(result[0]) if result else None
        return document

    def get_document_ids_by_tags(self, tags: dict) -> [str]:
        term_queries = [{"terms": {key: value}} for key, value in tags.items()]
        query = {"query": {"bool": {"must": term_queries}}}
        logger.debug(f"Tag filter query: {query}")
        result = self.client.search(index=self.index, body=query, size=10000)["hits"]["hits"]
        doc_ids = []
        for hit in result:
            doc_ids.append(hit["_id"])
        return doc_ids

    def write_documents(self, documents):
        for doc in documents:
            doc["_op_type"] = "create"
            doc["_index"] = self.index

        bulk(self.client, documents)

    def get_document_count(self):
        result = self.client.count()
        count = result["count"]
        return count

    def get_all_documents(self):
        result = scan(self.client, query={"query": {"match_all": {}}}, index=self.index)
        documents = [self._convert_es_hit_to_document(hit) for hit in result]

        return documents

    def query(
        self,
        query: str,
        filters: dict = None,
        top_k: int = 10,
        custom_query: str = None,
    ) -> [Document]:

        if custom_query:  # substitute placeholder for question and filters for the custom_query template string
            template = Template(custom_query)

            substitutions = {"question": query}  # replace all "${question}" placeholder(s) with query
            # replace all filter values placeholders with a list of strings(in JSON format) for each filter
            if filters:
                for key, values in filters.items():
                    values_str = json.dumps(values)
                    substitutions[key] = values_str
            custom_query_json = template.substitute(**substitutions)
            body = json.loads(custom_query_json)
        else:
            body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]
                    }
                },
            }

            if filters:
                filter_clause = []
                for key, values in filters.items():
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause

        if self.excluded_meta_data:
            body["_source"] = {"excludes": self.excluded_meta_data}

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=self.index, body=body)["hits"]["hits"]

        documents = [self._convert_es_hit_to_document(hit) for hit in result]
        return documents

    def query_by_embedding(self, query_emb, top_k=10, candidate_doc_ids=None) -> [Document]:
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

            documents = [self._convert_es_hit_to_document(hit, score_adjustment=-1) for hit in result]
            return documents

    def _convert_es_hit_to_document(self, hit, score_adjustment=0) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {k:v for k,v in hit["_source"].items() if k not in (self.text_field, self.external_source_id_field)}
        meta_data["name"] = meta_data.pop(self.name_field)

        document = Document(
            id=hit["_id"],
            text=hit["_source"][self.text_field],
            external_source_id=hit["_source"].get(self.external_source_id_field),
            meta=meta_data,
            query_score=hit["_score"] + score_adjustment if hit["_score"] else None,
        )
        return document
