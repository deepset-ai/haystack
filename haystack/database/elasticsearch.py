from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Document as ESDoc, Text, connections
from haystack.database.base import BaseDocumentStore


def create_index(index_name):
    """
    Example to create a simple, default ES index from scratch.
    You might want to modify the schema for your own purposes
    """

    class Document(ESDoc):
                name = Text()
                text = Text()
                tags = Text()
                document_id = Text()

                class Index:
                    name = index_name

    Document.init()  # create mapping if not exists.


class ElasticsearchDocumentStore(BaseDocumentStore):
    def __init__(self, host="localhost", username="", password="", index="document", search_fields="text",
                 text_field="text", name_field="name", doc_id_field="document_id", tag_fields=None):

        self.client = Elasticsearch(hosts=[{"host": host}], http_auth=(username, password))
        self.connections = connections.create_connection(hosts=[{"host": host}], http_auth=(username, password))
        self.index = index

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]
        self.search_fields = search_fields
        self.text_field = text_field
        self.name_field = name_field
        self.tag_fields = tag_fields
        self.doc_id_field = doc_id_field

    def get_document_by_id(self, id):
        query = {"filter": {"term": {"_id": id}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        if result:
            document = {"id": result["_id"], "name": result[self.name_field], "text": result[self.text_field]}
        else:
            document = None
        return document

    def get_document_by_name(self, name):
        query = {"filter": {"term": {self.name_field: name}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        if result:
            document = {"id": result["_id"], "name": result[self.name_field], "text": result[self.text_field]}
        else:
            document = None
        return document

    def get_document_ids_by_tags(self, tags):
        term_queries = [{"terms": {key: value}} for key, value in tags.items()]
        query = {
            "query": {
                "bool": {
                    "must": term_queries
                }
            }
        }
        result = self.client.search(index=self.index, body=query, size=10000)["hits"]["hits"]
        doc_ids = []
        for hit in result:
            doc_ids.append(hit["_id"])
        return doc_ids

    def write_documents(self, documents):
        for d in documents:
            # d = {"name": doc["name"],
            #     "text": doc["text"],
            #     "document_id": doc.get("document_id", None),
            #     "tags": doc.get("tags", None)}

            self.client.index(index=self.index, body=d)

    def get_document_count(self):
        s = Search(using=self.client, index=self.index)
        return s.count()

    def get_all_documents(self):
        search = Search(using=self.client, index=self.index).scan()
        documents = []
        for hit in search:
            documents.append(
                {
                    "id": hit.meta["id"],
                    "name": hit[self.name_field],
                    "text": hit[self.text_field],
                }
            )
        return documents

    def query(self, query, top_k=10, candidate_doc_ids=None):
        # TODO:
        # for now: we keep the current structure of candidate_doc_ids for compatibility with SQL documentstores
        # midterm: get rid of it and do filtering with tags directly in this query

        body = {"size": top_k,
                'query': {
                    "bool": {
                        "must": [
                            {'multi_match': {
                                'query': query,
                                'type': 'most_fields',
                                'fields': self.search_fields}}
                        ]
                    }}}
        if candidate_doc_ids:
            body["query"]["bool"]["filter"] = [{"terms": {"_id": candidate_doc_ids}}]
        result = self.client.search(index=self.index, body=body)["hits"]["hits"]
        paragraphs = []
        meta_data = []
        for hit in result:
            paragraphs.append(hit["_source"][self.text_field])
            meta_data.append({"paragraph_id": hit["_id"],
                              "document_id": hit["_source"][self.doc_id_field]})
        return paragraphs, meta_data


