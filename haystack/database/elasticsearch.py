from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Document as ESDoc, Text, connections
from haystack.database.base import BaseDocumentStore


class Document(ESDoc):
    name = Text()
    text = Text()
    tags = Text()

    class Index:
        name = "document"


class ElasticsearchDocumentStore(BaseDocumentStore):
    def __init__(self, host="localhost", username="", password="", index="document"):
        self.client = Elasticsearch(hosts=[{"host": host}], http_auth=(username, password))
        self.connections = connections.create_connection(hosts=[{"host": host}], http_auth=(username, password))
        Document.init()  # create mapping if not exists.
        self.index = index

    def get_document_by_id(self, id):
        query = {"filter": {"term": {"_id": id}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        if result:
            document = {"id": result["_id"], "name": result["name"], "text": result["text"]}
        else:
            document = None
        return document

    def get_document_ids_by_tags(self, tags):
        query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "terms": {
                                "tags": tags
                            }
                        }
                    ]
                }
            }
        }
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        documents = []
        for hit in result:
            documents.append({"id": hit["_id"], "name": hit["name"], "text": hit["text"]})
        return documents

    def write_documents(self, documents):
        for doc in documents:
            d = Document(
                name=doc["name"],
                text=doc["text"],
                document_id=doc.get("document_id", None),
                tags=doc.get("tags", None),
            )
            d.save()

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
                    "name": hit["name"],
                    "text": hit["text"],
                }
            )
        return documents

    def query(self, query, top_k=10):
        search = Search(using=self.client, index=self.index).query("match", text=query)[:top_k].execute()
        paragraphs = []
        meta_data = []
        for hit in search:
            paragraphs.append(hit["text"])
            meta_data.append({"paragraph_id": hit.meta["id"], "document_id": hit["document_id"]})
        return paragraphs, meta_data
