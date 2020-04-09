from elasticsearch import Elasticsearch

from haystack.api.config import DB_HOST, DB_USER, DB_PW

elasticsearch_client = Elasticsearch(
    hosts=[{"host": DB_HOST}], http_auth=(DB_USER, DB_PW), scheme="http", ca_certs=False, verify_certs=False
)
