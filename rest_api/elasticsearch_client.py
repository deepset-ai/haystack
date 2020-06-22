from elasticsearch import Elasticsearch

from rest_api.config import DB_HOST, DB_USER, DB_PW, DB_PORT, ES_CONN_SCHEME

elasticsearch_client = Elasticsearch(
    hosts=[{"host": DB_HOST, "port": DB_PORT}], http_auth=(DB_USER, DB_PW), scheme=ES_CONN_SCHEME, ca_certs=False, verify_certs=False
)
