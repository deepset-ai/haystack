docker run --name elasticsearch -d -p 9201:9200 -e "discovery.type=single-node" elasticsearch:7.17.6
docker run --name opensearch -d -p 9200:9200 -e "discovery.type=single-node" opensearchproject/opensearch:1.3.5
docker run --name waeviate -d -p 8080:8080 -e "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true" -e "PERSISTENCE_DATA_PATH=/var/lib/weaviate" semitechnologies/weaviate:1.20.3
