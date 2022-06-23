import time
import logging
import subprocess
from pathlib import Path

import requests


logger = logging.getLogger(__name__)
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"
OPENSEARCH_CONTAINER_NAME = "opensearch"
MILVUS1_CONTAINER_NAME = "milvus1"
WEAVIATE_CONTAINER_NAME = "weaviate"


def launch_es(sleep=15, delete_existing=False):
    # Start an Elasticsearch server via Docker

    logger.debug("Starting Elasticsearch ...")
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {ELASTICSEARCH_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)
    status = subprocess.run(
        [
            f'docker start {ELASTICSEARCH_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d -p 9200:9200 -e "discovery.type=single-node" --name {ELASTICSEARCH_CONTAINER_NAME} elasticsearch:7.9.2'
        ],
        shell=True,
    )
    if status.returncode:
        logger.warning(
            "Tried to start Elasticsearch through Docker but this failed. "
            "It is likely that there is already an existing Elasticsearch instance running. "
        )
    else:
        time.sleep(sleep)


def launch_opensearch(sleep=15, delete_existing=False):
    # Start an OpenSearch server via docker

    logger.debug("Starting OpenSearch...")
    # This line is needed since it is not possible to start a new docker container with the name opensearch if there is a stopped image with the same now
    # docker rm only succeeds if the container is stopped, not if it is running
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {OPENSEARCH_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)
    status = subprocess.run(
        [
            f'docker start {OPENSEARCH_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d -p 9201:9200 -p 9600:9600 -e "discovery.type=single-node" --name {OPENSEARCH_CONTAINER_NAME} opensearchproject/opensearch:1.2.4'
        ],
        shell=True,
    )
    if status.returncode:
        logger.warning(
            "Tried to start OpenSearch through Docker but this failed. "
            "It is likely that there is already an existing OpenSearch instance running. "
        )
    else:
        time.sleep(sleep)


def launch_weaviate(sleep=15):
    # Start a Weaviate server via Docker

    logger.debug("Starting Weaviate ...")
    status = subprocess.run(
        [
            f"docker start {WEAVIATE_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d -p 8080:8080 --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED='true' --env PERSISTENCE_DATA_PATH='/var/lib/weaviate' --name {WEAVIATE_CONTAINER_NAME} semitechnologies/weaviate:1.11.0"
        ],
        shell=True,
    )
    if status.returncode:
        logger.warning(
            "Tried to start Weaviate through Docker but this failed. "
            "It is likely that there is already an existing Weaviate instance running. "
        )
    else:
        time.sleep(sleep)


def stop_container(container_name, delete_container=False):
    logger.debug(f"Stopping {container_name}...")
    status = subprocess.run([f"docker stop {container_name}"], shell=True)
    if status.returncode:
        logger.warning(
            f"Tried to stop {container_name} but this failed. "
            f"It is likely that there was no Docker container with the name {container_name}"
        )
    if delete_container:
        status = subprocess.run([f"docker rm {container_name}"], shell=True)


def stop_opensearch(delete_container=False):
    stop_container(OPENSEARCH_CONTAINER_NAME, delete_container)


def stop_elasticsearch(delete_container=False):
    stop_container(ELASTICSEARCH_CONTAINER_NAME, delete_container)


def stop_milvus(delete_container=False):
    stop_container(MILVUS1_CONTAINER_NAME, delete_container)


def stop_weaviate(delete_container=False):
    stop_container(WEAVIATE_CONTAINER_NAME, delete_container)


def stop_service(document_store, delete_container=False):
    ds_class = str(type(document_store))
    if "OpenSearchDocumentStore" in ds_class:
        stop_opensearch(delete_container)
    elif "ElasticsearchDocumentStore" in ds_class:
        stop_elasticsearch(delete_container)
    elif "MilvusDocumentStore" in ds_class:
        stop_milvus(delete_container)
    elif "WeaviateDocumentStore" in ds_class:
        stop_weaviate(delete_container)
    else:
        logger.warning(f"No support yet for auto stopping the service behind a {type(document_store)}")


def launch_milvus(sleep=15, delete_existing=False):
    # Start a Milvus server via docker

    logger.debug("Starting Milvus ...")

    milvus_dir = Path.home() / "milvus"
    milvus_dir.mkdir(exist_ok=True)

    request = requests.get(
        "https://github.com/milvus-io/milvus/releases/download/v2.0.0/milvus-standalone-docker-compose.yml"
    )
    with open(milvus_dir / "docker-compose.yml", "wb") as f:
        f.write(request.content)

    status = subprocess.run(["cd /home/$USER/milvus/ && docker-compose up -d"], shell=True)

    if status.returncode:
        logger.warning(
            "Tried to start Milvus through Docker but this failed. "
            "It is likely that there is already an existing Milvus instance running. "
        )
    else:
        time.sleep(sleep)


def launch_milvus1(sleep=15):
    # Start a Milvus (version <2.0.0) server via docker

    logger.debug("Starting Milvus ...")
    logger.warning(
        "Automatic Milvus config creation not yet implemented. "
        "If you are starting Milvus using launch_milvus(), "
        "make sure you have a properly populated milvus/conf folder. "
        "See (https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md) for more details."
    )
    status = subprocess.run(
        [
            f"docker start {MILVUS1_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d --name {MILVUS1_CONTAINER_NAME} \
          -p 19530:19530 \
          -p 19121:19121 \
          milvusdb/milvus:1.1.0-cpu-d050721-5e559c"
        ],
        shell=True,
    )
    if status.returncode:
        logger.warning(
            "Tried to start Milvus through Docker but this failed. "
            "It is likely that there is already an existing Milvus instance running. "
        )
    else:
        time.sleep(sleep)
