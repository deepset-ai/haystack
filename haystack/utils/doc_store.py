import time
import logging
import subprocess
from pathlib import Path
from typing import Union, Tuple

import requests


logger = logging.getLogger(__name__)
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"
OPENSEARCH_CONTAINER_NAME = "opensearch"
WEAVIATE_CONTAINER_NAME = "weaviate"


def launch_es(sleep=15, delete_existing=False):
    """
    Start an Elasticsearch server via Docker.
    """

    logger.debug("Starting Elasticsearch ...")
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {ELASTICSEARCH_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)
    status = subprocess.run(
        [
            f'docker start {ELASTICSEARCH_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d -p 9200:9200 -e "discovery.type=single-node" --name {ELASTICSEARCH_CONTAINER_NAME} elasticsearch:7.17.6'
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


def launch_opensearch(sleep=15, delete_existing=False, local_port=9200):
    """
    Start an OpenSearch server via Docker.
    """
    logger.debug("Starting OpenSearch...")
    # This line is needed since it is not possible to start a new docker container with the name opensearch if there is a stopped image with the same now
    # docker rm only succeeds if the container is stopped, not if it is running
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {OPENSEARCH_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)
    status = subprocess.run(
        [
            f'docker start {OPENSEARCH_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d -p {local_port}:9200 -p 9600:9600 -e "discovery.type=single-node" --name {OPENSEARCH_CONTAINER_NAME} opensearchproject/opensearch:1.3.5'
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
    """
    Start a Weaviate server via Docker.
    """

    logger.debug("Starting Weaviate ...")
    status = subprocess.run(
        [
            f"docker start {WEAVIATE_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d -p 8080:8080 --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED='true' --env PERSISTENCE_DATA_PATH='/var/lib/weaviate' --name {WEAVIATE_CONTAINER_NAME} semitechnologies/weaviate:latest"
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
    logger.debug("Stopping %s...", container_name)
    status = subprocess.run([f"docker stop {container_name}"], shell=True)
    if status.returncode:
        logger.warning(
            "Tried to stop %s but this failed. It is likely that there was no Docker container with the name %s",
            container_name,
            container_name,
        )
    if delete_container:
        status = subprocess.run([f"docker rm {container_name}"], shell=True)


def stop_opensearch(delete_container=False):
    stop_container(OPENSEARCH_CONTAINER_NAME, delete_container)


def stop_elasticsearch(delete_container=False):
    stop_container(ELASTICSEARCH_CONTAINER_NAME, delete_container)


def stop_weaviate(delete_container=False):
    stop_container(WEAVIATE_CONTAINER_NAME, delete_container)


def stop_service(document_store, delete_container=False):
    ds_class = str(type(document_store))
    if "OpenSearchDocumentStore" in ds_class:
        stop_opensearch(delete_container)
    elif "ElasticsearchDocumentStore" in ds_class:
        stop_elasticsearch(delete_container)
    elif "WeaviateDocumentStore" in ds_class:
        stop_weaviate(delete_container)
    else:
        logger.warning("No support yet for auto stopping the service behind a %s", type(document_store))


def launch_milvus(sleep=15, delete_existing=False, timeout: Union[float, Tuple[float, float]] = 10.0):
    """Start a Milvus server via Docker

    :param sleep: How many seconds to wait after Milvus container is launched. Defaults to 15.
    :param delete_existing: Unused. Defaults to False.
    :param timeout: How many seconds to wait for the server to send data before giving up,
        as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
        Defaults to 10 seconds.
    """
    logger.debug("Starting Milvus ...")

    milvus_dir = Path.home() / "milvus"
    milvus_dir.mkdir(exist_ok=True)

    request = requests.get(
        "https://github.com/milvus-io/milvus/releases/download/v2.0.0/milvus-standalone-docker-compose.yml",
        timeout=timeout,
    )
    with open(milvus_dir / "docker-compose.yml", "wb") as f:
        f.write(request.content)

    status = subprocess.run([f"cd {milvus_dir} && docker-compose up -d"], shell=True)

    if status.returncode:
        logger.warning(
            "Tried to start Milvus through Docker but this failed. "
            "It is likely that there is already an existing Milvus instance running. "
        )
    else:
        time.sleep(sleep)
