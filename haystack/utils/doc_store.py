import time
import logging
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"
OPENSEARCH_CONTAINER_NAME = "opensearch"
WEAVIATE_CONTAINER_NAME = "weaviate"


def launch_es(
    sleep=15,
    delete_existing=False,
    version_tag: str = "7.17.6",
    password: Optional[str] = None,
    java_opts: Optional[str] = None,
):
    """
    Start an Elasticsearch server via Docker.

    :param sleep: Time to wait for Elasticsearch to start up.
    :param delete_existing: If True, delete an existing Elasticsearch instance before starting a new one.
    :param version_tag: Tag of the Elasticsearch version to use.
    :param password: Password for the Elasticsearch user 'elastic'. If set, 'xpack.security.enabled' will be set to
                     True. Otherwise, 'xpack.security.enabled' will be set to False.
    :param java_opts: Java options to pass to Elasticsearch.
    """

    logger.debug("Starting Elasticsearch ...")
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {ELASTICSEARCH_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)

    java_opts_str = f'-e ES_JAVA_OPTS="{java_opts}" ' if java_opts is not None else ""
    password_str = (
        f'-e "ELASTIC_PASSWORD={password}" -e "xpack.security.enabled=true" '
        if password is not None
        else '-e "xpack.security.enabled=false" '
    )

    command = (
        f"docker start {ELASTICSEARCH_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d "
        f'-p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" '
        f"{java_opts_str} {password_str}"
        f"--name {ELASTICSEARCH_CONTAINER_NAME} elasticsearch:{version_tag}"
    )

    status = subprocess.run([command], shell=True)
    if status.returncode:
        logger.warning(
            "Tried to start Elasticsearch through Docker but this failed. "
            "It is likely that there is already an existing Elasticsearch instance running. "
        )
    else:
        time.sleep(sleep)


def launch_opensearch(sleep=15, delete_existing=False, local_port=9200, java_opts: Optional[str] = None):
    """
    Start an OpenSearch server via Docker.
    """
    logger.debug("Starting OpenSearch...")
    # This line is needed since it is not possible to start a new docker container with the name opensearch if there is a stopped image with the same now
    # docker rm only succeeds if the container is stopped, not if it is running
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {OPENSEARCH_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)

    java_opts_str = f'-e OPENSEARCH_JAVA_OPTS="{java_opts}" ' if java_opts is not None else ""

    command = (
        f"docker start {OPENSEARCH_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d "
        f'-p {local_port}:9200 -p 9600:9600 -e "discovery.type=single-node" '
        f"{java_opts_str}"
        f"--name {OPENSEARCH_CONTAINER_NAME} opensearchproject/opensearch:1.3.5"
    )

    status = subprocess.run([command], shell=True)
    if status.returncode:
        logger.warning(
            "Tried to start OpenSearch through Docker but this failed. "
            "It is likely that there is already an existing OpenSearch instance running. "
        )
    else:
        time.sleep(sleep)


def launch_weaviate(sleep=15, delete_existing=False):
    """
    Start a Weaviate server via Docker.
    """

    logger.debug("Starting Weaviate ...")
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {WEAVIATE_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)
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
