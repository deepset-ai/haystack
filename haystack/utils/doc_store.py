import time
import logging
import subprocess


logger = logging.getLogger(__name__)


def launch_es(sleep=15):
    # Start an Elasticsearch server via Docker

    logger.debug("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'], shell=True
    )
    if status.returncode:
        logger.warning("Tried to start Elasticsearch through Docker but this failed. "
                       "It is likely that there is already an existing Elasticsearch instance running. ")
    else:
        time.sleep(sleep)

def launch_open_distro_es(sleep=15):
    # Start an Open Distro for Elasticsearch server via Docker

    logger.debug("Starting Open Distro for Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" amazon/opendistro-for-elasticsearch:1.13.2'], shell=True
    )
    if status.returncode:
        logger.warning("Tried to start Open Distro for Elasticsearch through Docker but this failed. "
                       "It is likely that there is already an existing Elasticsearch instance running. ")
    else:
        time.sleep(sleep)

def launch_opensearch(sleep=15):
    # Start an OpenSearch server via docker

    logger.debug("Starting OpenSearch...")
    # This line is needed since it is not possible to start a new docker container with the name opensearch if there is a stopped image with the same now
    # docker rm only succeeds if the container is stopped, not if it is running
    _ = subprocess.run(['docker rm opensearch'], shell=True, stdout=subprocess.DEVNULL)
    status = subprocess.run(
        ['docker run -d -p 9201:9200 -p 9600:9600 -e "discovery.type=single-node" --name opensearch opensearchproject/opensearch:1.0.0-rc1'],
        shell=True
    )
    if status.returncode:
        logger.warning("Tried to start OpenSearch through Docker but this failed. "
                       "It is likely that there is already an existing OpenSearch instance running. ")
    else:
        time.sleep(sleep)


def launch_weaviate(sleep=15):
    # Start a Weaviate server via Docker

    logger.debug("Starting Weaviate ...")
    status = subprocess.run(
        ["docker run -d -p 8080:8080 --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED='true' --env PERSISTENCE_DATA_PATH='/var/lib/weaviate' semitechnologies/weaviate:1.7.2"], shell=True
    )
    if status.returncode:
        logger.warning("Tried to start Weaviate through Docker but this failed. "
                       "It is likely that there is already an existing Weaviate instance running. ")
    else:
        time.sleep(sleep)


def stop_opensearch():
    logger.debug("Stopping OpenSearch...")
    status = subprocess.run(['docker stop opensearch'], shell=True)
    if status.returncode:
        logger.warning("Tried to stop OpenSearch but this failed. "
                       "It is likely that there was no OpenSearch Docker container with the name opensearch")
    status = subprocess.run(['docker rm opensearch'], shell=True)


def stop_service(document_store):
    ds_class = str(type(document_store))
    if "OpenSearchDocumentStore" in ds_class:
        stop_opensearch()
    else:
        logger.warning(f"No support yet for auto stopping the service behind a {ds_class}")


def launch_milvus(sleep=15):
    # Start a Milvus server via docker

    logger.debug("Starting Milvus ...")
    logger.warning("Automatic Milvus config creation not yet implemented. "
                   "If you are starting Milvus using launch_milvus(), "
                   "make sure you have a properly populated milvus/conf folder. "
                   "See (https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md) for more details.")
    status = subprocess.run(
        ['sudo docker run -d --name milvus_cpu_1.0.0 \
          -p 19530:19530 \
          -p 19121:19121 \
          -v /home/$USER/milvus/db:/var/lib/milvus/db \
          -v /home/$USER/milvus/conf:/var/lib/milvus/conf \
          -v /home/$USER/milvus/logs:/var/lib/milvus/logs \
          -v /home/$USER/milvus/wal:/var/lib/milvus/wal \
          milvusdb/milvus:1.0.0-cpu-d030521-1ea92e'
        ],
        shell=True
    )
    if status.returncode:
        logger.warning("Tried to start Milvus through Docker but this failed. "
                       "It is likely that there is already an existing Milvus instance running. ")
    else:
        time.sleep(sleep)
