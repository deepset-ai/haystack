#!/usr/bin/env sh
# Run Haystack API(on GPU) and Elasticsearch using Docker.
#
# docker-compose doesn't support GPUs in the current version. As a workaround,
# this script runs haystack-api and Elasticsearch Docker Images separately.
#
# To use GPU with Docker, ensure nvidia-docker(https://github.com/NVIDIA/nvidia-docker) is installed.

docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
# alternative: for a demo you can also use this elasticsearch image with already indexed GoT articles
#docker run -d -p 9200:9200 -e "discovery.type=single-node" deepset/elasticsearch-game-of-thrones

# wait for Elasticsearch server to start
sleep 30
docker run --net=host --gpus all -e READER_MODEL_PATH=deepset/roberta-base-squad2 -d deepset/haystack-gpu:latest
