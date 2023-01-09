docker run \
      --name elasticsearch \
      --net elastic \
      -p 9200:9200 \
      -e cluster.routing.allocation.disk.threshold_enabled=false \
      -e discovery.type=single-node \
      -e ES_JAVA_OPTS="-Xms3g -Xmx3g"\
      -e xpack.security.enabled=false \
      -it \
      docker.elastic.co/elasticsearch/elasticsearch:8.2.2
