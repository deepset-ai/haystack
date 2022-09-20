# Benchmarks



To start all benchmarks (e.g. for a new Haystack release), run:

````
python run.py --reader --retriever_index --retriever_query --update_json --save_markdown
````

For custom runs, you can specify which components and processes to benchmark with the following flags:
```
python run.py [--reader] [--retriever_index] [--retriever_query] [--ci] [--update_json] [--save_markdown]

where

**--reader** will trigger the speed and accuracy benchmarks for the reader. Here we simply use the SQuAD dev set.

**--retriever_index** will trigger indexing benchmarks

**--retriever_query** will trigger querying benchmarks (embeddings will be loaded from file instead of being computed on the fly)

**--ci** will cause the the benchmarks to run on a smaller slice of each dataset and a smaller subset of Retriever / Reader / DocStores. 

**--update-json** will cause the script to update the json files in docs/_src/benchmarks so that the website benchmarks will be updated.
 
**--save_markdown** save results additionally to the default csv also as a markdown file
```

Results will be stored in this directory as
- retriever_index_results.csv and retriever_index_results.md
- retriever_query_results.csv and retriever_query_results.md
- reader_results.csv and reader_results.md


# Temp. Quickfix for bigger runs

For bigger indexing runs (500k docs) the standard elastic / opensearch container that we spawn via haystack might run OOM. 
Therefore, start them manually before you trigger the benchmark script and assign more memory to them: 

`docker start opensearch > /dev/null 2>&1 || docker run -d -p 9201:9200 -p 9600:9600 -e "discovery.type=single-node" -e "OPENSEARCH_JAVA_OPTS=-Xms4096m -Xmx4096m" --name opensearch opensearchproject/opensearch:2.2.1`

and

`docker start elasticsearch > /dev/null 2>&1 || docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "ES_JAVA_OPTS=-Xms4096m -Xmx4096m" --name elasticsearch elasticsearch:7.9.2`
