# Benchmarks

Run the benchmarks with the following command:

```
python run.py [--reader] [--retriever_index] [--retriever_query] [--ci] [--update-json]
```

You can specify which components and processes to benchmark with the following flags.

**--reader** will trigger the speed and accuracy benchmarks for the reader. Here we simply use the SQuAD dev set.

**--retriever_index** will trigger indexing benchmarks

**--retriever_query** will trigger querying benchmarks (embeddings will be loaded from file instead of being computed on the fly)

**--ci** will cause the the benchmarks to run on a smaller slice of each dataset and a smaller subset of Retriever / Reader / DocStores. 

**--update-json** will cause the script to update the json files in docs/_src/benchmarks so that the website benchmarks will be updated.
 