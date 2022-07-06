# Benchmarks



To run all benchmarks (e.g. for a new haystack release):

````
python run.py --reader --retriever_index --retriever_query --update-json --save_markdown
````

For custom runs, you can specify which components and processes to benchmark with the following flags:
```
python run.py [--reader] [--retriever_index] [--retriever_query] [--ci] [--update-json] [--save_markdown]

where

**--reader** will trigger the speed and accuracy benchmarks for the reader. Here we simply use the SQuAD dev set.

**--retriever_index** will trigger indexing benchmarks

**--retriever_query** will trigger querying benchmarks (embeddings will be loaded from file instead of being computed on the fly)

**--ci** will cause the the benchmarks to run on a smaller slice of each dataset and a smaller subset of Retriever / Reader / DocStores. 

**--update-json** will cause the script to update the json files in docs/_src/benchmarks so that the website benchmarks will be updated.
 
**--save_markdown** save results additionally to the default csv also as a markdown file
```

Results will be stored in this directory as
- retriever_index_results.csv (+ .md)
- retriever_query_results.csv (+ .md)
- reader_results.csv (+ .md)