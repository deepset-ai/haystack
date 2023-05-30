# Benchmarks

The tooling provided in this directory allows running benchmarks on reader pipelines, retriever pipelines,
and retriever-reader pipelines.

## Defining configuration

To run a benchmark, you need to create a configuration file first. This file should be a Pipeline YAML file that
contains both the querying and, optionally, the indexing pipeline, in case the querying pipeline includes a retriever.

The configuration file should also have a **`benchmark_config`** section that includes the following information:

- **`labels_file`**: The path to a SQuAD-formatted JSON or CSV file that contains the labels to be benchmarked on.
- **`documents_directory`**: The path to a directory containing files intended to be indexed into the document store.
                             This is only necessary for retriever and retriever-reader pipelines.
- **`data_url`**: This is optional. If provided, the benchmarking script will download data from this URL and
                  save it in the **`data/`** directory.

Here is an example of how a configuration file for a retriever-reader pipeline might look like:

```yaml
components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
  - name: TextConverter
    type: TextConverter
  - name: Reader
    type: FARMReader
    params:
      model_name_or_path: deepset/roberta-base-squad2-distilled
  - name: Retriever
    type: BM25Retriever
    params:
      document_store: DocumentStore
      top_k: 10

pipelines:
  - name: indexing
    nodes:
      - name: TextConverter
        inputs: [File]
      - name: Retriever
        inputs: [TextConverter]
      - name: DocumentStore
        inputs: [Retriever]
  - name: querying
    nodes:
      - name: Retriever
        inputs: [Query]
      - name: Reader
        inputs: [Retriever]

benchmark_config:
  data_url: http://example.com/data.tar.gz
  documents_directory: /path/to/documents
  labels_file: /path/to/labels.csv
```

## Running benchmarks

Once you have your configuration file, you can run benchmarks by using the **`run.py`** script.

```bash
python run.py [--output OUTPUT] config
```

The script takes the following arguments:

- `config`: This is the path to your configuration file.
- `--output`: This is an optional path where benchmark results should be saved. If not provided, the script will create a JSON file with the same name as the specified config file.

## Metrics

The benchmarks yield the following metrics:

- Reader pipelines:
    - Exact match score
    - F1 score
    - Total querying time
    - Seconds/query
- Retriever pipelines:
    - Recall
    - Mean-average precision
    - Total querying time
    - Seconds/query
    - Queries/second
    - Total indexing time
    - Number of indexed Documents/second
- Retriever-Reader pipelines:
    - Exact match score
    - F1 score
    - Total querying time
    - Seconds/query
    - Total indexing time
    - Number of indexed Documents/second

You can find more details about the performance metrics in our [evaluation guide](https://docs.haystack.deepset.ai/docs/evaluation).
