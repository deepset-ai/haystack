# Contributing to Haystack

We are very open to community contributions and appreciate anything that improves `haystack`! This includes fixings typos, adding missing documentation, fixing bugs or adding new features.
To avoid unnecessary work on either side, please stick to the following process:

1. Check if there is already [a related issue](https://github.com/deepset-ai/haystack/issues).
2. Open a new issue to start a quick discussion. Some features might be a nice idea, but don't fit in the scope of Haystack and we hate to close finished PRs!
3. Create a pull request in an early draft version and ask for feedback. If this is your first pull request and you wonder how to actually create a pull request, checkout [this manual](https://opensource.com/article/19/7/create-pull-request-github).
4. Verify that all tests in the CI pass (and add new ones if you implement anything new)

## Formatting of Pull Requests

Please give a concise description in the first comment in the PR that includes: 
- What is changing?
- Why? 
- What are limitations?
- Breaking changes (Example of before vs. after)
- Link the issue that this relates to

## Running tests

### CI
Tests will automatically run in our CI for every commit you push to your PR. This is the most convenient way for you and we encourage you to create early "WIP Pull requests".

### Local
However, you can also run the tests locally by executing pytest in your terminal from the `/test` folder.

#### Running all tests
**Important**: If you want to run **all** tests locally, you'll need **all** document stores running in the background before you run the tests.
Many of the tests will then be executed multiple times with different document stores.

You can launch them like this:
```
docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "ES_JAVA_OPTS=-Xms128m -Xmx128m" elasticsearch:7.9.2
docker run -d -p 19530:19530 -p 19121:19121 milvusdb/milvus:1.1.0-cpu-d050721-5e559c
docker run -d -p 8080:8080 --name haystack_test_weaviate --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED='true' --env PERSISTENCE_DATA_PATH='/var/lib/weaviate' semitechnologies/weaviate:1.7.2
docker run -d -p 7200:7200 --name haystack_test_graphdb deepset/graphdb-free:9.4.1-adoptopenjdk11
docker run -d -p 9998:9998 -e "TIKA_CHILD_JAVA_OPTS=-JXms128m" -e "TIKA_CHILD_JAVA_OPTS=-JXmx128m" apache/tika:1.24.1
```
Then run all tests:
```
cd test
pytest
```

#### Recommendation: Running a subset of tests
In most cases you rather want to run a **subset of tests** locally that are related to your dev:

The most important option to reduce the number of tests in a meaningful way, is to shrink the "test grid" of document stores.
This is possible by adding the `--document_store_type` arg to your pytest command. Possible values are: `"elasticsearch, faiss, memory, milvus, weaviate"`.
For example, calling `pytest . --document_store_type="memory"` will run all tests that can be run with the InMemoryDocumentStore, i.e.: 
- all the tests that we typically run on the whole "document store grid" will only be run for InMemoryDocumentStore
- any test that is specific to other document stores (e.g. elasticsearch) and is not supported by the chosen document store will be skipped (and marked in the logs accordingly)


Run tests that are possible for a **selected document store**. The InMemoryDocument store is a very good starting point as it doesn't require any of the external docker containers from above: 
```
pytest . --document_store_type="memory"
```
Run tests using a **combination of document stores**:
```
pytest . --document_store_type="memory,elasticsearch"
```
*Note: You will need to launch the elasticsearch container here as described above'*

Just run **one individual test**:
```
pytest -v test_retriever.py::test_dpr_embedding
```
Select a **logical subset of tests** via markers and the optional "not" keyword:
```
pytest -m not elasticsearch
pytest -m elasticsearch
pytest -m generator
pytest -m tika
pytest -m not slow
...
```


## Writing tests

If you are writing a test that depend on a document store, there are a few conventions to define on which document store type this test should/can run:

### Option 1: The test should run on all document stores / those supplied in the CLI arg `--document_store_type`: 
Use one of the fixtures `document_store` or `document_store_with_docs` or `document_store_type`.
Do not parameterize it yourself. 

Example:
```
def test_write_with_duplicate_doc_ids(document_store):
        ...
        document_store.write(docs)
        ....

```

### Option 2: The test is only compatible with certain document stores: 
Some tests you don't want to run on all possible document stores. Either because the test is specific to one/few doc store(s) or the test is not really document store related and it's enough to test it on one document store and speed up the execution time.

Example:
```
# Currently update_document_meta() is not implemented for InMemoryDocStore so it's not listed here as an option

@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss"], indirect=True)
def test_update_meta(document_store):
    ....
``` 

### Option 3: The test is not using a `document_store`/ fixture, but still has a hard requirement for a certain document store:

Example:
```
@pytest.mark.elasticsearch
def test_elasticsearch_custom_fields(elasticsearch_fixture):
    client = Elasticsearch()
    client.indices.delete(index='haystack_test_custom', ignore=[404])
    document_store = ElasticsearchDocumentStore(index="haystack_test_custom", text_field="custom_text_field",
                                                embedding_field="custom_embedding_field")
``` 
