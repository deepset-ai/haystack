<!---
title: "Pipelines"
metaTitle: "Pipelines"
metaDescription: ""
slug: "/docs/pipelines"
date: "2020-11-05"
id: "pipelinesmd"
--->

# Pipelines

### Flexibility powered by DAGs
In order to build modern search pipelines, you need two things: powerful building blocks and an easy way to stick them together.
The `Pipeline` class is exactly built for this purpose and enables many search scenarios beyond QA. 
The core idea is that you can build a Directed Acyclic Graph (DAG) where each node is one building block (Reader, Retriever, Generator ...). 
Here's a simple example for a standard Open-Domain QA Pipeline: 

```python
from haystack import Pipeline

p = Pipeline()
p.add_node(component=retriever, name="ESRetriever1", inputs=["Query"])
p.add_node(component=reader, name="QAReader", inputs=["ESRetriever1"])
res = p.run(query="What did Einstein work on?")
```

You can **draw the DAG** to better inspect what you are building:
```python
p.draw(path="custom_pipe.png")
```
![image](https://user-images.githubusercontent.com/1563902/102451716-54813700-4039-11eb-881e-f3c01b47ca15.png)

### Arguments

Each node in a Pipeline defines the arguments the run() method accepts. The Pipeline class takes care of passing relevant
arguments to the node. In addition to mandatory inputs like `query`, the `run()` accepts optional node parameters like 
`top_k` with the `params` argument. For instance, `params={"top_k": 5}` will set the `top_k` of all nodes as 5. To 
target params to a specific node, the node name can be explicitly specifie as `params={"Retriever": {"top_k": 5}}`.


```python
res = pipeline.run(
    query="What did Einstein work on?",
    params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 3}}
)
```

### YAML File Definitions

For your convenience, there is also the option of defining and loading pipelines in YAML files.
Having your pipeline available in a YAML is particularly useful when 
you move between experimentation and production environments. 
Just export the YAML from your notebook / IDE and import it into your production environment. 
It also helps with version control of pipelines, allows you to share your pipeline easily with colleagues, 
and simplifies the configuration of pipeline parameters in production.

For example, you can define and save a simple Retriever Reader pipeline by saving the following to a file:

```yaml
version: '0.7'

components:    # define all the building-blocks for Pipeline
- name: MyReader       # custom-name for the component; helpful for visualization & debugging
  type: FARMReader    # Haystack Class name for the component
  params:
    no_ans_boost: -10
    model_name_or_path: deepset/roberta-base-squad2
- name: MyESRetriever
  type: ElasticsearchRetriever
  params:
    document_store: MyDocumentStore    # params can reference other components defined in the YAML
    custom_query: null
- name: MyDocumentStore
  type: ElasticsearchDocumentStore
  params:
    index: haystack_test

pipelines:    # multiple Pipelines can be defined using the components from above
- name: my_query_pipeline    # a simple extractive-qa Pipeline
  nodes:
  - name: MyESRetriever
    inputs: [Query]
  - name: MyReader
    inputs: [MyESRetriever]
```

To load, simply call:

```python
pipeline.load_from_yaml(Path("sample.yaml"))
```

For another example YAML config, check out [this file](https://github.com/deepset-ai/haystack/blob/master/rest_api/pipeline/pipelines.yaml).

### Multiple retrievers
You can now also use multiple Retrievers and join their results: 
```python
p = Pipeline()
p.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
p.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["Query"])
p.add_node(component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["ESRetriever", "DPRRetriever"])
p.add_node(component=reader, name="QAReader", inputs=["JoinResults"])
res = p.run(query="What did Einstein work on?", params={"ESRetriever": {"top_k": 1}, "DPRRetriever": {"top_k": 3}})
```
![image](https://user-images.githubusercontent.com/1563902/102451782-7bd80400-4039-11eb-9046-01b002a783f8.png)

### Custom nodes
It is easy to build custom nodes. Just respect the following requirements: 

1. Create a Class that inherits from `BaseComponent`. 
2. Add a `run()` method to your class with any parameters it needs to process the input. Ensure that the parameters are either passed with `params` to the pipeline or are returned by the preceding nodes.
3. Do whatever you want within `run()` (e.g., reformatting the query).
4. Return a tuple that contains your output data (for the next node) and the name of the outgoing edge `output_dict, "output_1`.
5. Add a class attribute `outgoing_edges = 1` that defines your node's number of output options. You only need a higher number here if you have a decision node (see below).

### Decision nodes
Or you can add decision nodes where only one "branch" is executed afterwards. This allows, for example, to classify an incoming query and depending on the result routing it to different modules: 
![image](https://user-images.githubusercontent.com/1563902/102452199-41229b80-403a-11eb-9365-7038697e7c3e.png)
```python
    from haystack import BaseComponent, Pipeline

    class QueryClassifier(BaseComponent):
        outgoing_edges = 2

        def run(self, query):
            if "?" in query:
                return {}, "output_1"

            else:
                return {}, "output_2"

    pipe = Pipeline()
    pipe.add_node(component=QueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipe.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_1"])
    pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_2"])
    pipe.add_node(component=JoinDocuments(join_mode="concatenate"), name="JoinResults",
                  inputs=["ESRetriever", "DPRRetriever"])
    pipe.add_node(component=reader, name="QAReader", inputs=["JoinResults"])
    res = p.run(query="What did Einstein work on?", params={"ESRetriever": {"top_k": 1}, "DPRRetriever": {"top_k": 3}})
```

### Evaluation nodes

There are nodes in Haystack that are used to evaluate the performance of readers, retrievers and combine systems.
To get hands on with this kind of node, have a look at the [evaluation tutorial](/docs/latest/tutorial5md).

### Default Pipelines (replacing the "Finder")
Last but not least, we added some "Default Pipelines" that allow you to run standard patterns with very few lines of code.
This is replacing the `Finder` class which was deprecated with Haystack 0.6.0 .

```
from haystack.pipeline import DocumentSearchPipeline, ExtractiveQAPipeline, Pipeline, JoinDocuments

# Extractive QA
qa_pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)
res = qa_pipe.run(query="When was Kant born?", params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 5}})

# Document Search
doc_pipe = DocumentSearchPipeline(retriever=retriever)
res = doc_pipe.run(query="Physics Einstein", params={"Retriever": {"top_k": 3}})

# Generative QA
doc_pipe = GenerativeQAPipeline(generator=rag_generator, retriever=retriever)
res = doc_pipe.run(query="Physics Einstein", params={"Retriever": {"top_k": 3}})

# FAQ based QA
doc_pipe = FAQPipeline(retriever=retriever)
res = doc_pipe.run(query="How can I change my address?", params={"Retriever": {"top_k": 3}})

```    
So to migrate your QA system from the deprecated `Finder` to `ExtractiveQAPipeline` you'd need to: 
```
# 1. Change import
from haystack.pipeline import  ExtractiveQAPipeline

# 2. Replace the Finder 
qa_pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# 3. Replace get_answers() with run()
res = qa_pipe.run(query="When was Kant born?", top_k_retriever=3, top_k_reader=5)

# 4. Access your results from ["documents"] rather than ["answers"]
print(res["documents"])
```
See also the [Pipelines API documentation](/docs/latest/apipipelinesmd) for more details. 

We plan many more features around the new pipelines incl. parallelized execution, distributed execution, dry runs - so stay tuned ...  

