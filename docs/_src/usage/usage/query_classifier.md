<!---
title: "Query Classifier"
metaTitle: "Query Classifier"
metaDescription: ""
slug: "/docs/query_classifier"
date: "2021-08-17"
id: "query_classifiermd"
--->

# Query Classifier

Queries come in all shapes and forms. A keyword-based search differs from a question posed in natural language. In Haystack, we can account for these differences by integrating a special node into our QA pipeline: the query classifier. 

A query classifier puts each incoming query into one of two predefined classes, and routes it to the appropriate section of the pipeline.
Haystack comes with classifiers to distinguish between the three most common query types (Keywords, Question, Statement) and allows two different types of models (SKlearn and Transformer). 

Using a query classifier can potentially yield the following benefits:

*  Getting better search results (e.g. by routing only proper questions to DPR / QA branches and not keyword queries)
*  Less GPU costs (e.g. if 50% of your traffic is only keyword queries you could just use elastic here and save the GPU resources for the other 50% of traffic with semantic queries)


### Common Query types

#### 1. Keyword Queries: 
Such queries don't have semantic meaning, merely consist of keywords and the order of words does not matter:
*   arya stark father
*   jon snow country
*   arya stark younger brothers

#### 2. Questions (Interrogative Queries): 
In such queries users ask a question in a complete, "natural" sentence. Regardless of the presence of "?" in the query the goal here is to detect the intent of the user whether any question is asked or not in the query:

*   who is the father of arya stark?
*   which country was jon snow filmed in
*   who are the younger brothers of arya stark?

#### 3. Statements (Declarative Queries): 
Such queries consist also of a regular, natural sentence with semantic relations between the words. However, they are rather a statement than a question:

*   Arya stark was a daughter of a lord.
*   Show countries that Jon snow was filmed in.
*   List all brothers of Arya.

### Usage standalone: Try a Query Classifier
To test how a query classifier works before integrating it into a pipeline, you can run it just as an individual component:

```python
from haystack.pipeline import TransformersQueryClassifier

queries = ["Arya Stark father","Jon Snow UK",
           "who is the father of arya stark?","Which country was jon snow filmed in?"]

question_classifier = TransformersQueryClassifier(model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection")
# Or Sklearn based:  

for query in queries:
    result = question_classifier.run(query=query)
    if result[1] == "output_1":
        category = "question"
    else:
        category = "keywords"

    print(f"Query: {query}, raw_output: {result}, class: {category}")

# Returns:
# Query: Arya Stark father, raw_output: ({'query': 'Arya Stark father'}, 'output_2'), class: keywords
# Query: Jon Snow UK, raw_output: ({'query': 'Jon Snow UK'}, 'output_2'), class: keywords
# Query: who is the father of arya stark?, raw_output: ({'query': 'who is the father of arya stark?'}, 'output_1'), class: question
# Query: Which country was jon snow filmed in?, raw_output: ({'query': 'Which country was jon snow filmed in?'}, 'output_1'), class: question

```
Note how the node returns two objects: the query (e.g.'Arya Stark father') and the name of the output edge (e.g. "output_2"). This information can be leveraged in a pipeline for routing the query to the next node.  

### Usage in a pipeline: Use different retrievers depending on the query type

You can use a Query Classifier within a pipeline as a "decision node". Depending on the output of the classifier other parts of the pipeline will be executed. For example, we can route keyword queries to an ElasticsearchRetriever and semantic queries (questions/statements) to DPR.  

![image](https://user-images.githubusercontent.com/6007894/127831511-f55bad86-4b4f-4b54-9889-7bba37e475c6.png)

Below, we define a pipeline with a `TransformersQueryClassifier` that routes questions/statements to the node's `output_1` and keyword queries to `output_2`. We leverage this structure in the pipeline by connecting the DPRRetriever to `QueryClassifier.output_1` and the ESRetriever to `QueryClassifier.output_2`. 

```python
from haystack.pipeline import TransformersQueryClassifier, Pipeline
from haystack.utils import print_answers

query_classifier = TransformersQueryClassifier(model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection")

pipe = Pipeline()
pipe.add_node(component=query_classifier, name="QueryClassifier", inputs=["Query"])
pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
pipe.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])

# Pass a question -> run DPR
res_1 = pipe.run(
    query="Who is the father of Arya Stark?",
    top_k_retriever=10
)

# Pass keywords -> run the ElasticsearchRetriever
res_2 = pipe.run(
    query="arya stark father",
    top_k_retriever=10
)

```
### Usage in a pipeline: Run QA only on proper questions

If you add QA to an existing search system, it can make sense to only use it for real questions that come in and keep a basic document search with elasticsearch for the remaining keyword queries. You can use a Query Classifier to build such a hybrid pipeline: 

```python
haystack.pipeline import TransformersQueryClassifier, Pipeline
from haystack.utils import print_answers

query_classifier = TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier")

pipe = Pipeline()
pipe.add_node(component=query_classifier, name="QueryClassifier", inputs=["Query"])
pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
pipe.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
pipe.add_node(component=reader, name="QAReader", inputs=["DPRRetriever"])

# Pass a question -> run DPR + QA -> return answers
res_1 = pipe.run(
    query="Who is the father of Arya Stark?",
    top_k_retriever=10
)

# Pass keywords -> run only ElasticsearchRetriever -> return docs
res_2 = pipe.run(
    query="arya stark father",
    top_k_retriever=10
)

```


### Which models are available?
The transformer classifier is more accurate than the SkLearn classifier as it can use the context and order of words. However, it requires more memory and most probably GPU for faster inference. You can mitigate those down sides by choosing a very small transformer model. The default models we trained are using a mini BERT architecture which is only about `50 MB` in size and allows relatively fast inference on CPU.

#### Transformers 
Pass your own `Transformer` binary classification model from file/huggingface or use one of the following pretrained ones hosted on Huggingface:
1) Keywords vs. Questions/Statements (Default)

   ```python
   TransformersQueryClassifier(model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection") 
   # output_1 => question/statement 
   # output_2 => keyword query 
   ```
   
   [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)  


2) Questions vs. Statements
    ```python
    TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier") 
    # output_1 => question  
    # output_2 => statement 
    ```
    
    [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)  


#### Sklearn
Pass your own `Sklearn` binary classification model or use one of the following pretrained Gradient boosting models:

1) Keywords vs. Questions/Statements (Default)

    ```python
    SklearnQueryClassifier(query_classifier = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle",
                      query_vectorizer = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle")
                      
    # output_1 => question/statement  
    # output_2 => keyword query  
    ```
    [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)  


2) Questions vs. Statements

    ```python
    SklearnQueryClassifier(query_classifier = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/model.pickle",
                      query_vectorizer = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/vectorizer.pickle")

    output_1 => question  
    output_2 => statement 
    ```
    [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)  
