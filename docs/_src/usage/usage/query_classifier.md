<!---
title: "Query Classifier"
metaTitle: "Query Classifier"
metaDescription: ""
slug: "/docs/query_classifier"
date: "2021-08-17"
id: "query_classifiermd"
--->

# Query Classifier

The query classifier can potentially optimize the overall flow of Haystack pipeline by detecting the nature of user queries. Now, the Haystack can detect primarily three types of queries using both light-weight SKLearn Gradient Boosted classifier or Transformer based more robust classifier. The three categories of queries are as follows:


#### 1. Keyword Queries: 
Such queries don't have semantic meaning and merely consist of keywords. For instance these three are the examples of keyword queries.

*   arya stark father
*   jon snow country
*   arya stark younger brothers

#### 2. Interrogative Queries: 
In such queries users usually ask a question, regardless of presence of "?" in the query the goal here is to detect the intent of the user whether any question is asked or not in the query. For example:

*   who is the father of arya stark ?
*   which country was jon snow filmed ?
*   who are the younger brothers of arya stark ?

#### 3. Declarative Queries: 
Such queries are variation of keyword queries, however, there is semantic relationship between words. Fo example:

*   Arya stark was a daughter of a lord.
*   Jon snow was filmed in a country in UK.
*   Bran was brother of a princess.

### Keyword vs Question/Statement Sklearn Classifier

The keyword vs question/statement query classifier essentially distinguishes between the keyword queries and statements/questions. So you can intelligently route to different retrieval nodes based on the nature of the query. Using this classifier can potentially yield the following benefits:

*  Getting better search results (e.g. by routing only proper questions to DPR / QA branches and not keyword queries)
*  Less GPU costs (e.g. if 50% of your traffic is only keyword queries you could just use elastic here and save the GPU resources for the other 50% of traffic with semantic queries)

![image](https://user-images.githubusercontent.com/6007894/127831511-f55bad86-4b4f-4b54-9889-7bba37e475c6.png)

Below, we define a pipeline `SklQueryClassifier` and show how to use it:
Read more about the trained model and dataset used [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)

```python
from haystack.pipeline import SklearnQueryClassifier,
sklearn_keyword_classifier = Pipeline()
sklearn_keyword_classifier.add_node(component=SklearnQueryClassifier(), name="QueryClassifier", inputs=["Query"])
sklearn_keyword_classifier.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
sklearn_keyword_classifier.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
sklearn_keyword_classifier.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "DPRRetriever"])

# Run only the dense retriever on the full sentence query
res_1 = sklearn_keyword_classifier.run(
    query="Who is the father of Arya Stark?",
    top_k_retriever=10
)
print("DPR Results" + "\n" + "="*15)
print_answers(res_1)

# Run only the sparse retriever on a keyword based query
res_2 = sklearn_keyword_classifier.run(
    query="arya stark father",
    top_k_retriever=10
)
print("ES Results" + "\n" + "="*15)
print_answers(res_2)
```
### Keyword vs Question/Statement Transformer Classifier
The transformer classifier is more accurate than SkLearn classifier however, it requires more memory and most probably GPU for faster inference however the transformer size is roughly `50 MBs`. Whereas, SkLearn is less accurate however is much more faster and doesn't require GPU for inference.

Below, we define a `TransformersQueryClassifier` and show how to use it:

```python 
from haystack.pipeline import TransformersQueryClassifier
transformer_keyword_classifier = Pipeline()
transformer_keyword_classifier.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
transformer_keyword_classifier.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
transformer_keyword_classifier.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
transformer_keyword_classifier.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "DPRRetriever"])
```

### Question vs Statement Classifier

One possible use case of this classifier could be to route queries after the document retrieval to only send questions to QA reader and in case of declarative sentence, just return the DPR/ES results back to user to enhance user experience and only show answers when user explicitly asks it.

![image](https://user-images.githubusercontent.com/6007894/127864452-f931ea7f-2e62-4f59-85dc-056d56eb9295.png)

Below, we define a `TransformersQueryClassifier` and show how to use it:

Read more about the trained model and dataset used [here](https://huggingface.co/shahrukhx01/question-vs-statement-classifier)
```python
from haystack.pipeline import TransformersQueryClassifier
transformer_question_classifier = Pipeline()
transformer_question_classifier.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["Query"])
transformer_question_classifier.add_node(component=TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier"), name="QueryClassifier", inputs=["DPRRetriever"])
transformer_question_classifier.add_node(component=reader, name="QAReader", inputs=["QueryClassifier.output_1"])
```

### Standalone Query Classifier
Below we run queries classifiers standalone to better understand their outputs on each of the three types of queries

```python
from haystack.pipeline import TransformersQueryClassifier

queries = ["Lord Eddard was the father of Arya Stark.","Jon Snow was filmed in United Kingdom.",
           "who is the father of arya stark?","Which country was jon snow filmed in?"]

question_classifier = TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier")

for query in queries:
    result = question_classifier.run(query=query)
    if result[1] == "output_1":
        category = "question"
    else:
        category = "statement"

    print(f"Query: {query}, raw_output: {result}, class: {category}")
```
