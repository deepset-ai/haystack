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


### 1. Keyword Queries: 
Such queries don't have semantic meaning and merely consist of keywords. For instance these three are the examples of keyword queries.

*   arya stark father
*   jon snow country
*   arya stark younger brothers

### 2. Interrogative Queries: 
In such queries users usually ask a question, regardless of presence of "?" in the query the goal here is to detect the intent of the user whether any question is asked or not in the query. For example:

*   who is the father of arya stark ?
*   which country was jon snow filmed ?
*   who are the younger brothers of arya stark ?

### 3. Declarative Queries: 
Such queries are variation of keyword queries, however, there is semantic relationship between words. Fo example:

*   Arya stark was a daughter of a lord.
*   Jon snow was filmed in a country in UK.
*   Bran was brother of a princess.

## Keyword vs Question/Statement Classifier

The keyword vs question/statement query classifier essentially distinguishes between the keyword queries and statements/questions. So you can intelligently route to different retrieval nodes based on the nature of the query. Using this classifier can potentially yield the following benefits:

*  Getting better search results (e.g. by routing only proper questions to DPR / QA branches and not keyword queries)
*  Less GPU costs (e.g. if 50% of your traffic is only keyword queries you could just use elastic here and save the GPU resources for the other 50% of traffic with semantic queries)

![image](https://user-images.githubusercontent.com/6007894/127831511-f55bad86-4b4f-4b54-9889-7bba37e475c6.png)


