<!---
title: "Rest API"
metaTitle: "Rest API"
metaDescription: ""
slug: "/docs/rest_api"
date: "2020-09-03"
id: "rest_apimd"
--->

# Rest API

We provide a REST API based on [FastAPI](https://fastapi.tiangolo.com/) that you can extend for your own purposes.

## Run 
To serve the API, clone the GitHub repository, adjust the config in `rest_api/config.py` and run:

``` 
gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker -t 300
```

<div class="alert info">
The REST API expects a running DocumentStore that is already containing your Documents.
If you don't know how to do that, check out the Tutorials.
</div>

## Documentation 
You will find the Swagger API documentation with all the detailed endpoints at http://127.0.0.1:8000/docs

## Major endpoints

`/models/{model_id}/doc-qa`  
Search answers in documents using extractive question answering

`/models/{model_id}/faq-qa`  
Search answers by comparing user question to existing questions (aka "FAQ-Style QA")

`/doc_qa_feedback`  
Collect user feedback on answers to gain domain-specific training data
