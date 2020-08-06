What can Haystack Do?
=====================

Haystack is an open-source toolkit for building end-to-end open domain QA systems.
It can be used as the backend of your search system, as a flexible information extraction system or a ................
Based on DL transformer architectures.
Staying in touch with state of the art

Search System
-------------

Perform smarter search on your own documents
Haystack allows you to store documents in database of your choice (Elasticsearch, in memory, etc)
And perform highly specific question driven queries
E.g. Nvidia example

Information Extractor / Classifier
----------------------------------

You have a collection of similarly structured documents, or an incoming stream of documents like patents or earnings calls
You want to ask the same question for each
How did revenue develop in 2020?
Returns either a section of text that answers, or gives a no_answer meaning that the document does not have the info to answer the question

FAQ Style Question Answering
----------------------------

You have FAQ documents
New queries are matched to the most similar FAQ questions
Routes user on to that FAQ answer

API
---

Chatbot with predefined intents
but when can't classify intent, the incoming request can become an API call to Haystack
Haystack has flexibility

