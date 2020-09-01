What can Haystack Do?
=====================

!! This could be landing page !!

Haystack is an open-source toolkit for building end-to-end open domain QA systems.
Recent advances in Deep Learning and Language Modeling have enabled the application of QA to real world settings
and Haystack is designed to be the bridge between research and industry.

Search System
-------------

Take the leap from using keyword search on your own documents to semantic search with Haystack.
Store your documents in the database of your choice (Elasticsearch, SQL, in memory, FAISS) and
perform question driven queries.
Expect to see results that highlight the very sentence that contains the answer to your question.
Thanks to the power of Transformer based language models, results are chosen based on compatibility in meaning
rather than lexical overlap.
Search systems built with Haystack are designed to bridge the gap between the web search experience and local offline search.

!!Image!!

Information Extractor
---------------------

Automate the extraction of relevant information from a set of documents that pertain to the same topics but for different entities.
Say you have the financial reports for different companies over different years and
you have a set of standard questions which are applicable to each financial report,
like *what is the revenue forecast for 2020?* or *what are the main sources of income?*.
Haystack can be set up so that you can ask a set of questions and retrieve answers for each document separately.
In cases where the question can't be answered by a document, Haystack has the ability to refrain from giving an answer.

!!Image!!

We've seen this style of application be particularly effective in the sphere of finance and patent law
but we see a lot of potential in using this to gain a better overview of academic papers and internal business documents.

FAQ Style Question Answering
----------------------------

Leverage existing FAQ documents to answer new incoming questions.
Through the power of semantically driven similarity search,
Haystack will match incoming questions to questions in your FAQs and return the most relevant answer.
This is a quick way to give your customers more dynamic access to your existing documentation.

!!Image!!

API
---

Call on Haystack in your own applications through the API.
Haystack can be used as a flexible fall back option when chatbots are unable to classify user intents.

!!More?!!
