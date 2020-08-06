Overview
========

Retriever-Reader Pipeline
-------------------------

Haystack based on powerful 2 stage process

What is reader
power of QA
Transformer DL based

Retriever works in conjunction
Light weight search for relevant documents
Proposes candidates

Diagram!!!

Trade off?
but most documents probs irrelevant anyway so let's throw out the obviously irrel

See benchmarks

Indexing and Querying
---------------------

Two very different stages
Depends somehwat on which approach is chosen

Indexing is the process of storing your documents in a way that will optimize for when you search
Indexing performed just once at the point of loading data into database
For keyword based retrievers like elastic search, this involves the creation of an inverted index (word -> pages).
In dense approaches, this involves creating the document embeddings that get compared to your query embedding. (REPHRASE)

Diagrams of inverted index / document embeds

Highlight some words.

.. code-block::

    # Database to store your docs
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # Clean & index your docs
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    document_store.write_documents(dicts)

Give query, engage full system on and indexed database
Gain answer
Good indexing methods take time away from querying

.. code-block::

    # The Finder sticks together reader and retriever in a pipeline to answer our actual questions.
    finder = Finder(reader, retriever)

    # ## Voilà! Ask a question!
    prediction = finder.get_answers(question="Who is the father of Sansa Stark?", top_k_retriever=10, top_k_reader=5)

Customising Haystack
--------------------

Many options for each component
Non exhaustive list

Doc store - inmemory, elasticsearch, sql, dpr
Reader - BERT, RoBERTa, ELECTRA etc in their FARM and Transformers variants
Retriever - BM25, TFIDF, DPR
File Converters - txt, pdf, docx
Top K
Reader model params (batch size, max seq len, doc stride)

