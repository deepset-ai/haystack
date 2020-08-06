Get Started
===========

How to install
--------------

Setup python and environment

.. tabs::
    .. tab:: Ubuntu

        This is what you do for Ubuntu

    .. tab:: MacOS

        This is what you do for MacOS

    .. tab:: Windows

        This is what you do for Windows

.. tabs::
    .. tab:: Pip

        The simplest way to install the latest release of Haystack is through pip.

        .. code-block:: bash

            $ pip install farm-haystack

    .. tab:: Git

        If you're after the latest version of the master branch, you can clone the repository to a local directory and run Haystack from there.

        .. code-block:: bash

            $ git clone https://github.com/deepset-ai/haystack.git
            $ cd haystack
            $ pip install --editable .

        This will also allow you to run the code with any edits that you make to it.
        Update by simply calling:

        .. code-block:: bash

            $ git pull

Docker
------

.. code-block::

    !~TODO~!

Basic Code Example
------------------

For a working code example, see ~TODO~ (https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#ref-role)
Maybe use a button directive? (.. button::)

.. code-block::

    # Database to store your docs
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # Clean & index your docs
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    document_store.write_documents(dicts)

    # Init Retriever: Fast & simple algo to identify most promising candidate docs
    retriever = ElasticsearchRetriever(document_store=document_store)

    # Init Reader:
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

    # The Finder sticks together reader and retriever in a pipeline to answer our actual questions.
    finder = Finder(reader, retriever)

    # ## Voilà! Ask a question!
    prediction = finder.get_answers(question="Who is the father of Sansa Stark?", top_k_retriever=10, top_k_reader=5)
    print_answers(prediction, details="minimal")

