Get Started
===========

Installation
--------------

.. tabs::
    .. tab:: Basic

        The most straightforward way to install Haystack is through pip.

        .. code-block:: bash

            $ pip install farm-haystack

    .. tab:: Editable

        If you'd like to run a specific, unreleased version of Haystack, or make edits to the way Haystack runs,
        you'll want to install it using ``git`` and ``pip --editable``.
        This clones a copy of the repo to a local directory and runs Haystack from there.

        .. code-block:: bash

            $ git clone https://github.com/deepset-ai/haystack.git
            $ cd haystack
            $ pip install --editable .

        By default, this will give you the latest version of the master branch.
        Use regular git commands to switch between different branches and commits.

..
   _comment: !! Have a tab for docker!!

..
   -comment: !! Have a hello world example!!

Basic Code Example
------------------

Here's a sample of some Haystack code showing the most important components.
For a working code example, check out our starter tutorial (!!link!!).

.. code-block::

    # Database
    document_store = ElasticsearchDocumentStore()

    # Clean & index your docs
    dicts = convert_files_to_dicts(doc_dir, clean_func=clean_wiki_text)
    document_store.write_documents(dicts)

    # Init Retriever
    retriever = ElasticsearchRetriever(document_store)

    # Init Reader
    model_name = "deepset/roberta-base-squad2"
    reader = FARMReader(model_name)

    # Combine Reader and Retriever in Finder
    finder = Finder(reader, retriever)

    # Voilà! Ask a question!
    question = "Who is the father of Sansa Stark?"
    prediction = finder.get_answers(question)
    print_answers(prediction)
