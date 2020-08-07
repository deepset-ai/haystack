Reader
======

The Reader, also known as Open-Domain QA systems in Machine Learning speak,
is the core component that enables Haystack to find the answers that you need.
Haystack uses Readers built on the latest transformer based language models.
Their strong grasp of semantics and sensitivity to syntactic structure
have enabled them to reach state-of-the-art performance on question answering tasks such as SQuAD and Natural Questions.

In Haystack, they can be initialized either through the deepset FARM library or HuggingFace Transformers.
All steps required to perform question answering, including tokenization, embedding computation,
span prediction and candidate aggregation, are handled by the Reader.

.. tabs::

    .. tab:: FARM

        .. code-block:: python

            model = "deepset/roberta-base-squad2"
            reader = FARMReader(model, use_gpu=True)

    .. tab:: Transformers

        .. code-block:: python

            model = "deepset/roberta-base-squad2"
            reader = TransformersReader(model, use_gpu=1)

While these models can run on CPU it is generally recommended that they be run with GPU acceleration (!!see benchmarks!!).

On the higher level, the Reader is combined with the Retriever using the Finder class.

.. code-block:: python

    # Combine Reader and Retriever in Finder
    finder = Finder(reader, retriever)

From Language Model to Reader
-----------------------------

See blog article for more about how these reader models
adapt language models
pick a best start and end token
deal with documents of any length
Handle other answer types like no answer

When fed a query and a document, the model will predict the probability of each token being the start or end of an answer span.


Choosing the Right Model
------------------------

A general purpose language model further trained on a QA dataset, currently SQuAD is standard
Many options out there right now (BERT, RoBERTa, ELECTRA)
Evaluate on a trade off of accuracy and speed
Look at our benchmarks page and also SQuAD leaderboards
Models such as RoBERTa base trained on SQuAD (give model hub model name) are a good starting point - performant and can also run on V100
Very promising next generation of distilled models which shrink down size but retain performance
ALBERT XL is best performance but practically speaking too large

Can download from HF model hub

code snippet on how to load a model

FARM vs Transformers
--------------------

HF Transformers has become Core LM implementation
but pipelines around LM that make it QA are diff
Diff aggregation strat, diff speed, diff saving loading
Do we have stats on any of this? (point to Benchmarks)

Use tabbed element to show how the two are initialized

Languages other than English
----------------------------

By default, we have so far been talking about English Readers
If you want to do QA in a language other than English, there are a few different routes

1) Train on a non-English dataset
e.g. if you're working in French, take a French LM like CamemBERT (link), train on FQuAD (link)
In the LM world there is a lot of coverage of different languages
In terms of QA datasets, it is limited (see blog) (German coming soon?)

Code Snippet on how to do this? either trianing or loading one

2) Zero shot
Models like multilingual BERT, XLM and XLM-RoBERTa are trained to handle a broad range of languages
zero shot learning is to train a German Reader without any German QA data
The process goes somehting like this: Teach a multilingual model to do QA in English (or any other with a good training dataset)
If trained well, the hope is that the model should learn the language independent principles of QA
You can start using it for your language
In practice, zero shotting has effect but not yet really performant

Code Snippet for zero shot trained model?
Either train one, or just load a zero shot XLMR

Saving and Loading
------------------

CODE SNIPPET

FARM vs TRANSFORMERS Which ones can be loaded from local save, which ones can't
