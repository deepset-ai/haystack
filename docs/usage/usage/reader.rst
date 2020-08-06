Reader
======

Core tech that powers Haystack
Deep neural networks, currently almost exclusively transformers
a.k.a. closed domain QA systems in ML speak
Performs close reading of text to extract best answers

Combined with Retriever using Finder (link)
Code snippet on how to initialize
Code snippet on how to combine with retriever

From Language Model to Reader
-----------------------------

See blog article for more about how these reader models
adapt language models
pick a best start and end token
deal with documents of any length
Handle other answer types like no answer

Choosing the Right Model
------------------------

A general purpose language model further trained on a QA dataset, currently SQuAD is standard
Many options out there right now (BERT, RoBERTa, ELECTRA)
Evaluate on a trade off of accuracy and speed
Look at our benchmarks page and also SQuAD leaderboards
Models such as RoBERTa base trained on SQuAD (give model hub model name) are a good starting point - performant and can also run on V100
Very promising next generation of distilled models which shrink down size but retain performance
ALBERT XL is best performance but practically speaking too large

code snippet on how to load a model

FARM vs Transformers
--------------------

HF Transformers has become Core LM implementation
but pipelines around LM that make it QA are diff
Diff aggregation strat, diff speed, diff saving loading
Do we have stats on any of this? (point to Benchmarks)

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
