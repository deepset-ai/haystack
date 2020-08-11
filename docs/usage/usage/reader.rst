Reader
======

The Reader, also known as Open-Domain QA systems in Machine Learning speak,
is the core component that enables Haystack to find the answers that you need.
Haystack uses Readers built on the latest transformer based language models.
Their strong grasp of semantics and sensitivity to syntactic structure
have enabled them to reach state-of-the-art performance on question answering tasks such as SQuAD and Natural Questions.

In Haystack, you can start using pretrained QA models simply by providing its HuggingFace model hub name to the Reader.
The loading of model weights is handled by Haystack,
and you have the option of using the QA pipeline from deepset FARM or HuggingFace Transformers (see :ref:`FARM vs Transformers` for details).

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

Language models form the core of most modern NLP systems and that includes the Readers in Haystack.
They build a general understanding of language when performing training tasks such as Masked Language Modeling or Replaced Token Detection
on large amounts of text.
Well trained language models capture the word distribution in one or more languages
but more importantly, convert input text into a set of word vectors that capture elements of syntax and semantics.
The latest generation of language models are generally transformer based and include models such as BERT, RoBERTa and ELECTRA.

In order to convert a language model into a Reader model, it needs first to be trained on a Question Answering dataset.
To do so requires the addition of a question answering prediction head on top of the language model
which performs answer span extraction.
The task can be thought of as a token classification task where every input token is assigned a probability of being
either the start or end token of the correct answer.
In cases where the answer is not contained within the passage, the prediction head is also expected to return a ``no_answer`` prediction.

!! Diagram of language model / prediction head !!

Since language models are limited in the number of tokens which they can process in a single forward pass,
a sliding window mechanism is implemented to handle variable length documents.
This functions by slicing the document into overlapping passages of (approximately) ``max_seq_length``
that are each offset by ``doc_stride`` number of tokens.
These can be set when the Reader is initialized.

.. tabs::

    .. tab:: FARM

        .. code-block:: python

            reader = FARMReader(... max_seq_len=384, doc_stride=128 ...)

    .. tab:: Transformers

        !! CAN'T CURRENTLY BE SET YET !!

        .. code-block:: python

            reader = TransformersReader(model, use_gpu=1)


Predictions are made on each individual passage and the process of aggregation picks the best candidates across all passages.
If you'd like to learn more about what is happening behind the scenes, have a look at `this <https://medium.com/deepset-ai/modern-question-answering-systems-explained-4d0913744097>`_ article.

!! Diagram from Blog !!

.. farm-vs-trans:

FARM vs Transformers
--------------------

Apart from the model weights, Haystack Readers contain all the components found in end-to-end open domain QA systems.
This includes tokenization, embedding computation, span prediction and candidate aggregation.
While the handling of model weights is the same between the FARM and Transformers libraries, their QA pipelines differ in some ways.
The major points are:

* The **TransformersReader** will sometimes predict the same span twice while duplicates are removed in the **FARMReader**
* The **FARMReader** currently uses the tokenizers from the Transformers library while the **TransformersReader** uses the tokenizers from the Tokenizers library
* Start and end logits are normalized per passage and multiplied in the **TransformersReader** while they are summed and not normalised in the **FARMReader**

If you're interested in the finer details of these points, have a look at `this <https://github.com/deepset-ai/haystack/issues/248#issuecomment-661977237>`_ GitHub comment.

We see value in maintaining both kinds of Readers since Transformers is a very familiar library to many of Haystack's users
but we at deepset can more easily update and optimise the FARM pipeline for speed and performance.

!! benchmarks !!

Haystack also has a close integration with FARM which means that you can further fine-tune your Readers on labelled data using a FARMReader.
See this tutorial (!!link!!) for an end-to-end example or below for a shortened example.

.. code-block:: python

    # Initialise Reader
    model = "deepset/roberta-base-squad2"
    reader = FARMReader(model)

    # Perform finetuning
    train_data = "PATH/TO_YOUR/TRAIN_DATA"
    train_filename = "train.json"
    save_dir = "finetuned_model"
    reader.train(train_data, train_filename, save_dir=save_dir)

    # Load
    finetuned_reader = FARMReader(save_dir)

Choosing the Right Model
------------------------

Any QA model that has been uploaded to the HuggingFace model hub can easily be used in Haystack,
simply by initializing with the Reader (FARM or Transformers) with its model hub name.

Currently, there are a lot of different models out there and it can be rather overwhelming trying to pick the one that fits your use case.
On top of the BERT based Readers, there are a slew of BERT variants (RoBERTa, ALBERT), distilled models (MiniLM, distilBERT) and a new generation transformers (ELECTRA).
Language models are standardly finetuned on the SQuAD dataset but there are other datasets available too like TriviaQA and Natural Questions.

!! Diagram of LMs !!

To get you started, we have a few recommendations for you to try out.

.. tabs::

    .. tab:: FARM

        .. tabs::

            .. tab:: RoBERTa (base)

                .. code-block:: python

                    reader = FARMReader("deepset/roberta-base-squad2")

            .. tab:: ELECTRA (base)

                .. code-block:: python

                    reader = FARMReader("deepset/electra-base-squad2")

            .. tab:: MiniLM

                .. code-block:: python

                    reader = FARMReader("deepset/minilm-uncased-squad2")

            .. tab:: ALBERT (XXL)

                    .. code-block:: python

                        reader = FARMReader("ahotrod/albert_xxlargev1_squad2_512")

    .. tab:: Transformers

        .. tabs::

            .. tab:: RoBERTa (base)

                .. code-block:: python

                    reader = TransformersReader("deepset/roberta-base-squad2")

            .. tab:: ELECTRA (base)

                .. code-block:: python

                    reader = TransformersReader("deepset/electra-base-squad2")

            .. tab:: MiniLM

                .. code-block:: python

                    reader = TransformersReader("deepset/minilm-uncased-squad2")

            .. tab:: ALBERT (XXL)

                .. code-block:: python

                    reader = TransformersReader("ahotrod/albert_xxlargev1_squad2_512")


In the class of base sized models trained on SQuAD, RoBERTa has shown better performance than BERT
and can be capably handled by any machine equipped with a single NVidia V100 GPU.
We recommend this as the starting point for anyone wanting to create a performant and computationally reasonable instance of Haystack.

If speed and GPU memory are more of a priority to you than accuracy,
you should try the MiniLM model.
It is a smaller model that is trained to mimic larger models through the distillation process,
and it outperforms the BERT base on SQuAD even though it is about 40% smaller.
In our tests we found that it was XX% faster than BERT and ~X% better in perfomance.
Compared to RoBERTa, it is only off by about X% absolute,

!! BENCHMARKS !!

!! See HF model card for performance? !!

For most, ALBERT XXL will be too large to feasibly work with.
But if performance is your sole concern, and you have the computational resources,
you might like to try ALBERT XXL which has set SoTA performance on SQuAD 2.0.

!! How good is it? How much computation resource do you need to run it? !!


Languages other than English
----------------------------

Haystack is also very well suited to doing open-domain QA on languages other than English.
While models are comparatively more performant on English,
thanks to a wealth of available English training data,
there are a couple QA models that are directly usable in Haystack and also worth mentioning.

.. tabs::

    .. tab:: FARM

        .. tabs::

            .. tab:: French

                .. code-block:: python

                    reader = FARMReader("illuin/camembert-base-fquad")

            .. tab:: Italian

                .. code-block:: python

                    reader = FARMReader("mrm8488/bert-italian-finedtuned-squadv1-it-alfa")

            .. tab:: Zero-shot

                .. code-block:: python

                    reader = FARMReader("deepset/xlm-roberta-large-squad2")

    .. tab:: Transformers

        .. tabs::

            .. tab:: French

                .. code-block:: python

                    reader = TransformersReader("illuin/camembert-base-fquad")

            .. tab:: Italian

                .. code-block:: python

                    reader = TransformersReader("mrm8488/bert-italian-finedtuned-squadv1-it-alfa")

            .. tab:: Zero-shot

                .. code-block:: python

                    reader = TransformersReader("deepset/xlm-roberta-large-squad2")

The French and Italian models are both monolingual langauge models trained on French and Italian versions of the SQuAD dataset
and their authors report decent results in their model cards here (!!link!!) and here (!!link!!).
Note that there is also a large variant of the French model available on the model hub.
There also exist Korean QA models on the model hub but their performance is not reported.

!! DPR IS ENGLISH SPECIFIC !!

The zero-shot model that is shown above is a multilingual XLM-RoBERTa Large that is trained on English SQuAD.
It is clear, from our evaluations, that the model has been able to transfer some of its English QA capabilities to other languages,
but still its performance lags behind that of the monolingual models (!!see model card for eval results!!).
Nonetheless, if there is not yet a monolingual model for your language and it is one of the 100 supported by XLM-RoBERTa,
this zero-shot model may serve as a decent first baseline.

If you are interested in the work around the world being done on bringing QA to other languages,
you should have a read of this blog article (!!link!!).

