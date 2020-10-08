<!---
title: "Reader"
metaTitle: "Reader"
metaDescription: ""
slug: "/docs/reader"
date: "2020-09-03"
id: "readermd"
--->

# Reader

The Reader, also known as Open-Domain QA systems in Machine Learning speak,
is the core component that enables Haystack to find the answers that you need.
Haystack’s Readers are:


* built on the latest transformer based language models


* strong in their grasp of semantics


* sensitive to syntactic structure


* state-of-the-art in QA tasks like SQuAD and Natural Questions

<div class="tabs tabsreaderreader">

<div class="tab">
<input type="radio" id="tab-0-1" name="tab-group-0" checked>
<label class="labelouter" for="tab-0-1">FARM</label>
<div class="tabcontent">

```python
model = "deepset/roberta-base-squad2"
reader = FARMReader(model, use_gpu=True)
finder = Finder(reader, retriever)
```

</div> 
</div>

<div class="tab">
<input type="radio" id="tab-0-2" name="tab-group-0">
<label class="labelouter" for="tab-0-2">Transformers</label>
<div class="tabcontent">

```python
model = "deepset/roberta-base-squad2"
reader = TransformersReader(model, use_gpu=1)
finder = Finder(reader, retriever)
```

</div> 
</div>

</div>

While these models can work on CPU, it is recommended that they are run using GPUs to keep query times low.

## Choosing the Right Model

In Haystack, you can start using pretrained QA models simply by providing its HuggingFace Model Hub name to the Reader.
The loading of model weights is handled by Haystack,
and you have the option of using the QA pipeline from deepset FARM or HuggingFace Transformers (see FARM vs Transformers for details).

Currently, there are a lot of different models out there and it can be rather overwhelming trying to pick the one that fits your use case.
To get you started, we have a few recommendations for you to try out.

<div class="tabs tabsreader">

<div class="tab">
<input type="radio" id="tab-1" name="tab-group-1" checked>
<label class="labelouter" for="tab-1">FARM</label>
<div class="tabcontent">

<div class="tabs innertabs">

<div class="tab">
<input type="radio" id="tab-1-1" name="tab-group-2" checked>
<label class="labelinner" for="tab-1-1">RoBERTa (base)</label>
<div class="tabcontentinner">

**An optimised variant of BERT and a great starting point.**

```python
reader = FARMReader("deepset/roberta-base-squad2")
```

* **Pro**: Strong all round model

* **Con**: There are other models that are either faster or more accurate

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-2" name="tab-group-2">
<label class="labelinner" for="tab-1-2">MiniLM</label>
<div class="tabcontentinner">

**A cleverly distilled model that sacrifices a little accuracy for speed.**

```python
reader = FARMReader("deepset/minilm-uncased-squad2")
```

* **Pro**: Inference speed up to 50% faster than BERT base

* **Con**: Still doesn’t match the best base sized models in accuracy

</div>
</div>

<div class="tab">
<input type="radio" id="tab-1-3" name="tab-group-2">
<label class="labelinner" for="tab-1-3">ALBERT (XXL)</label>
<div class="tabcontentinner">

**Large, powerful, SotA model.**

```python
reader = FARMReader("ahotrod/albert_xxlargev1_squad2_512")
```

* **Pro**: Better accuracy than any other open source model in QA

* **Con**: The computational power needed make it impractical for most use cases

</div>
</div>

</div>

</div> 
</div>

<div class="tab">
<input type="radio" id="tab-2" name="tab-group-1">
<label class="labelouter" for="tab-2">Transformers</label>
<div class="tabcontent">

<div class="tabs innertabs">

<div class="tab">
<input type="radio" id="tab-2-1" name="tab-group-3" checked>
<label class="labelinner" for="tab-2-1">RoBERTa (base)</label>
<div class="tabcontentinner">

**An optimised variant of BERT and a great starting point.**

```python
reader = TransformersReader("deepset/roberta-base-squad2")
```

* **Pro**: Strong all round model

* **Con**: There are other models that are either faster or more accurate

</div>
</div>

<div class="tab">
<input type="radio" id="tab-2-2" name="tab-group-3">
<label class="labelinner" for="tab-2-2">MiniLM</label>
<div class="tabcontentinner">

**A cleverly distilled model that sacrifices a little accuracy for speed.**

```python
reader = TransformersReader("deepset/minilm-uncased-squad2")
```

* **Pro**: Inference speed up to 50% faster than BERT base

* **Con**: Still doesn’t match the best base sized models in accuracy

</div>
</div>

<div class="tab">
<input type="radio" id="tab-2-3" name="tab-group-3">
<label class="labelinner" for="tab-2-3">ALBERT (XXL)</label>
<div class="tabcontentinner">

**Large, powerful, SotA model.**

```python
reader = TransformersReader("ahotrod/albert_xxlargev1_squad2_512")
```

* **Pro**: Better accuracy than any other open source model in QA

* **Con**: The computational power needed make it impractical for most use cases

</div>
</div>

</div>

</div> 
</div>

</div>

**All-rounder**: In the class of base sized models trained on SQuAD, **RoBERTa** has shown better performance than BERT
and can be capably handled by any machine equipped with a single NVidia V100 GPU.
We recommend this as the starting point for anyone wanting to create a performant and computationally reasonable instance of Haystack.

**Built for Speed**: If speed and GPU memory are more of a priority to you than accuracy,
you should try the MiniLM model.
It is a smaller model that is trained to mimic larger models through the distillation process,
and it outperforms the BERT base on SQuAD even though it is about 40% smaller.

<!-- _comment: !! In our tests we found that it was XX% faster than BERT and ~X% better in perfomance. Compared to RoBERTa, it is only off by about X% absolute, -->
**State of the Art Accuracy**: For most, **ALBERT XXL** will be too large to feasibly work with.
But if performance is your sole concern, and you have the computational resources,
you might like to try ALBERT XXL which has set SoTA performance on SQuAD 2.0.

<!-- _comment: !! How good is it? How much computation resource do you need to run it? !! -->
## Languages other than English

Haystack is also very well suited to open-domain QA on languages other than English.
While models are comparatively more performant on English,
thanks to a wealth of available English training data,
there are a couple QA models that are directly usable in Haystack.

<div class="tabs tabsreaderlanguage">

<div class="tab">
<input type="radio" id="tab-4-1" name="tab-group-4" checked>
<label class="labelouter" for="tab-4-1">FARM</label>
<div class="tabcontent">

<div class="tabs innertabslanguage">

<div class="tabinner">
<input type="radio" id="tab-5-1" name="tab-group-5" checked>
<label class="labelinner" for="tab-5-1">French</label>
<div class="tabcontentinner">

```python
reader = FARMReader("illuin/camembert-base-fquad")
```

</div>
</div>

<div class="tabinner">
<input type="radio" id="tab-5-2" name="tab-group-5">
<label class="labelinner" for="tab-5-2">Italian</label>
<div class="tabcontentinner">

```python
reader = FARMReader("mrm8488/bert-italian-finedtuned-squadv1-it-alfa")
```

</div>
</div>

<div class="tabinner">
<input type="radio" id="tab-5-3" name="tab-group-5">
<label class="labelinner" for="tab-5-3">Zero-shot</label>
<div class="tabcontentinner">

```python
reader = FARMReader("deepset/xlm-roberta-large-squad2")
```

</div>
</div>

</div>

</div> 
</div>

<div class="tab">
<input type="radio" id="tab-4-2" name="tab-group-4">
<label class="labelouter" for="tab-4-2">Transformers</label>
<div class="tabcontent">

<div class="tabs innertabslanguage">

<div class="tabinner2">
<input type="radio" id="tab-6-1" name="tab-group-6" checked>
<label class="labelinner" for="tab-6-1">French</label>
<div class="tabcontentinner">

```python
reader = TransformersReader("illuin/camembert-base-fquad")
```

</div>
</div>

<div class="tabinner2">
<input type="radio" id="tab-6-2" name="tab-group-6">
<label class="labelinner" for="tab-6-2">Italian</label>
<div class="tabcontentinner">

```python
reader = TransformersReader("mrm8488/bert-italian-finedtuned-squadv1-it-alfa")
```

</div>
</div>

<div class="tabinner2">
<input type="radio" id="tab-6-3" name="tab-group-6">
<label class="labelinner" for="tab-6-3">Zero-shot</label>
<div class="tabcontentinner">

```python
reader = TransformersReader("deepset/xlm-roberta-large-squad2")
```

</div>
</div>

</div>

</div> 
</div>

</div>

The **French** and **Italian models** are both monolingual language models trained on French and Italian versions of the SQuAD dataset
and their authors report decent results in their model cards
[here](https://huggingface.co/illuin/camembert-base-fquad) and [here](https://huggingface.co/mrm8488/bert-italian-finedtuned-squadv1-it-alfa) .
Note that there is also a [large variant](https://huggingface.co/illuin/camembert-large-fquad) of the French model available on the model hub.
There also exist Korean QA models on the model hub but their performance is not reported.

The **zero-shot model** that is shown above is a **multilingual XLM-RoBERTa Large** that is trained on English SQuAD.
It is clear, from our [evaluations](https://huggingface.co/deepset/xlm-roberta-large-squad2#model_card),
that the model has been able to transfer some of its English QA capabilities to other languages,
but still its performance lags behind that of the monolingual models.
Nonetheless, if there is not yet a monolingual model for your language and it is one of the 100 supported by XLM-RoBERTa,
this zero-shot model may serve as a decent first baseline.

When using a Reader of any language, it’s important to ensure that the Retriever is also compatible.
While sparse methods like BM25 and TF-IDF are language agnostic,
dense method like Dense Passage Retrieval are trained for a particular language.

<!-- farm-vs-trans: -->
## Deeper Dive: FARM vs Transformers

Apart from the **model weights**, Haystack Readers contain all the components found in end-to-end open domain QA systems.
This includes **tokenization**, **embedding computation**, **span prediction** and **candidate aggregation**.
While the handling of model weights is the same between the FARM and Transformers libraries, their QA pipelines differ in some ways.
The major points are:


* The **TransformersReader** will sometimes predict the same span twice while duplicates are removed in the **FARMReader**


* The **FARMReader** currently uses the tokenizers from the HuggingFace Transformers library while the **TransformersReader** uses the tokenizers from the HuggingFace Tokenizers library


* Start and end logits are normalized per passage and multiplied in the **TransformersReader** while they are summed and not normalised in the **FARMReader**

If you’re interested in the finer details of these points, have a look at [this](https://github.com/deepset-ai/haystack/issues/248#issuecomment-661977237) GitHub comment.

We see value in maintaining both kinds of Readers since Transformers is a very familiar library to many of Haystack’s users
but we at deepset can more easily update and optimise the FARM pipeline for speed and performance.

<!-- _comment: !! benchmarks !! -->
Haystack also has a close integration with FARM which means that you can further fine-tune your Readers on labelled data using a FARMReader.
See our tutorials for an end-to-end example or below for a shortened example.

```python
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
```

## Deeper Dive: From Language Model to Haystack Reader

Language models form the core of most modern NLP systems and that includes the Readers in Haystack.
They build a general understanding of language when performing training tasks such as Masked Language Modeling or Replaced Token Detection
on large amounts of text.
Well trained language models capture the word distribution in one or more languages
but more importantly, convert input text into a set of word vectors that capture elements of syntax and semantics.

In order to convert a language model into a Reader model, it needs first to be trained on a Question Answering dataset.
To do so requires the addition of a question answering prediction head on top of the language model.
The task can be thought of as a token classification task where every input token is assigned a probability of being
either the start or end token of the correct answer.
In cases where the answer is not contained within the passage, the prediction head is also expected to return a `no_answer` prediction.

<!-- _comment: !! Diagram of language model / prediction head !! -->
Since language models are limited in the number of tokens which they can process in a single forward pass,
a sliding window mechanism is implemented to handle variable length documents.
This functions by slicing the document into overlapping passages of (approximately) `max_seq_length`
that are each offset by `doc_stride` number of tokens.
These can be set when the Reader is initialized.

<div class="tabs tabsreaderdeep">

<div class="tab">
<input type="radio" id="tab-7-1" name="tab-group-7" checked>
<label class="labelouter" for="tab-7-1">FARM</label>
<div class="tabcontent">

```python
reader = FARMReader(... max_seq_len=384, doc_stride=128 ...)
```

</div> 
</div>

<div class="tab">
<input type="radio" id="tab-7-2" name="tab-group-7">
<label class="labelouter" for="tab-7-2">Transformers</label>
<div class="tabcontent">

```python
reader = TransformersReader(... max_seq_len=384, doc_stride=128 ...
```

</div> 
</div>

</div>

Predictions are made on each individual passage and the process of aggregation picks the best candidates across all passages.
If you’d like to learn more about what is happening behind the scenes, have a look at [this](https://medium.com/deepset-ai/modern-question-answering-systems-explained-4d0913744097) article.

<!-- _comment: !! Diagram from Blog !! -->
