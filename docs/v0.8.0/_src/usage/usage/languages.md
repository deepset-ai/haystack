<!---
title: "Languages Other Than English"
metaTitle: "Languages Other Than English"
metaDescription: ""
slug: "/docs/languages"
date: "2020-11-05"
id: "languagesmd"
--->

# Languages Other Than English

Haystack is well suited to open-domain QA on languages other than English.
While our defaults are tuned for English,
you will find some tips and tricks here for using Haystack in your language. 

##Retrievers

The sparse retriever methods themselves(BM25, TF-IDF) are language agnostic.
Their only requirement is that the text be split into words.
The ElasticsearchDocumentStore relies on an analyzer to impose word boundaries,
but also to handle punctuation, casing and stop words.

The default analyzer is an English analyzer. 
While it can still work decently for a large range of languages,
you will want to set it to your language's analyzer for optimal performance.
In some cases, such as with Thai, the default analyzer is completely incompatible.
See [this page](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html) 
for the full list of language specific analyzers.

```python
document_store = ElasticsearchDocumentStore(analyzer="thai")
```

The models used in dense retrievers are language specific. 
Be sure to check language of the model used in your EmbeddingRetriever. 
The default model that is loaded in the DensePassageRetriever is for English.
We are currently working on training a German DensePassageRetriever model and know other teams who work on further languages.
If you have a language model and a question answering dataset in your own language, you can also train a DPR model using Haystack!
Below is a simplified example.
See the [API reference](/docs/latest/apiretrievermd#train) for `DensePassageRetriever.train()` for more details.

```python
dense_passage_retriever.train(self,
                              data_dir: str,
                              train_filename: str,
                              dev_filename: str = None,
                              test_filename: str = None,
                              batch_size: int = 16,
                              embed_title: bool = True,
                              num_hard_negatives: int = 1,
                              n_epochs: int = 3)
```

##Readers

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
[here](https://huggingface.co/illuin/camembert-base-fquad) and [here](https://huggingface.co/mrm8488/bert-italian-finedtuned-squadv1-it-alfa).
There also exist Korean QA models on the model hub but their performance is not reported.

The **zero-shot model** that is shown above is a **multilingual XLM-RoBERTa Large** that is trained on English SQuAD.
It is clear, from our [evaluations](https://huggingface.co/deepset/xlm-roberta-large-squad2#model_card),
that the model has been able to transfer some of its English QA capabilities to other languages,
but still its performance lags behind that of the monolingual models.
Nonetheless, if there is not yet a monolingual model for your language and it is one of the 100 supported by XLM-RoBERTa,
this zero-shot model may serve as a decent first baseline.

[//]: # (Add link to Reader training, create section in reader.md on training Reader)

