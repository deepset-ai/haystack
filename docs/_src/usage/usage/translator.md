<!---
title: "Translator"
metaTitle: "Translator"
metaDescription: ""
slug: "/docs/translator"
date: "2021-02-10"
id: "translatormd"
--->

# Translator

Texts come in different languages. This is not different for search and there are plenty of options to deal with it. 
One of them is actually to translate the incoming query, the documents or the search results. 

Let's imagine you have an English corpus of technical docs, but the mother tongue of many of your users is French. 
You can use a Translator node in your pipeline to
1. Translate the incoming query from French to English
2. Search in your English corpus for the right document / answer
3. Translate the results back from English to French

<div class="recommendation">

**Example (Stand-alone Translator)**

You can use the Translator component directly to translate your query or document(s): 
```python
DOCS = [
        Document(
            text="""Heinz von Foerster was an Austrian American scientist 
                  combining physics and philosophy, and widely attributed 
                  as the originator of Second-order cybernetics."""
        )
    ]
translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-fr")
res = translator.translate(documents=DOCS, query=None)
```

**Example (Wrapping another Pipeline)**

You can also wrap one of your existing pipelines and "add" the translation nodes at the beginning and at the end of your pipeline.
For example, lets translate the incoming query to from French to English, then do our document retrieval and then translate the results back from English to French:

```python
from haystack.pipeline import TranslationWrapperPipeline, DocumentSearchPipeline
from haystack.translator import TransformersTranslator

pipeline = DocumentSearchPipeline(retriever=my_dpr_retriever)

in_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-fr-en")
out_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-fr")

pipeline_with_translation = TranslationWrapperPipeline(input_translator=in_translator,
                                                       output_translator=out_translator,
                                                       pipeline=pipeline)
```


</div>
