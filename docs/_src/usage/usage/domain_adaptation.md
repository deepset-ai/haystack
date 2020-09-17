<!---
title: "Domain Adaptation"
metaTitle: "Domain Adaptation"
metaDescription: ""
slug: "/docs/domain_adaptation"
date: "2020-09-03"
id: "domain_adaptationmd"
--->

# Domain Adaptation

## Generalisation

In our experience, language models trained on SQuAD show very strong general question answering capabilities.
Though SQuAD is composed entirely of Wikipedia articles, these models are flexible enough to deal with many different styles of text.

Before trying to adapt these models to your domain, we’d recommend trying one of the off the shelf models.
We’ve found that these models are often flexible enough for a wide range of use cases.

**Intuition**: Most people probably don’t know what an HP Valve is.
But you don’t always need to know what a HP Valve is to answer “What is connected to a HP Valve?”
The answer might be there in plain language.
In the same way, many QA models have a good enough grasp of language to answer questions about concepts in an unseen domain.

## Finetuning

Any model that can be loaded into Haystack can also be finetuned within Haystack.
Simply provide the domain specific dataset and call `Reader.train()` on an initialised model.

```
reader.train(data_dir=train_data,
             train_filename="dev-v2.0.json",
             n_epochs=1,
             save_dir="my_model")
```

At the end of training, the finetuned model will be saved in the specified `save_dir` and can be loaded as a `Reader`.
See Tutorial 2 for a runnable example of this process.
If you’re interested in measuring how much your model has improved,
please also check out Tutorial 5 which walks through the steps needed to perform evaluation.

## Generating Labels

Using our [Haystack Annotate tool](https://annotate.deepset.ai/login) (Beta),
you can easily create a labelled dataset using your own documents featuring your own question/ answer pairs.



![image](./../../img/annotation_tool.png)

Features include:


* Structured workspaces via organisations, projects and users


* Easy upload of your own documents and labels in a variety of formats (txt, pdf, SQuAD style)


* Export of labels to be used directly in Haystack

Annotate also supports two different workflows:


* Think up questions and answers while reading passages (SQuAD style)


* Have a set of predefined questions and look for answers in the document (~ Natural Questions style)

## User Feedback

A simpler and faster process to retrain models is to utilise user feedback.
Users can give thumbs up or thumbs down to search results through the Rest API
and these labels will be stored in the `DocumentStore`
where they can already be used to retrain the `Reader`.
