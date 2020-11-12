<!---
title: "Generator"
metaTitle: "Generator"
metaDescription: ""
slug: "/docs/generator"
date: "2020-11-05"
id: "generatormd"
--->

# Generator

While extractive QA highlights the span of text that answers a query,
generative QA can return a novel text answer that it has composed.
The best current approaches, such as [Retriever-Augmented Generation](https://arxiv.org/abs/2005.11401),
can draw upon both the knowledge it gained during language model pretraining (parametric)




In this tutorial, you will learn how to set up a generative system using the
[RAG model](https://arxiv.org/abs/2005.11401) which conditions the
answer generator on a set of retrieved documents.

We explore RAG models which use the input sequence to retrieve text passages and use these passages as additional context when generating the target sequencey


More smooth answers
answers that draw upon multiple documents
RAG diff tokens conditioned on different docs
potential for synthesis / reasoning

Draw more directly on knowledge stored within the language model

Failure more dangerous
silent failure
No way to verify that what it generated is based on accurate understanding of text
maybe influenced by biased text it read from doc store, or from lm pretraining
 