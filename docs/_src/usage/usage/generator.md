<!---
title: "Generator"
metaTitle: "Generator"
metaDescription: ""
slug: "/docs/generator"
date: "2020-11-05"
id: "generatormd"
--->

# Generator

<div class="recommendation">

**Example**

See [Tutorial 7](/docs/latest/tutorial7md) for a guide on how to build your own generative QA system.

</div>

While extractive QA highlights the span of text that answers a query,
generative QA can return a novel text answer that it has composed.
The best current approaches, such as [Retriever-Augmented Generation](https://arxiv.org/abs/2005.11401),
can draw upon both the knowledge it gained during language model pretraining (parametric memory)
as well as passages provided to it with a retriever (non-parametric memory).
With the advent of Transformer based retrieval methods such as [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906),
retriever and generator can be trained concurrently from the one loss signal.

Pros
* More appropriately phrased answers
* Able to syntehsize information from different texts
* Can draw on latent knowledge stored in language model

Cons
* Not easy to track what piece of information the generator is basing its response off of

