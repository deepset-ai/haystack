<!---
title: "Frequently Asked Questions"
metaTitle: "Frequently Asked Questions"
metaDescription: ""
slug: "/docs/faq"
date: "2020-09-03"
id: "faqmd"
--->

#Frequently Asked Questions

##Why am I seeing duplicate answers being returned?

The ElasticsearchDocumentStore and MilvusDocumentStore rely on Elasticsearch and Milvus backend services which 
persist after your Python script has finished running.
If you rerun your script without deleting documents, you could end up with duplicate 
copies of your documents in your database.
The easiest way to avoid this is to call `DocumentStore.delete_documents()` after initialization
to ensure that you are working with an empty DocumentStore.

DocumentStores also have a `duplicate_documents` argument in their `__init__()` and `write_documents` methods
where you can define whether you'd like skip writing duplicates, overwrite existing duplicates or raise an error when there are duplicates.

##How can I make sure that my GPU is being engaged when I use Haystack?

You will want to ensure that a CUDA enabled GPU is being engaged when Haystack is running (you can check by running `nvidia-smi -l` on your command line).
Components which can be sped up by GPU have a `use_gpu` argument in their constructor which you will want to set to `True`.

##How do I speed up my predictions?

There are many different ways to speed up the performance of your Haystack system.

The Reader is usually the most computationally expensive component in a pipeline 
and you can often speed up your system by using a smaller model, like `deepset/minilm-uncased-squad2` (see [benchmarks](https://huggingface.co/deepset/minilm-uncased-squad2)). This usually comes with a small trade-off in accuracy.

You can reduce the work load on the Reader by instructing the Retriever to pass on less documents. 
This is done by setting the `top_k_retriever` parameter to a lower value.

Making sure that your documents are shorter can also increase the speed of your system. You can split
your documents into smaller chunks by using the `PreProcessor` (see [tutorial](https://haystack.deepset.ai/docs/v0.9.0/tutorial11md)).

For more optimization suggestions, have a look at our [optimization page](https://haystack.deepset.ai/docs/v0.9.0/optimizationmd)
and also our [blogs](https://medium.com/deepset-ai)

##How do I use Haystack for my language?

The components in Haystack, such as the `Retriever` or the `Reader`, are designed in a language agnostic way. However you may
have to set certain parameters or load models pretrained for your language in order to get good performance out of Haystack.
See our [languages page](https://haystack.deepset.ai/docs/v0.9.0/languagesmd) for more details.

##How can I add metadata to my documents so that I can apply filters?

When providing your documents in the input format (see [here](https://haystack.deepset.ai/docs/v0.9.0/documentstoremd#Input-Format))
you can provide metadata information as a dictionary under the `meta` key. At query time, you can provide a `filters` argument
(most likely through `Pipelines.run()`) that specifies the accepted values for a certain metadata field
(for an example of what a `filters` dictionary might look like, please refer to [this example](https://haystack.deepset.ai/docs/v0.9.0/apiretrievermd#__init__))

##How can I see predictions during evaluation?

To see predictions during evaluation, you want to initialize the `EvalDocuments` or `EvalAnswers` with `debug=True`. 
This causes their `EvalDocuments.log` or `EvalAnswers.log` to be populated with a record of each prediction made.

##How can I serve my Haystack model?

Haystack models can be wrapped in a REST API. For basic details on how to set this up, please refer to this section 
on our [Github page](https://github.com/deepset-ai/haystack/blob/master/README.md#7-rest-api). 
More comprehensive documentation coming soon!

##How can I interpret the confidence scores being returned by the Reader?

The confidence scores are in the range of 0 and 1 and reflect how confident the model is in each prediction that it makes.
Having a confidence score is particularly useful in cases where you need Haystack to work with a certain accuracy threshold.
Many of our users have built systems where predictions below a certain confidence value are routed on to a fallback system.

For more information on model confidence and how to tune it, please refer to [this section](https://haystack.deepset.ai/docs/v0.9.0/readermd#Confidence-Scores).

##My documents aren't showing up in my DocumentStore even though I've called `DocumentStore.write_documents()`

When indexing, retrieving or querying for documents from a DocumentStore, you can specify an `index` on which to perform this action. 
This can be specified in almost all methods of `DocumentStore` as well as `Retriever.retrieve()`.
Ensure that you are performing these operations on the one index! 
Note that this also applies at evaluation where labels are written into their own separate DocumentStore index.

##What is the difference between the FARMReader and the TransformersReader?

In short, the FARMReader using a QA pipeline implementation that comes from our own 
[FARM framework](https://github.com/deepset-ai/FARM) that we can more easily update and also optimize for performance. 
By contrast, the TransformersReader uses a QA pipeline implementation that comes from HuggingFace's [Transformers](https://github.com/huggingface/transformers).
See [this section](https://haystack.deepset.ai/docs/v0.9.0/readermd#Deeper-Dive-FARM-vs-Transformers) 
for a more details about their differences!