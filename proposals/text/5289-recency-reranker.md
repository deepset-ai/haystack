- Title: Addition of a 'RecencyReranker' node
- Decision driver: @elundaeva
- Start Date: 2023-07-05
- Proposal PR: https://github.com/deepset-ai/haystack/pull/5289
- Github Issue or Discussion: some links available in the "Detailed design" section below

# Summary

This reranker allows to have retrieved documents sorted not only by relevance (default) but also with recency factored in.

# Basic example

The RecentnessReranker would be instantiated as follows:

  ``` python

	  reranker = RecentnessReranker(
	      date_identifier="date",
	      weight="0.5",
          top_k=3,
	      method="reciprocal_rank_fusion",
	  )
  ```

And here is an example of how the node would work in the context of a generative QA pipeline:

  ``` yaml

version: '1.18.0'
name: 'Example pipeline'

components:
- name: DocumentStore
  type: DeepsetCloudDocumentStore
- name: EmbeddingRetriever
  type: EmbeddingRetriever
  params:
    document_store: DocumentStore
    embedding_model: [embedding model here]
    model_format: sentence_transformers
    top_k: 30
- name: BM25Retriever
  type: BM25Retriever
  params:
    document_store: DocumentStore
    top_k: 30
- name: JoinDocuments
  type: JoinDocuments
  params:
    top_k_join: 30
    join_mode: reciprocal_rank_fusion
- name: Ranker
  type: SentenceTransformersRanker
  params:
    model_name_or_path: [cross-encoder model here]
    top_k: 15
- name: RecentnessRanker
  type: RecentnessRanker
  params:
    date_identifier: release_date
    top_k: 3
    method: score
- name: qa_template
  type: PromptTemplate
  params:
    output_parser:
        type: AnswerParser
    prompt: "prompt text here"
- name: PromptNode
  type: PromptNode
  params:
    default_prompt_template: qa_template
    max_length: 300
    model_kwargs:
      temperature: 0
    model_name_or_path: gpt-3.5-turbo
- name: FileTypeClassifier
  type: FileTypeClassifier
- name: TextConverter
  type: TextConverter
- name: PDFConverter
  type: PDFToTextConverter
- name: Preprocessor
  params:
    language: en
    split_by: word
    split_length: 200
    split_overlap: 10
    split_respect_sentence_boundary: true
  type: PreProcessor

pipelines:
- name: query
  nodes:
    - name: EmbeddingRetriever
      inputs: [Query]
    - name: BM25Retriever
      inputs: [Query]
    - name: JoinDocuments
      inputs: [EmbeddingRetriever, BM25Retriever]
    - name: Ranker
      inputs: [JoinDocuments]
    - name: RecentnessRanker
      inputs: [Reranker]
    - name: PromptNode
      inputs: [RecentnessRanker]

- name: indexing
  nodes:
  - inputs:
    - File
    name: FileTypeClassifier
  - inputs:
    - FileTypeClassifier.output_1
    name: TextConverter
  - inputs:
    - FileTypeClassifier.output_2
    name: PDFConverter
  - inputs:
    - TextConverter
    - PDFConverter
    name: Preprocessor
  - inputs:
    - Preprocessor
    name: EmbeddingRetriever
  - inputs:
    - EmbeddingRetriever
    name: DocumentStore

  ```

# Motivation

Initially this reranker was implemented by Timo for a customer case where the date of the document mattered for retrieval. The reason we would like to add it to Haystack is because we see wider use for this node in future customer and community cases. One example where document recency matters is in a QA solution based on technical documentation with release notes of a software product - the older release notes should naturally have less priority in the responses than the most recent ones. And another example is news content - news articles retrieval can definitely benefit from recency being factored into the relevance calculation.

# Detailed design

The reranker has already been implemented by Timo here: https://github.com/deepset-ai/deepset-cloud-custom-nodes/blob/main/deepset_cloud_custom_nodes/rankers/recentness_reranker.py & PR was reviewed at the time by Seb and Florian: https://github.com/deepset-ai/deepset-cloud-custom-nodes/pull/54.

Additionally, you can see the code for this proposal here: https://github.com/deepset-ai/haystack/pull/5301/files. It is the same code as above, just with small naming changes (e.g. "rff" method got changed to "reciprocal_rank_fusion" to match the existing naming used Haystack's JoinDocuments node)

As a general description, the reranker works by sorting the documents based on date and modifying the relevance score calculation slightly to include recency-based weight.

# Drawbacks

Since this is a relatively small change without any effect on existing nodes, I do not see major reasons not to add this reranker. The only important limitation to using this node is the need to have a metadata field with document date already present and for the "score" method the need to double-check that the previous node (e.g. CohereRanker, SentenceTransformersRanker, EmbeddingRetriever) outputs a score within [0,1] range.

# Alternatives

Without adding this feature it will not be possible to handle customer and community cases where recency of documents matters for the response, see examples in the Motivation section.

# Adoption strategy

This is not a breaking change and there does not seem to be any need for a migration script. Existing Haystack users can just start using this node on as-needed basis in combination with existing retrieval options (sparse/dense/hybrid).

# How we teach this

A small change like this might not require creating a whole new tutorial (although it is of course up to you), although it can be interesting to discuss this reranker with example usage in blog post format like we have for metadata filtering (https://www.deepset.ai/blog/metadata-filtering-in-haystack).

As for documentation needs, it would be good to add some info on how to use this recency reranker - it can be added to the same page where the other existing rerankers are explained. If you need help writing the documentation and/or the blog post/tutorial, please do not hesitate to reach out to me.

# Unresolved questions

Since it has already been implemented and is functional, there are not many known unresolved design questions. We just need to make sure that if/when the custom node is deprecated and we transition to using this node in Haystack, there are no disruptions to production pipelines that have been using this node and they get adjusted accordingly.
