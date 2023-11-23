- Title: Addition of a RecentnessRanker node
- Decision driver: @elundaeva
- Start Date: 2023-07-05
- Proposal PR: https://github.com/deepset-ai/haystack/pull/5289
- Github Issue or Discussion: some links available in the "Detailed design" section below

# Summary

This ranker allows to have retrieved documents sorted not only by relevance (default) but also with recency factored in.

# Basic example

The RecentnessRanker would be instantiated as follows:

  ``` python

	  ranker = RecentnessRanker(
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
      inputs: [Ranker]
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

Initially this ranker was implemented by Timo for a customer case where the date of the document mattered for retrieval. The reason we would like to add it to Haystack is because we see wider use for this node in future customer and community cases. One example where document recency matters is in a QA solution based on technical documentation with release notes of a software product - the older release notes should naturally have less priority in the responses than the most recent ones. And another example is news content - news articles retrieval can definitely benefit from recency being factored into the relevance calculation.

# Detailed design

You can see the code for this proposal here: https://github.com/deepset-ai/haystack/pull/5301/files.

As a general description, the ranker has the following parameters (date_identifier and method are required, the rest are optional):
- date_identifier (string pointing to the date field in the metadata)
- weight (the options are:
          - 0.5 default, relevance and recency will have the same impact in the calculation;
          - 0 only relevance will be considered for the calculation, so the RecentnessRanker is effectively disabled;
          - 1 only recency will be considered for the calculation)
- top_k (number of documents to return, works the same way as top-k in other rankers as well as retrievers)
- method (the options are:
          - "reciprocal_rank_fusion" which does not require any relevance score from the previous node;
          - "score" requires a 0-1 relevance score provided from the previous node in the pipeline.
          More information on method compatibility with different retrievers is in the Drawbacks section below)

The RecentnessRanker works by:
1. Adjusting the relevance score based on the chosen weight.
  For the "reciprocal_rank_fusion" the calculation is rrf * (1 - weight). The rrf is calculated as 1 / (k + rank) where k=61 (see reasoning below).
  And the "score" method performs the calculation as relevance score * (1 - weight).
2. Adding to the relevance score the recentness score by:
  For the "reciprocal_rank_fusion" - performing the rrf * weight calculation on the documents dictionary sorted by date where rrf is 1 / (k + rank), k=61.
  For the "score" method - performing the recentness score * weight calculation where recentness score is (amount of documents - rank) / amount of documents.
3. Returning top-k documents in the documents dictionary sorted by final score (relevance score + recentness score both adjusted by weight).

k is set to 61 in reciprocal rank fusion based on a University of Waterloo paper (co-authored with Google) called "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" [https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf] where k=60 was suggested, and 1 was added as python lists are 0-based and the paper used 1-based ranking.

# Drawbacks

Since this is a relatively small change without any effect on existing nodes, I do not see major reasons not to add this ranker. The only important limitation to using this node is the need to have a metadata field with document date already present.

For the "score" method, you would also need to double-check that the previous node outputs a score within [0,1] range (e.g. CohereRanker, SentenceTransformersRanker, EmbeddingRetriever). With the "reciprocal_rank_fusion" method, you do not need to have the relevance score pre-calculated, so using this method allows to combine RecentnessRanker with other retrieval nodes, like BM25 retriever.

# Alternatives

Without adding this feature it will not be possible to handle customer and community cases where recency of documents matters for the response, see examples in the Motivation section.

# Adoption strategy

This is not a breaking change and there does not seem to be any need for a migration script. Existing Haystack users can just start using this node on as-needed basis in combination with existing retrieval options (sparse/dense/hybrid).

# How we teach this

A small change like this might not require creating a whole new tutorial (although it is of course up to you), although it can be interesting to discuss this ranker with example usage in blog post format like we have for metadata filtering (https://www.deepset.ai/blog/metadata-filtering-in-haystack).

As for documentation needs, it would be good to add some info on how to use this recentness ranker - it can be added to the same page where the other existing rankers are explained. If you need help writing the documentation and/or the blog post/tutorial, please do not hesitate to reach out to me.

# Unresolved questions

Since it has already been implemented and is functional, there are not many known unresolved design questions. We just need to make sure that if/when the custom node is deprecated and we transition to using this node in Haystack, there are no disruptions to production pipelines that have been using this node and they get adjusted accordingly.
