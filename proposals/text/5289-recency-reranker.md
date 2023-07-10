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

# Motivation

Initially this reranker was implemented by Timo for a customer case where the date of the document mattered for retrieval. The reason we would like to add it to Haystack is because we see wider use for this node in future customer cases. One example where document recency matters is in a QA solution based on technical documentation with release notes of a software product - the older release notes should naturally have less priority in the responses than the most recent ones. 

# Detailed design

The reranker has already been implemented by Timo here: https://github.com/deepset-ai/deepset-cloud-custom-nodes/blob/main/deepset_cloud_custom_nodes/rankers/recentness_reranker.py & PR was reviewed at the time by Seb and Florian: https://github.com/deepset-ai/deepset-cloud-custom-nodes/pull/54. 

Additionally, you can see the code for this proposal here: https://github.com/deepset-ai/haystack/pull/5301/files. It is the same code as above, just with small naming changes (e.g. "rff" method got changed to "reciprocal_rank_fusion" to match the existing naming used Haystack's JoinDocuments node)

As a general description, the reranker works by sorting the documents based on date and modifying the relevance score calculation slightly to include recency-based weight.

# Drawbacks

Since this is a relatively small change without any effect on existing nodes, I do not see major reasons not to add this reranker. The only important limitation to using this node is the need to have a metadata field with document date already present and for the "score" method the need to double-check that your retreiver node outputs a score within [0,1] range. 

# Alternatives

Without adding this feature it will not be possible to handle customer cases where recency of documents matters for the response, see examples in the Motivation section.

# Adoption strategy

This is not a breaking change and there does not seem to be any need for a migration script. Existing Haystack users can just start using this node on as-needed basis in combination with existing retrieval options (sparse/dense/hybrid).

# How we teach this

A small change like this might not require creating a whole new tutorial (although it is of course up to you), although it can be interesting to discuss this reranker with example usage in blog post format like we have for metadata filtering (https://www.deepset.ai/blog/metadata-filtering-in-haystack). 

As for documentation needs, it would be good to add some info on how to use this recency reranker - it can be added to the same page where the other existing rerankers are explained. If you need help writing the documentation and/or the blog post/tutorial, please do not hesitate to reach out to me.

# Unresolved questions

Since it has already been implemented and is functional, there are not many known unresolved design questions.