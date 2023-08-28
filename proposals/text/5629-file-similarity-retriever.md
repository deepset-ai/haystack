- Title: Adding FileSimilarityRetriever to haystack
- Decision driver: @elundaeva
- Start Date: 2023-08-28
- Proposal PR: https://github.com/deepset-ai/haystack/pull/5629
- Github Issue or Discussion: some links available in the "Detailed design" section below

# Summary

The retriever takes a file ID as query, searches for all documents from that file in the doc store and then performs one query for these documents, finding similar files for each.

# Basic example

The FileSimilarityRetriever would be instantiated as follows:

''' python

	  retriever = FileSimilarityRetriever(
	      document_store = ElasticSearchDocumentStore,
	      primary_retriever = EmbeddingRetriever, # defined separately, see full pipeline example below
          top_k=30,
	      file_aggregation_key = "file_id",
          max_num_queries = 50
	  )
'''

And here is an example of how the node would work in the context of a full pipeline:

''' yaml

version: '1.19.0'
name: 'FileSim'

components:
  - name: DocumentStore
    type: DeepsetCloudDocumentStore
    params:
      similarity: cosine
      embedding_dim: 768
      return_embedding: false

  - name: EmbeddingRetriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      embedding_model: sentence-transformers/all-mpnet-base-v2
      max_seq_len: 400
      batch_size: 32
      model_format: sentence_transformers
      top_k: 50
  - name: TextConverter # Converts files into documents
    type: TextConverter
  - name: FileSimilarityRetriever
    type: FileSimilarityRetriever
    params:
      document_store: DocumentStore
      primary_retriever: EmbeddingRetriever
      top_k: 30
      file_aggregation_key: file_id
      max_num_queries: 50 # changed 2022-11-14

pipelines:
  - name: query
    nodes:
      - name: FileSimilarityRetriever
        inputs: [Query]

  - name: indexing
    nodes:
      - name: TextConverter
        inputs: [File]

      - name: EmbeddingRetriever
        inputs: [TextConverter]

      - name: DocumentStore
        inputs: [EmbeddingRetriever]

'''

# Motivation

Initially this retriever was implemented by Mathis for a customer case to quickly retrieve documents similar to a given file. The reason we would like to add it to Haystack is because we see wider use for this node in future customer and community cases.

One example where this could be useful is in academic writing, if you have a large number of scientific sources (journals, books, conference proceedings) stored and you'd like to find all that are similar to a specific article or to what you've written so far (in the latter case, it can also be helpful for plagiarism detection). And as for industry use cases, report writing in any field can be facilitated by file similarity retrieval. It can also be very helpful for lawyers preparing a case as well as journalists doing research for their articles.

# Detailed design

You can see the code for this proposal here: https://github.com/deepset-ai/haystack/pull/xxxx/files.

As a general description, the FileSimilarityRetriever works by:
1. Getting all documents corresponding to the provided file ID from the document store. The documents can be obtained together with their embeddings, to save resources by avoiding re-calculating them at query time.
2. Retrieving similar docs to each document from the file, using one or two chosen retrieval methods.
3.  a) Returning the top-k retrieved similar documents, if only one retrieval method was used.
    b) Returning an aggregated list of similar documents retrieved by both retrievers, if two retrieval methods were chosen. The results are aggregated based on the reciprocal rank fusion score, though it is also possible to keep the original score (if one of the retrievers assigned the document a relevance score) in the metadata.
    Reciprocal rank fusion is calculated as 1 / (k + index)
    k is set to 61 in reciprocal rank fusion based on a University of Waterloo paper (co-authored with Google) called "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" [https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf] where k=60 was suggested, and 1 was added as python lists are 0-based and the paper used 1-based ranking.


FileSimilarityRetriever has the following parameters:
    document_store: The document store that the retriever should retrieve from.
    file_aggregation_key: The meta data key that should be used to aggregate documents to the file level.
    primary_retriever: A clutch until haystack supports passing a list of retrievers.
    secondary_retriever: A clutch until haystack supports passing a list of retrievers.
    keep_original_score: Set this to store the original score of the returned document in the document's meta field. The document's score property will be replaced with the reciprocal rank fusion score.
    top_k: How many documents to return.
    max_query_len: How many chars can be in a query document. The document will be cut off if it is longer
            than the maximum length. We need this here because there might be an issue with queries that are too long
            and the BM25Retriever because an error will be thrown if the query excees the `max_clause_count` search
            setting (https://www.elastic.co/guide/en/elasticsearch/reference/7.17/search-settings.html)
    max_num_queries: The maximum number of queries that should be run for a single file. If the number of
            query documents exceeds this limit, the query documents will be split into n parts so that
            n < max_num_queries and every nth document will be kept.
    use_existing_embeddings: Whether to re-use the existing embeddings from the index.
            To optimize speed for the file similarity retrieval you should set this parameter to `True`.
            This way the FileSimilarityRetriever can run on the CPU.

# Drawbacks

Look at the feature from the other side: what are the reasons why we should _not_ work on it? Consider the following:

- What's the implementation cost, both in terms of code size and complexity?
- Can the solution you're proposing be implemented as a separate package, outside of Haystack?
- Does it teach people more about Haystack?
- How does this feature integrate with other existing and planned features?
- What's the cost of migrating existing Haystack pipelines (is it a breaking change?)?

There are tradeoffs to choosing any path. Attempt to identify them here.

# Alternatives

What other designs have you considered? What's the impact of not adding this feature?

# Adoption strategy

If we implement this proposal, how will the existing Haystack users adopt it? Is
this a breaking change? Can we write a migration script?

# How we teach this

Would implementing this feature mean the documentation must be re-organized
or updated? Does it change how Haystack is taught to new developers at any level?

How should this feature be taught to the existing Haystack users (for example with a page in the docs,
a tutorial, ...).

# Unresolved questions

Not many unresolved questions, just need to adopt the retriever so it inherits from the BaseRetriever and not BaseComponent. Plus looks like I'll need to add the "retrieve_batch" method as currently it only has "retrieve".
