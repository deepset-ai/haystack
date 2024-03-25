- Title: Adding FileSimilarityRetriever to haystack
- Decision driver: @elundaeva
- Start Date: 2023-08-28
- Proposal PR: https://github.com/deepset-ai/haystack/pull/5629
- Github Issue or Discussion: some links available in the "Detailed design" section below

# Summary

The retriever takes a metadata aggregation key (e.g. "file_id") as query, searches for all documents from that file in the doc store and then performs a query for each document to find similar documents for each. Then these search results for each document are aggregated to produce a list of similar files.

# Basic example

The FileSimilarityRetriever would be instantiated as follows:

``` python

	  retriever = FileSimilarityRetriever(
	      document_store = ElasticSearchDocumentStore,
	      primary_retriever = EmbeddingRetriever, # defined separately, see full pipeline example below
        top_k=30,
	      file_aggregation_key = "file_id",
        max_num_queries = 50
	  )
```

And here is an example of how the node would work in the context of a full pipeline:

``` yaml

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
  - name: TextConverter
    type: TextConverter
  - name: FileSimilarityRetriever
    type: FileSimilarityRetriever
    params:
      document_store: DocumentStore
      primary_retriever: EmbeddingRetriever
      top_k: 30
      file_aggregation_key: file_id
      max_num_queries: 50

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

```

# Basic example - updated

When implemented as two sub-components (see the section **Detailed design - updated** below), this retriever would be instantiated as follows:

``` python

	  DocumentFetcher = DocumentFetcher(
	      document_store = ElasticSearchDocumentStore,
	      file_aggregation_key = "file_id",
        max_num_docs = 50
	  )

    >>> then one or two retrievers here >>>

    DocumentAggregator = DocumentAggregator(
        top_k = 5,
        file_aggregation_key = "file_id",
        output = "top_document", # This new param is explained in Detailed design section below
        keep_original_score = True
    )

```

It would also be possible to skip the DocumentFetcher and just add the DocumentAggregator on top of your retriever(s) to receive e.g. file names (if file_aggregation_key is set to "name" and output is set to "file_aggregation_key") as output instead of documents directly from the retriever.

# Motivation

Initially this retriever was implemented by Mathis for a customer case to quickly retrieve similar files given a file as input. The reason we would like to add it to Haystack is because we see wider use for this node in future customer and community cases.

One example where this could be useful is in academic writing, if you have a large number of scientific sources (journals, books, conference proceedings) stored and you'd like to find all that are similar to a specific article or to what you've written so far (in the latter case, it can also be helpful for plagiarism detection). And as for industry use cases, report writing in any field can be facilitated by file similarity retrieval. It can also be very helpful for lawyers preparing a case as well as journalists doing research for their articles.

# Detailed design

You can see the code for this proposal here: https://github.com/deepset-ai/haystack/pull/5666/files.

As a general description, the FileSimilarityRetriever works by:
1. Getting all documents corresponding to the provided file ID from the document store. The documents can be obtained together with their embeddings, to save resources by avoiding re-calculating them at query time.

2. Retrieving similar docs to each document from the file, using one or two chosen retrieval methods.

3.  Returning the retrieved similar documents (top-k from the BM25 and/or semantic retriever's definition). The results are then aggregated based on the reciprocal rank fusion score, though it is also possible to keep the original score (if one of the retrievers assigned the document a relevance score) in the metadata.
Reciprocal rank fusion is calculated as 1 / (k + index)
k is set to 61 in reciprocal rank fusion based on a University of Waterloo paper (co-authored with Google) called "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" [https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf] where k=60 was suggested, and 1 was added as python lists are 0-based and the paper used 1-based ranking.

4. Returning the final similar files (top-k from the FileSimilarityRetriever definition), based on an aggregated RRF score of all documents pertaining to the file. In the results the user sees the most relevant document from each file.
We can also provide the user a choice to have the values of the metadata key used for aggregation (e.g. file IDs) returned instead of the most relevant documents, e.g. via an additional optional param "output" set by the user to "top_document" or "file_aggregation_key", with "top_document" being default since it looks more similar to the output of all pipelines (as haystack pipelines usually return natural language responses).

The score aggregation approach is based on the paper "PARM: A Paragraph Aggregation Retrieval Model for Dense Document-to-Document Retrieval" [https://arxiv.org/pdf/2201.01614.pdf] by Althammer et al., with the exception that reciprocal rank fusion (RRF) is used instead of vector-based aggregation with reciprocal rank fusion weighting (VRRF) to make implementation more straightforward.

FileSimilarityRetriever has the following parameters:
- document_store: The document store that the retriever should retrieve from.
- file_aggregation_key: The meta data key that should be used to aggregate documents to the file level.
- primary_retriever: First retriever.
- secondary_retriever (optional): Second retriever (if applicable).
- keep_original_score (optional): Set this to store the original relevance score of the returned document in the document's meta field (the documents' scores get replaced in the FileSimilarityRetriever output with the reciprocal rank fusion score).
- top_k: How many documents to return.
- max_query_len: How many characters can be in a query document. The documents exceeding this limit will be cut off. This was added because in some cases BM25Retriever threw an error if the query exceeded the `max_clause_count` search setting (https://www.elastic.co/guide/en/elasticsearch/reference/7.17/search-settings.html)
- max_num_queries (optional): The maximum number of queries that should be run for a single file. If the number of query documents exceeds this limit, the query documents will be split into n parts so that n < max_num_queries and every nth document will be kept.
- use_existing_embeddings: Whether to re-use the existing embeddings from the index. To optimize speed for the file similarity retrieval you should set this parameter to `True`. This way the FileSimilarityRetriever can run on the CPU.

# Detailed design - updated

During the proposal discussion, it was decided to break up the initial FileSimilarityRetriever into two sub-components in order to simplify the design and testing and also to make it possible to use only one of the sub-components, if needed, thus covering more use cases.

The new design would go as follows:
- DocumentFetcher that gets all docs corresponding to a file based on a provided aggregation meta key and returns a list of docs to be used by one or two retrievers as input for batch retrieval.
- DocumentAggregator placed after the retriever(s) that aggregates the document list(s) using RRF into a single ranked list of files, which are put together based on file_ids of retrieved docs.

These sub-components will have the following params (for full explanation see Detailed design section above):
- DocumentFetcher
  - document_store
  - file_aggregation_key
  - max_num_docs: optional, in the original version it was called max_num_queries
  Note: When researching how to change the params from the original implementation I found out there is no longer need for max_query_len that was required in the original FileSimilarityRetriever. It was implemented to not overshoot the max_clause_count limit in ElasticSearch settings but this got deprecated as of vs 8.0.0 according to their documentation - https://www.elastic.co/guide/en/elasticsearch/reference/current/search-settings.html

- DocumentAggregator
  - documents: can be a list or a list of list of docs
  - file_aggregation_key: default would be "file_id", but can be changed to e.g. "name"
  - output: default "top_document", but can also be "file_aggregation_key"
  - keep_original_score
  - top_k = 5

# Drawbacks

Since this is a relatively small addition without any effect on existing nodes, I do not see major reasons not to add this retriever. The only consideration when using this node is the need to have a metadata field for aggregating documents to file level. The default file_id meta field works well for that. It is also important to make sure that the document store is compatible with the chosen retrieval method(s), but that is the case when using any other retriever node as well.

# Alternatives

Without adding this feature it will not be possible to quickly retrive similar documents to a specific file, see the Motivation section for example situations where this is useful.

# Adoption strategy

This is not a breaking change and there does not seem to be any need for a migration script. Existing Haystack users can just start using this node on as-needed basis by itself, no need for a reader/PromptNode after the FileSimilarityRetriever.

# How we teach this

It could be good to have a short tutorial about how this node is used, as it's slightly different in the query input and in the outputs from other retriever nodes. Alternatively, a blog post could be written about it.

As for documentation needs, some info on how to use this retriever would be good to add to the Retrievers page (https://docs.haystack.deepset.ai/docs/retriever). If you need help writing the documentation and/or the blog post/tutorial, please do not hesitate to reach out to me.

# Unresolved questions

Not many unresolved questions, I'll just need to see if the retriever can be adopted so it inherits from the BaseRetriever and not BaseComponent. Plus I might need to add the "retrieve_batch" method as currently it only has "retrieve".

Another open question is whether it would be a good idea to enable providing a JoinDocuments node in the parameters (after the primary_retriever and secondary_retriever), to make results aggregation more flexible. This would make it possible to (in the definition of JoinDocuments) choose the join_mode (concatenate/merge/reciprocal_rank_fusion) and in case "merge" is chosen, it would also be possible to set weights per retriever.

Alternatively, we could change how the FileSimilarityRetriever works and instead of primary_retriever + secondary_retriever + join_node provide it right away with a hybrid document search pipeline that includes all these elements, and just make FileSimilarityRetriever iteratively perform the document search for all docs pertaining to a file and output the top_k documents found. But this looping of doc search pipeline execution within a filesim pipeline is not a typical design pattern in Haystack v1 and we are unsure if it would be a good approach.
