- Start Date: 2022-11-11
- Proposal PR: https://github.com/deepset-ai/haystack/pull/3558
- Github Issue:

> ⚠️ Superseded by https://github.com/deepset-ai/haystack/blob/main/proposals/text/5390-embedders.md

  # Summary

- Current EmbeddingRetriever doesn't allow Haystack users to provide new embedding methods and is
  currently constricted to farm, transformers, sentence transformers, OpenAI and Cohere based
  embedding approaches. Any new encoding methods need to be explicitly added to Haystack
  and registered with the EmbeddingRetriever.


- We should allow users to easily plug-in new embedding methods to EmbeddingRetriever. For example, a Haystack user should be able to
  add custom embeddings without having to commit additional code to Haystack repository.

  # Basic example
    EmbeddingRetriever is instantiated with:

  ``` python
	  retriever = EmbeddingRetriever(
	      document_store=document_store,
	      embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
	      model_format="sentence_transformers",
	  )
  ```
- The current approach doesn't provide a pluggable abstraction point of composition but
  rather attempts to satisfy various embedding methodologies by having a lot of
  parameters which keep ever expanding.


- The new approach allows creation of the underlying embedding mechanism (EmbeddingEncoder)
  which is then in turn plugged into EmbeddingRetriever.  For example:

  ``` python
	  encoder = SomeNewFancyEmbeddingEncoder(api_key="asdfklklja",
                                          query_model="text-search-query",
                                          doc_model="text-search-doc")
  ```

- EmbeddingEncoder is then used for the creation of EmbeddingRetriever. EmbeddingRetriever
  init method doesn't get polluted with additional parameters as all of the peculiarities
  of a particular encoder methodology are contained on in its abstraction layer.

  ``` python
	  retriever = EmbeddingRetriever(
	      document_store=document_store,
	      encoder=encoder
	  )
  ```

  # Motivation

- Why are we doing this? What use cases does it support? What is the expected outcome?

  We could certainly keep the current solution as is; it does implement a decent level
  of composition/decoration to lower coupling between EmbeddingRetriever and the underlying
  mechanism of embedding (sentence transformers, OpenAI, etc). However, the current mechanism
  in place basically hard-codes available embedding implementations and prevents our users from
  adding new embedding mechanism by themselves outside of Haystack repository. We also might
  want to have a non-public dC embedding mechanism in the future. In the current design a non-public
  dC embedding mechanism would be impractical. In addition, the more underlying implementations we
  add we'll continue to "pollute" EmbeddingRetriever init method with more and more parameters.
  This is certainly less than ideal long term.


- EmbeddingEncoder classes should be subclasses of BaseComponent! As subclasses of BaseComponent,
  we can use them outside the EmbeddingRetriever context in indexing pipelines, generating the
  embeddings. We are currently employing a kludge of using Retrievers which is quite counter-intuitive
  and confusing for our users.


- EmbeddingEncoder classes might sound overly complicated, especially with a distinguishing mechanism
  name pre-appended (i.e CohereEmbeddingEncoder). Therefore, we'll adopt <specific>Embedder
  naming scheme, i.e. CohereEmbedder, SentenceTransformerEmbedder and so on.

  # Detailed design

- Our new EmbeddingRetriever would still wrap the underlying encoding mechanism in the form of
  _BaseEmbedder. _BaseEmbedder still needs to implement methods:
	- embed_queries
	- embed_documents


- The new design approach differs is in the creation of EmbeddingRetriever - rather than hiding the underlying encoding
  mechanism one could simply create the EmbeddingRetriever with a specific encoder directly. For example:

  ```
	  retriever = EmbeddingRetriever(
	      document_store=document_store,
	      encoder=OpenAIEmbedder(api_key="asdfklklja", model="ada"),
	      #additional EmbeddingRetriever-abstraction-level parameters
	  )
  ```

- If the "two-step approach" of EmbeddingRetriever initialization is no longer the ideal solution (issues with current
  schema generation and loading/saving via YAML pipelines) we might simply add the EmbeddingRetriever
  class for every supported encoding approach. For example, we could have OpenAIEmbeddingRetriever, CohereEmbeddingRetriever,
  SentenceTransformerEmbeddingRetriever and so on. Each of these retrievers will delegate the bulk of the work to an
  existing EmbeddingRetriever with a per-class-specific Embedder set in the class constructor (for that custom
  encoding part). We'll get the best of both worlds. Each <Specific>EmeddingRetriever will have only the relevant primitives
  parameters for the **init()** constructor; the underlying EmbeddingRetriever attribute in <Specific>EmeddingRetriever
  will handle most of the business logic of retrieving, yet each retriever will use an appropriate per-class-specific
  Embedder for the custom encoding part.



  # Drawbacks
- The main shortcoming are:
	- The "two-step approach" in EmbeddingRetriever initialization
	- Likely be an issue for the current schema generation and loading/saving via YAML pipelines (see solution above)
	- It is a API breaking change so it'll require code update for all EmbeddingRetriever usage both in our codebase and for Haystack users
	- Can only be done in major release along with other breaking changes

  # Alternatives

  We could certainly keep everything as is :-)

  # Adoption strategy
- As it is a breaking change, we should implement it for the next major release.

  # How do we teach this?
- This change would require only a minor change in documentation.
- The concept of embedding retriever remains, just the mechanics are slightly changed
- All docs and tutorials need to be updated
- Haystack users are informed about a possibility to create and use their own embedders for embedding retriever.
- # Unresolved questions

  Optional, but suggested for first drafts. What parts of the design are still
  TBD?
