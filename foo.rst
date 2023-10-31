.. _relnotes_v1.22.0-rc1:

v1.22.0-rc1
===========

.. _relnotes_v1.22.0-rc1_Upgrade Notes:

Upgrade Notes
-------------

- This update enables all Pinecone index types to be used, including Starter.
  Previously, Pinecone Starter index type couldn't be used as document store. Due to limitations of this index type
  (https://docs.pinecone.io/docs/starter-environment), in current implementation fetching documents is limited to
  Pinecone query vector limit (10000 vectors). Accordingly, if the number of documents in the index is above this limit,
  some of PineconeDocumentStore functions will be limited.

- Removes the `audio`, `ray`, `onnx` and `beir` extras from the extra group `all`.


.. _relnotes_v1.22.0-rc1_New Features:

New Features
------------

- Add experimental support for asynchronous `Pipeline` run


.. _relnotes_v1.22.0-rc1_Enhancement Notes:

Enhancement Notes
-----------------

- Added support for Apple Silicon GPU acceleration through "mps pytorch", enabling better performance on Apple M1 hardware.

- Document writer returns the number of documents written.

- added support for using `on_final_answer` trough `Agent` `callback_manager`

- Add asyncio support to the OpenAI invocation layer.

- PromptNode can now be run asynchronously by calling the `arun` method.

- Add `search_engine_kwargs` param to WebRetriever so it can be propagated
  to WebSearch. This is useful, for example, to pass the engine id when
  using Google Custom Search.

- Upgrade Transformers to the latest version 4.34.1.
  This version adds support for the new Mistral, Persimmon, BROS, ViTMatte, and Nougat models.

- Make JoinDocuments return only the document with the highest score if there are duplicate documents in the list.

- Add `list_of_paths` argument to `utils.convert_files_to_docs` to allow 
  input of list of file paths to be converted, instead of, or as well as, 
  the current `dir_path` argument.

- Optimize particular methods from PineconeDocumentStore (delete_documents and _get_vector_count)

- Update the deepset Cloud SDK to the new endpoint format for new saving pipeline configs.

- Add alias names for Cohere embed models for an easier map between names


.. _relnotes_v1.22.0-rc1_Deprecation Notes:

Deprecation Notes
-----------------

- Deprecate `OpenAIAnswerGenerator` in favor of `PromptNode`.
  `OpenAIAnswerGenerator` will be removed in Haystack 1.23.


.. _relnotes_v1.22.0-rc1_Bug Fixes:

Bug Fixes
---------

- Fixed the bug that prevented the correct usage of ChatGPT invocation layer
  in 1.21.1.
  Added async support for ChatGPT invocation layer.

- Added documents_store.update_embeddings call to pipeline examples so that embeddings are calculated for newly added documents.

- Remove unsupported `medium` and `finance-sentiment` models from supported Cohere embed model list


.. _relnotes_v1.22.0-rc1_Haystack 2.0 preview:

Haystack 2.0 preview
--------------------

- Add AzureOCRDocumentConverter to convert files of different types using Azure's Document Intelligence Service.

- Add ByteStream type to send binary raw data across components
  in a pipeline.

- Introduce ChatMessage data class to facilitate structured handling and processing of message content
  within LLM chat interactions.

- Adds `ChatMessage` templating in `PromptBuilder`

- Adds HTMLToDocument component to convert HTML to a Document.

- Adds SimilarityRanker, a component that ranks a list of Documents based on their similarity to the query.

- Introduce the StreamingChunk dataclass for efficiently handling chunks of data streamed from a language model,
  encapsulating both the content and associated metadata for systematic processing.

- Adds TopPSampler, a component selects documents based on the cumulative probability of the Document scores using top p (nucleus) sampling.

- Add `dumps`, `dump`, `loads` and `load` methods to save
  and load pipelines in Yaml format.

- Adopt Hugging Face `token` instead of the deprecated `use_auth_token`.
  Add this parameter to `ExtractiveReader` and `SimilarityRanker` to allow
  loading private models.
  Proper handling of `token` during serialization: if it is a string (a possible valid token)
  it is not serialized.

- Add `mime_type` field to `ByteStream` dataclass.

- The Document dataclass checks if `id_hash_keys` is None or empty in
  __post_init__. If so, it uses the default factory to set a default valid value.

- Rework `Document.id` generation, if an `id` is not explicitly set it's generated
  using all `Document` field' values, `score` is not used.

- Change `Document`'s `embedding` field type from `numpy.ndarray` to `List[float]`

- Fixed a bug that caused TextDocumentSplitter and DocumentCleaner to ignore id_hash_keys and create Documents with duplicate ids if the documents differed only in their metadata.

- Fix TextDocumentSplitter failing when run with an empty list

- Better management of API key in GPT Generator. The API key is never serialized.
  Make the `api_base_url` parameter really used (previously it was ignored).

- Add a minimal version of HuggingFaceLocalGenerator, a component that can run
  Hugging Face models locally to generate text.

- Migrate RemoteWhisperTranscriber to OpenAI SDK.

- Add OpenAI Document Embedder.
  It computes embeddings of Documents using OpenAI models.
  The embedding of each Document is stored in the `embedding` field of the Document.

- Add the `TextDocumentSplitter` component for Haystack 2.0 that splits a Document with long text into multiple Documents with shorter texts. Thereby the texts match the maximum length that the language models in Embedders or other components can process.

- Refactor OpenAIDocumentEmbedder to enrich documents with embeddings instead of recreating them.

- Refactor SentenceTransformersDocumentEmbedder to enrich documents with embeddings instead of recreating them.

- Remove "api_key" from serialization of AzureOCRDocumentConverter and SerperDevWebSearch.

- Removed implementations of from_dict and to_dict from all components where they had the same effect as the default implementation from Canals: https://github.com/deepset-ai/canals/blob/main/canals/serialization.py#L12-L13 This refactoring does not change the behavior of the components.

- Remove `array` field from `Document` dataclass.

- Remove `id_hash_keys` field from `Document` dataclass.
  `id_hash_keys` has been also removed from Components that were using it:
  * `DocumentCleaner`
  * `TextDocumentSplitter`
  * `PyPDFToDocument`
  * `AzureOCRDocumentConverter`
  * `HTMLToDocument`
  * `TextFileToDocument`
  * `TikaDocumentConverter`

- Enhanced file routing capabilities with the introduction of `ByteStream` handling, and
  improved clarity by renaming the router to `FileTypeRouter`.

- Rename `MemoryDocumentStore` to `InMemoryDocumentStore`
  Rename `MemoryBM25Retriever` to `InMemoryBM25Retriever`
  Rename `MemoryEmbeddingRetriever` to `InMemoryEmbeddingRetriever`

- Renamed ExtractiveReader's input from `document` to `documents` to match its type List[Document].

- Rename `SimilarityRanker` to `TransformersSimilarityRanker`,
  as there will be more similarity rankers in the future.

- Allow specifying stopwords to stop text generation for `HuggingFaceLocalGenerator`.

- Add basic telemetry to Haystack 2.0 pipelines

- Added DocumentCleaner, which removes extra whitespace, empty lines, headers, etc. from Documents containing text.
  Useful as a preprocessing step before splitting into shorter text documents.

- Add TextLanguageClassifier component so that an input string, for example a query, can be routed to different components based on the detected language.

- Upgrade canals to 0.9.0 to support variadic inputs for Joiner components and "/" in connection names like "text/plain"


