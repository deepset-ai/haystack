# v1.22.0-rc1

## Upgrade Notes

-   This update enables all Pinecone index types to be used, including
    Starter. Previously, Pinecone Starter index type couldn't be used as
    document store. Due to limitations of this index type
    (<https://docs.pinecone.io/docs/starter-environment>), in current
    implementation fetching documents is limited to Pinecone query
    vector limit (10000 vectors). Accordingly, if the number of
    documents in the index is above this limit, some of
    PineconeDocumentStore functions will be limited.
-   Removes the <span class="title-ref">audio</span>,
    <span class="title-ref">ray</span>,
    <span class="title-ref">onnx</span> and
    <span class="title-ref">beir</span> extras from the extra group
    <span class="title-ref">all</span>.

## New Features

-   Add experimental support for asynchronous
    <span class="title-ref">Pipeline</span> run

## Enhancement Notes

-   Added support for Apple Silicon GPU acceleration through "mps
    pytorch", enabling better performance on Apple M1 hardware.
-   Document writer returns the number of documents written.
-   added support for using
    <span class="title-ref">on_final_answer</span> trough
    <span class="title-ref">Agent</span>
    <span class="title-ref">callback_manager</span>
-   Add asyncio support to the OpenAI invocation layer.
-   PromptNode can now be run asynchronously by calling the
    <span class="title-ref">arun</span> method.
-   Add <span class="title-ref">search_engine_kwargs</span> param to
    WebRetriever so it can be propagated to WebSearch. This is useful,
    for example, to pass the engine id when using Google Custom Search.
-   Upgrade Transformers to the latest version 4.34.1. This version adds
    support for the new Mistral, Persimmon, BROS, ViTMatte, and Nougat
    models.
-   Make JoinDocuments return only the document with the highest score
    if there are duplicate documents in the list.
-   Add <span class="title-ref">list_of_paths</span> argument to
    <span class="title-ref">utils.convert_files_to_docs</span> to allow
    input of list of file paths to be converted, instead of, or as well
    as, the current <span class="title-ref">dir_path</span> argument.
-   Optimize particular methods from PineconeDocumentStore
    (delete_documents and \_get_vector_count)
-   Update the deepset Cloud SDK to the new endpoint format for new
    saving pipeline configs.
-   Add alias names for Cohere embed models for an easier map between
    names

## Deprecation Notes

-   Deprecate <span class="title-ref">OpenAIAnswerGenerator</span> in
    favor of <span class="title-ref">PromptNode</span>.
    <span class="title-ref">OpenAIAnswerGenerator</span> will be removed
    in Haystack 1.23.

## Bug Fixes

-   Fixed the bug that prevented the correct usage of ChatGPT invocation
    layer in 1.21.1. Added async support for ChatGPT invocation layer.
-   Added documents_store.update_embeddings call to pipeline examples so
    that embeddings are calculated for newly added documents.
-   Remove unsupported <span class="title-ref">medium</span> and
    <span class="title-ref">finance-sentiment</span> models from
    supported Cohere embed model list

## Haystack 2.0 preview

-   Add AzureOCRDocumentConverter to convert files of different types
    using Azure's Document Intelligence Service.
-   Add ByteStream type to send binary raw data across components in a
    pipeline.
-   Introduce ChatMessage data class to facilitate structured handling
    and processing of message content within LLM chat interactions.
-   Adds <span class="title-ref">ChatMessage</span> templating in
    <span class="title-ref">PromptBuilder</span>
-   Adds HTMLToDocument component to convert HTML to a Document.
-   Adds SimilarityRanker, a component that ranks a list of Documents
    based on their similarity to the query.
-   Introduce the StreamingChunk dataclass for efficiently handling
    chunks of data streamed from a language model, encapsulating both
    the content and associated metadata for systematic processing.
-   Adds TopPSampler, a component selects documents based on the
    cumulative probability of the Document scores using top p (nucleus)
    sampling.
-   Add <span class="title-ref">dumps</span>,
    <span class="title-ref">dump</span>,
    <span class="title-ref">loads</span> and
    <span class="title-ref">load</span> methods to save and load
    pipelines in Yaml format.
-   Adopt Hugging Face <span class="title-ref">token</span> instead of
    the deprecated <span class="title-ref">use_auth_token</span>. Add
    this parameter to <span class="title-ref">ExtractiveReader</span>
    and <span class="title-ref">SimilarityRanker</span> to allow loading
    private models. Proper handling of
    <span class="title-ref">token</span> during serialization: if it is
    a string (a possible valid token) it is not serialized.
-   Add <span class="title-ref">mime_type</span> field to
    <span class="title-ref">ByteStream</span> dataclass.
-   The Document dataclass checks if
    <span class="title-ref">id_hash_keys</span> is None or empty in
    \_\_post_init\_\_. If so, it uses the default factory to set a
    default valid value.
-   Rework <span class="title-ref">Document.id</span> generation, if an
    <span class="title-ref">id</span> is not explicitly set it's
    generated using all <span class="title-ref">Document</span> field'
    values, <span class="title-ref">score</span> is not used.
-   Change <span class="title-ref">Document</span>'s
    <span class="title-ref">embedding</span> field type from
    <span class="title-ref">numpy.ndarray</span> to
    <span class="title-ref">List\[float\]</span>
-   Fixed a bug that caused TextDocumentSplitter and DocumentCleaner to
    ignore id_hash_keys and create Documents with duplicate ids if the
    documents differed only in their metadata.
-   Fix TextDocumentSplitter failing when run with an empty list
-   Better management of API key in GPT Generator. The API key is never
    serialized. Make the <span class="title-ref">api_base_url</span>
    parameter really used (previously it was ignored).
-   Add a minimal version of HuggingFaceLocalGenerator, a component that
    can run Hugging Face models locally to generate text.
-   Migrate RemoteWhisperTranscriber to OpenAI SDK.
-   Add OpenAI Document Embedder. It computes embeddings of Documents
    using OpenAI models. The embedding of each Document is stored in the
    <span class="title-ref">embedding</span> field of the Document.
-   Add the <span class="title-ref">TextDocumentSplitter</span>
    component for Haystack 2.0 that splits a Document with long text
    into multiple Documents with shorter texts. Thereby the texts match
    the maximum length that the language models in Embedders or other
    components can process.
-   Refactor OpenAIDocumentEmbedder to enrich documents with embeddings
    instead of recreating them.
-   Refactor SentenceTransformersDocumentEmbedder to enrich documents
    with embeddings instead of recreating them.
-   Remove "api_key" from serialization of AzureOCRDocumentConverter and
    SerperDevWebSearch.
-   Removed implementations of from_dict and to_dict from all components
    where they had the same effect as the default implementation from
    Canals:
    <https://github.com/deepset-ai/canals/blob/main/canals/serialization.py#L12-L13>
    This refactoring does not change the behavior of the components.
-   Remove <span class="title-ref">array</span> field from
    <span class="title-ref">Document</span> dataclass.
-   Remove <span class="title-ref">id_hash_keys</span> field from
    <span class="title-ref">Document</span> dataclass.
    <span class="title-ref">id_hash_keys</span> has been also removed
    from Components that were using it:
    -   <span class="title-ref">DocumentCleaner</span>
    -   <span class="title-ref">TextDocumentSplitter</span>
    -   <span class="title-ref">PyPDFToDocument</span>
    -   <span class="title-ref">AzureOCRDocumentConverter</span>
    -   <span class="title-ref">HTMLToDocument</span>
    -   <span class="title-ref">TextFileToDocument</span>
    -   <span class="title-ref">TikaDocumentConverter</span>
-   Enhanced file routing capabilities with the introduction of
    <span class="title-ref">ByteStream</span> handling, and improved
    clarity by renaming the router to
    <span class="title-ref">FileTypeRouter</span>.
-   Rename <span class="title-ref">MemoryDocumentStore</span> to
    <span class="title-ref">InMemoryDocumentStore</span> Rename
    <span class="title-ref">MemoryBM25Retriever</span> to
    <span class="title-ref">InMemoryBM25Retriever</span> Rename
    <span class="title-ref">MemoryEmbeddingRetriever</span> to
    <span class="title-ref">InMemoryEmbeddingRetriever</span>
-   Renamed ExtractiveReader's input from
    <span class="title-ref">document</span> to
    <span class="title-ref">documents</span> to match its type
    List\[Document\].
-   Rename <span class="title-ref">SimilarityRanker</span> to
    <span class="title-ref">TransformersSimilarityRanker</span>, as
    there will be more similarity rankers in the future.
-   Allow specifying stopwords to stop text generation for
    <span class="title-ref">HuggingFaceLocalGenerator</span>.
-   Add basic telemetry to Haystack 2.0 pipelines
-   Added DocumentCleaner, which removes extra whitespace, empty lines,
    headers, etc. from Documents containing text. Useful as a
    preprocessing step before splitting into shorter text documents.
-   Add TextLanguageClassifier component so that an input string, for
    example a query, can be routed to different components based on the
    detected language.
-   Upgrade canals to 0.9.0 to support variadic inputs for Joiner
    components and "/" in connection names like "text/plain"
