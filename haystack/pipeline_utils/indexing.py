import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, List, Any, Dict
from typing import Union, Type

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, OpenAIDocumentEmbedder
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DocumentStore


def download_files(sources: List[str]) -> List[str]:
    """
    Downloads a list of files from the web and returns a list of their paths where they are stored locally.
    :param sources: A list of URLs to download.
    :type sources: List[str]
    :return: A list of paths to the downloaded files.
    """

    fetcher = LinkContentFetcher(user_agents=["Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"])
    streams = fetcher.run(urls=sources)

    all_files = []
    for stream in streams["streams"]:
        file_suffix = ".html" if stream.meta["content_type"] == "text/html" else ".pdf"
        f = NamedTemporaryFile(delete=False, suffix=file_suffix)
        stream.to_file(Path(f.name))
        all_files.append(f.name)

    return all_files


def build_indexing_pipeline(
    document_store: Any,
    embedding_model: str = "intfloat/e5-base-v2",
    embedding_model_kwargs: Optional[Dict[str, Any]] = None,
    supported_mime_types: Optional[List[str]] = None,
):
    """
    Returns a prebuilt pipeline for indexing documents into a DocumentStore. Indexing pipeline automatically detects
    the file type of the input files and converts them into Documents. The supported file types are: .txt,
    .pdf, and .html but by default only .txt and .html are indexed unless the supported_mime_types parameter is set.

    Example usage:

    ```python
    from haystack.utils import build_indexing_pipeline
    indexing_pipe = build_indexing_pipeline(document_store=your_document_store_instance)
    indexing_pipe.run(files=["path/to/file1", "path/to/file2"])
    >>> {'documents_written': 2}
    ```

    One can also pass an embedding model to the pipeline, which will then calculate embeddings for the documents
    and store them in the DocumentStore. Example usage:
    ```python
    indexing_pipe = build_indexing_pipeline(document_store=your_document_store_instance,
                                            embedding_model="sentence-transformers/all-mpnet-base-v2")
    indexing_pipe.run(files=["path/to/file1", "path/to/file2"])
    >>> {'documents_written': 2}
    ```

    After running indexing pipeline, the documents are indexed in the DocumentStore and can be used for querying.


    :param document_store: An instance of a DocumentStore to index documents into.
    :param embedding_model: The name of the model to use for document embeddings.
                            Needs top be compatible with SentenceTransformers
    :param embedding_model_kwargs: Keyword arguments to pass to the embedding model class.
    :param supported_mime_types: List of MIME types to support in the pipeline. If not given,
                                     defaults to ["text/plain", "text/html"].

    """
    return _IndexingPipeline(
        document_store=document_store,
        embedding_model=embedding_model,
        embedding_model_kwargs=embedding_model_kwargs,
        supported_mime_types=supported_mime_types,
    )


class _IndexingPipeline:
    """
    An internal class to simplify creation of prebuilt pipeline for indexing documents into a DocumentStore. Indexing
    pipeline automatically detect the file type of the input files and converts them into Documents. The supported
    file types are: .txt, .pdf, and .html
    """

    def __init__(
        self,
        document_store: DocumentStore,
        embedding_model: Optional[str] = None,
        embedding_model_kwargs: Optional[Dict[str, Any]] = None,
        supported_mime_types: Optional[List[str]] = None,
    ):
        """
        :param document_store: An instance of a DocumentStore to index documents into.
        :param embedding_model: The name of the model to use for document embeddings.
        :param supported_mime_types: List of MIME types to support in the pipeline. If not given,
                                     defaults to ["text/plain", "application/pdf", "text/html"].
        """

        if supported_mime_types is None:
            supported_mime_types = ["text/plain", "text/html"]

        self.pipeline = Pipeline()
        self.pipeline.add_component("file_type_router", FileTypeRouter(mime_types=supported_mime_types))
        converters_used: List[str] = []
        # Add converters dynamically based on MIME types
        if "text/plain" in supported_mime_types:
            self.pipeline.add_component("text_file_converter", TextFileToDocument())
            self.pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
            converters_used.append("text_file_converter")

        if "application/pdf" in supported_mime_types:
            from haystack.components.converters import PyPDFToDocument

            self.pipeline.add_component("pdf_file_converter", PyPDFToDocument())
            self.pipeline.connect("file_type_router.application/pdf", "pdf_file_converter.sources")
            converters_used.append("pdf_file_converter")

        if "text/html" in supported_mime_types:
            from haystack.components.converters import HTMLToDocument

            self.pipeline.add_component("html_file_converter", HTMLToDocument())
            self.pipeline.connect("file_type_router.text/html", "html_file_converter.sources")
            converters_used.append("html_file_converter")

        # Add remaining common components
        self.pipeline.add_component("document_joiner", DocumentJoiner())
        self.pipeline.add_component("document_cleaner", DocumentCleaner())
        self.pipeline.add_component("document_splitter", DocumentSplitter())

        # Connect converters to joiner, if they exist
        for converter_name in converters_used:
            self.pipeline.connect(f"{converter_name}.documents", "document_joiner.documents")

        # Connect joiner to cleaner and splitter
        self.pipeline.connect("document_joiner.documents", "document_cleaner.documents")
        self.pipeline.connect("document_cleaner.documents", "document_splitter.documents")

        if embedding_model:
            embedder_instance = self._find_embedder(embedding_model, embedding_model_kwargs)
            self.pipeline.add_component("storage_sink", DocumentWriter(document_store=document_store))
            self.pipeline.add_component("writer", embedder_instance)
            self.pipeline.connect("writer", "storage_sink")
        else:
            self.pipeline.add_component("writer", DocumentWriter(document_store=document_store))

        self.pipeline.connect("document_splitter.documents", "writer.documents")

        # this is more of a sanity check for the maintainer of the pipeline, to make sure that the pipeline is
        # configured correctly
        if len(self.pipeline.inputs()) < 1:
            raise RuntimeError("IndexingPipeline needs at least one input component.")
        if len(self.pipeline.outputs()) < 1:
            raise RuntimeError("IndexingPipeline needs at least one output component.")

    def run(self, files: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Performs indexing of the given list of documents into the DocumentStore.
        :param files: A list of paths to files to index.
        :type files: List[Union[str, Path]]

        :return: the output of the pipeline run, which is a dictionary containing the number of documents written
        """
        if not files:
            return {"documents_written": 0}
        input_files = self._process_files(files)
        pipeline_output = self.pipeline.run(data={"file_type_router": {"sources": input_files}})
        aggregated_results = {}
        # combine the results of all outputs into one dictionary
        for component_result in pipeline_output.values():
            aggregated_results.update(component_result)
        return aggregated_results

    def _find_embedder(self, embedding_model: str, init_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        embedder_patterns = {
            r"^text-embedding.*": OpenAIDocumentEmbedder,
            r"^sentence-transformers.*": SentenceTransformersDocumentEmbedder,
            r"^intfloat/.*": SentenceTransformersDocumentEmbedder,
            # add more patterns or adjust them here
        }
        embedder_class = next((val for pat, val in embedder_patterns.items() if re.match(pat, embedding_model)), None)
        if not embedder_class:
            raise ValueError(
                f"Could not find an embedder for the given embedding model name {embedding_model}. "
                f"Please provide a valid embedding model name. "
                f"Valid embedder classes are {embedder_patterns.values()}."
            )
        return self._create_embedder(embedder_class, embedding_model, init_kwargs)

    def _create_embedder(self, embedder_class: Type, model: str, init_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        # Note: here we assume the embedder accepts a parameter called `model` that takes the model's name or path.
        # See https://github.com/deepset-ai/haystack/issues/6534
        kwargs = {**(init_kwargs or {}), "model": model}
        return embedder_class(**kwargs)

    def _list_files_recursively(self, path: Union[str, Path]) -> List[str]:
        """
        List all files in a directory recursively as a list of strings, or return the file itself
        if it's not a directory.
        :param path: the path to list files from
        :type path: Union[str, Path]
        :return: a list of strings, where each string is a path to a file
        """

        if os.path.isfile(path):
            return [str(path)]
        elif os.path.isdir(path):
            file_list: List[str] = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_list.append(os.path.join(root, file))
            return file_list
        else:
            return []

    def _process_files(self, files: List[Union[str, Path]]) -> List[str]:
        """
        Process a list of files and directories, listing all files recursively and removing duplicates.
        :param files: A list of files and directories to process.
        :type files: List[Union[str, Path]]
        :return: A list of unique files.
        """
        nested_file_lists = [self._list_files_recursively(file) for file in files]
        combined_files = [item for sublist in nested_file_lists for item in sublist]
        unique_files = list(set(combined_files))
        return unique_files
