# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Literal

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.image.document_to_image import DocumentToImageContent
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import TextContent
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)

ExtractionMode = Literal["content", "metadata"]


DEFAULT_PROMPT_TEMPLATE = """
You are part of an information extraction pipeline that extracts the content of image-based documents.

Extract the content from the provided image.
You need to extract the content exactly.
Format everything as markdown.
Make sure to retain the reading order of the document.

**Visual Elements**
Do not extract figures, drawings, maps, graphs or any other visual elements.
Instead, add a caption that describes briefly what you see in the visual element.
You must describe each visual element.
If you only see a visual element without other content, you must describe this visual element.
Enclose each image caption with [img-caption][/img-caption]

**Tables**
Make sure to format the table in markdown.
Add a short caption below the table that describes the table's content.
Enclose each table caption with [table-caption][/table-caption].
The caption must be placed below the extracted table.

**Forms**
Reproduce checkbox selections with markdown.

Go ahead and extract!

Document:"""


DEFAULT_METADATA_PROMPT_TEMPLATE = """
You are part of an information extraction pipeline that extracts metadata from image-based documents.

Look at the provided image and extract metadata that describes the document. Return only the extracted metadata.

Extract whatever is visible and relevant, such as:
- title: document or page title if visible
- author: author or creator if visible
- date: publication date, creation date, or any date shown
- document_type: letter, invoice, form, report, receipt, etc.
- summary: one or two sentences describing what the document is about
- keywords: important topics or entities (e.g. names, organizations, amounts)

Format your response as a flat list of key-value pairs, one per line, e.g.:
title: Example Document
author: John Doe
date: 2024-01-15
document_type: invoice

If a field is not visible or not applicable, omit it. Be concise.

Document:"""


def _validate_prompt_no_variables(prompt: str, context: str) -> None:
    ast = SandboxedEnvironment().parse(prompt)
    template_variables = meta.find_undeclared_variables(ast)
    variables = list(template_variables)
    if variables:
        raise ValueError(
            f"The prompt must not have any variables, only instructions on how to {context} "
            f"the image-based document. Found {','.join(variables)} in the prompt."
        )


@component
class LLMDocumentContentExtractor:
    """
    Extracts textual content or metadata from image-based documents using a vision-enabled Large-Language Model (LLM).

    This component converts each input document into an image using the DocumentToImageContent component,

    This component can run in two modes, chosen at init and optionally overridden at runtime:
    - **content**: the LLM output is written to the document's content (default, backward compatible).
    - **metadata**: the LLM output is stored in the document's metadata under a configurable key.

    One LLM call is made per document. The component converts each document to an image via
    DocumentToImageContent, sends it with a prompt to the ChatGenerator, and writes the response
    to content or metadata according to the effective extraction mode.

    Prompts must not contain Jinja variables,they should only include instructions for the LLM. Image data
    and the instructions for the LLM.

    Documents for which the LLM fails to extract content are returned in a separate `failed_documents` list. These
    failed documents will have either a `content_extraction_error` or `metadata_extraction_error` entry in their
    metadata. This metadata can be used for debugging or for reprocessing the documents later.

    ### Usage example (content mode, default)
    ```python
    from haystack import Document
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.extractors.image import LLMDocumentContentExtractor

    chat_generator = OpenAIChatGenerator()
    extractor = LLMDocumentContentExtractor(chat_generator=chat_generator)
    documents = [
        Document(content="", meta={"file_path": "image.jpg"}),
        Document(content="", meta={"file_path": "document.pdf", "page_number": 1}),
    ]
    updated_documents = extractor.run(documents=documents)["documents"]
    print(updated_documents)
    # [Document(content='Extracted text from image.jpg',
    #           meta={'file_path': 'image.jpg'}),
    #  ...]
    ```

    ### Usage example (metadata mode, runtime override)
    ```python
    extractor = LLMDocumentContentExtractor(chat_generator=chat_generator, extraction_mode="metadata")
    result = extractor.run(documents=documents)
    # result["documents"][0].meta["extracted_metadata"] contains the extraction

    # Override to content for this run only
    result = extractor.run(documents=documents, extraction_mode="content")
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        extraction_mode: ExtractionMode = "content",
        prompt: str = DEFAULT_PROMPT_TEMPLATE,
        metadata_prompt: str = DEFAULT_METADATA_PROMPT_TEMPLATE,
        metadata_key: str = "extracted_metadata",
        file_path_meta_field: str = "file_path",
        root_path: str | None = None,
        detail: Literal["auto", "high", "low"] | None = None,
        size: tuple[int, int] | None = None,
        raise_on_failure: bool = False,
        max_workers: int = 3,
    ):
        """
        Initialize the LLMDocumentContentExtractor component.

        :param chat_generator: A ChatGenerator instance representing the LLM used to extract text. This generator must
            support vision-based input and return a plain text response.
        :param extraction_mode: Where to write the LLM output: "content" (default) or "metadata". Can be overridden
            in run().
        :param prompt: Prompt used when extraction_mode is "content". Must not contain Jinja variables.
        :param metadata_prompt: Prompt used when extraction_mode is "metadata". Must not contain Jinja variables.
        :param metadata_key: Metadata key used when extraction_mode is "metadata". Ignored in content mode.
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param detail: Optional detail level of the image (only supported by OpenAI). Can be "auto", "high", or "low".
            This will be passed to chat_generator when processing the images.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param raise_on_failure: If True, exceptions from the LLM are raised. If False, failed documents are logged
            and returned.
        :param max_workers: Maximum number of threads used to parallelize LLM calls across documents using a
            ThreadPoolExecutor.
        """
        self._chat_generator = chat_generator
        self.extraction_mode = extraction_mode
        self.content_prompt = prompt
        self.metadata_prompt = metadata_prompt
        self.metadata_key = metadata_key
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.detail = detail
        self.size = size
        _validate_prompt_no_variables(prompt, "extract the content of")
        _validate_prompt_no_variables(metadata_prompt, "extract the metadata of")
        self.raise_on_failure = raise_on_failure
        self.max_workers = max_workers
        self._document_to_image_content = DocumentToImageContent(
            file_path_meta_field=file_path_meta_field, root_path=root_path, detail=detail, size=size
        )
        self._is_warmed_up = False

    def warm_up(self):
        """
        Warm up the ChatGenerator if it has a warm_up method.
        """
        if not self._is_warmed_up:
            if hasattr(self._chat_generator, "warm_up"):
                self._chat_generator.warm_up()
            self._is_warmed_up = True

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self._chat_generator, name="chat_generator"),
            extraction_mode=self.extraction_mode,
            prompt=self.content_prompt,
            metadata_prompt=self.metadata_prompt,
            metadata_key=self.metadata_key,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
            detail=self.detail,
            size=self.size,
            raise_on_failure=self.raise_on_failure,
            max_workers=self.max_workers,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMDocumentContentExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")
        return default_from_dict(cls, data)

    def _run_on_thread(self, message: ChatMessage | None) -> dict[str, Any]:
        """
        Execute the LLM inference in a separate thread for each document.

        :param message: A ChatMessage containing the prompt and image content for the LLM.
        :returns:
            The LLM response if successful, or a dictionary with an "error" key on failure.
        """
        # If message is None, return an error dictionary
        if message is None:
            return {"error": "Document has no content, skipping LLM call."}

        try:
            result = self._chat_generator.run(messages=[message])
        except Exception as e:
            if self.raise_on_failure:
                raise e
            logger.error(
                "LLM {class_name} execution failed. Skipping metadata extraction. Failed with exception '{error}'.",
                class_name=self._chat_generator.__class__.__name__,
                error=e,
            )
            result = {"error": "LLM failed with exception: " + str(e)}
        return result

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    def run(
        self, documents: list[Document], extraction_mode: ExtractionMode | None = None
    ) -> dict[str, list[Document]]:
        """
        Run extraction on image-based documents. One LLM call per document;

        Note that the output goes to content or metadata according to the extraction mode.

        :param documents: A list of image-based documents to process. Each must have a valid file path in its metadata.
        :param extraction_mode: If set, overrides the instance extraction_mode for this run only.
        :returns:
            A dictionary with:
            - "documents": Successfully processed documents (content or metadata updated).
            - "failed_documents": Documents that failed processing, annotated with failure metadata.
        """

        if not documents:
            return {"documents": [], "failed_documents": []}

        if not self._is_warmed_up:
            self.warm_up()

        mode: ExtractionMode = extraction_mode if extraction_mode is not None else self.extraction_mode
        prompt = self.content_prompt if mode == "content" else self.metadata_prompt
        error_meta_key = "content_extraction_error" if mode == "content" else "metadata_extraction_error"

        image_contents = self._document_to_image_content.run(documents=documents)["image_contents"]
        all_messages: list[ChatMessage | None] = []
        for image_content in image_contents:
            if image_content is None:
                all_messages.append(None)
                continue
            message = ChatMessage.from_user(content_parts=[TextContent(text=prompt), image_content])
            all_messages.append(message)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._run_on_thread, all_messages)

        successful_documents = []
        failed_documents = []
        for document, result in zip(documents, results):
            if "error" in result:
                new_meta = {**document.meta, error_meta_key: result["error"]}
                failed_documents.append(replace(document, meta=new_meta))
                continue

            new_meta = {**document.meta}
            new_meta.pop("content_extraction_error", None)
            new_meta.pop("metadata_extraction_error", None)
            extracted_text = result["replies"][0].text

            if mode == "content":
                successful_documents.append(replace(document, content=extracted_text, meta=new_meta))
            else:
                new_meta[self.metadata_key] = extracted_text
                successful_documents.append(replace(document, meta=new_meta))

        return {"documents": successful_documents, "failed_documents": failed_documents}
