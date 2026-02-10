# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Literal

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.image.document_to_image import DocumentToImageContent
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ImageContent, TextContent
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)


ExtractionMode = Literal["content", "metadata", "both"]

# Reserved key in the LLM JSON response that holds the main document text (used in content and both modes).
DOCUMENT_CONTENT_KEY = "document_content"


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

Return a single JSON object. It must contain the key "document_content" with the extracted text as value.
You may include other keys for metadata. No markdown, no code fence, only raw JSON.

Document:"""


DEFAULT_METADATA_PROMPT_TEMPLATE = """
You are part of an information extraction pipeline that extracts metadata from image-based documents.

Look at the provided image and extract metadata that describes the document. Return only a JSON object.

Extract whatever is visible and relevant, such as:
- title: document or page title if visible
- author: author or creator if visible
- date: publication date, creation date, or any date shown
- document_type: letter, invoice, form, report, receipt, etc.
- summary: one or two sentences describing what the document is about
- keywords: list of important topics or entities (e.g. names, organizations, amounts, dates)

Return a single JSON object with string values, e.g.:
{"title": "Example Document", "author": "John Doe", "date": "2024-01-15", "document_type": "invoice"}

If a field is not visible or not applicable, omit it. Be concise. No markdown, no code fence, only raw JSON.

Document:"""


def _parse_json_response(llm_answer: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Parse LLM response as JSON.

    Returns (parsed_dict, None) on success, or (None, error_message) on parse failure. Assumes the ChatGenerator is
    configured for JSON output.
    """
    try:
        parsed = json.loads(llm_answer)
    except json.JSONDecodeError as e:
        return None, "Response is not valid JSON. Received JSONDecodeError: " + str(e)
    if not isinstance(parsed, dict):
        return None, "Response must be a JSON object, not an array or primitive."
    return parsed, None


def _validate_prompt_no_variables(prompt: str, context: str) -> None:
    ast = SandboxedEnvironment().parse(prompt)
    template_variables = meta.find_undeclared_variables(ast)
    variables = list(template_variables)
    if variables:
        raise ValueError(
            f"The prompt must not have any variables, only instructions on how to {context} "
            f"the image-based document. Found {','.join(variables)} in the prompt."
        )


def _build_messages(prompt: str, image_contents: list[ImageContent | None]) -> list[ChatMessage | None]:
    """Build a list of ChatMessages (prompt + image) per image_content; None image yields None message."""
    messages: list[ChatMessage | None] = []
    for image_content in image_contents:
        if image_content is None:
            # If the image content is None, it means the document could not be converted to an image.
            # We skip this document.
            # We don't log a warning here since it is already logged in the DocumentToImageContent component.
            messages.append(None)
            continue
        messages.append(ChatMessage.from_user(content_parts=[TextContent(text=prompt), image_content]))
    return messages


def _cleaned_document_meta(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of meta with extraction-error keys removed (for a fresh run)."""
    out = {**metadata}
    out.pop("content_extraction_error", None)
    out.pop("metadata_extraction_error", None)
    out.pop("metadata_extraction_response", None)
    return out


def _process_content_response(response_text: str) -> tuple[str | None, dict[str, Any], str | None]:
    """
    Parse content-mode JSON response. Returns (content, meta_updates, error).

    On success: error is None, content is the document_content value, meta_updates are remaining keys.
    On failure: error is set, content is None, meta_updates is empty.
    """
    parsed, parse_error = _parse_json_response(response_text)
    if parse_error:
        return None, {}, parse_error
    if parsed is None:
        return None, {}, "Invalid response"
    if DOCUMENT_CONTENT_KEY not in parsed:
        return None, {}, f"JSON response must contain the key '{DOCUMENT_CONTENT_KEY}'."
    content = parsed.pop(DOCUMENT_CONTENT_KEY)
    return content, parsed, None


@component
class LLMDocumentContentExtractor:
    """
    Extracts textual content or metadata from image-based documents using a vision-enabled Large-Language Model (LLM).

    The ChatGenerator must be configured to return JSON (e.g. ``response_format={"type": "json_object"}``
    in ``generation_kwargs``). All three modes expect a JSON object from the LLM.

    - **content**: The LLM returns a JSON object that must contain the reserved key ``document_content``
      with the extracted text. That value is written to the document's content. Any other keys in the
      JSON are merged into the document's metadata.

    - **metadata**: The LLM returns a JSON object; every key is merged into the document's metadata.
      When ``expected_keys`` is set, a warning is logged if the response is missing any of those keys.

    - **both**: Two LLM calls per document. The first response must be JSON with ``document_content``
      (used for document content) and may include other keys (merged into metadata). The second
      response is JSON whose keys are merged into metadata.

    One LLM call is made per document in "content" or "metadata" mode; two in "both" mode. The
    component converts each document to an image via DocumentToImageContent and sends it to the
    ChatGenerator.

    Prompts must not contain Jinja variables; they should only include instructions for the LLM.

    Documents for which the LLM fails to extract content or metadata are returned in a separate `failed_documents`
    list. Failed documents will have `content_extraction_error` and/or `metadata_extraction_error` in their metadata;
    when metadata extraction fails (e.g. invalid JSON), `metadata_extraction_response` is also set with the raw LLM
    reply for debugging or reprocessing.

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
    # result["documents"][0].meta is updated with extracted keys (e.g. title, author, document_type)

    # Override to content for this run only
    result = extractor.run(documents=documents, extraction_mode="content")
    ```

    ### Usage example (both modes)
    ```python
    extractor = LLMDocumentContentExtractor(chat_generator=chat_generator, extraction_mode="both")
    result = extractor.run(documents=documents)
    # result["documents"][0].content has extracted text, result["documents"][0].meta has merged metadata
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        extraction_mode: ExtractionMode = "content",
        prompt: str = DEFAULT_PROMPT_TEMPLATE,
        metadata_prompt: str = DEFAULT_METADATA_PROMPT_TEMPLATE,
        expected_keys: list[str] | None = None,
        file_path_meta_field: str = "file_path",
        root_path: str | None = None,
        detail: Literal["auto", "high", "low"] | None = None,
        size: tuple[int, int] | None = None,
        raise_on_failure: bool = False,
        max_workers: int = 3,
    ):
        """
        Initialize the LLMDocumentContentExtractor component.

        :param chat_generator: A ChatGenerator instance that supports vision input and is configured to return JSON
            (e.g. ``response_format={"type": "json_object"}`` in ``generation_kwargs``). Content mode expects a key
            ``document_content`` in the JSON; metadata mode merges all keys into document metadata.
        :param extraction_mode: Where to write the LLM output: "content" (default), "metadata", or "both". Can be
            overridden in run().
        :param prompt: Prompt used when extraction_mode is "content". Must not contain Jinja variables.
        :param metadata_prompt: Prompt used when extraction_mode is "metadata" or "both". Must not contain Jinja
            variables. The LLM returns a JSON object; its keys are merged into the document's metadata.
        :param expected_keys: The keys expected in the JSON output from the LLM when extracting metadata (used in
            "metadata" and "both" modes). If provided and the LLM response is missing any of these keys, a warning
            is logged but extraction continues with the received output.
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
        self.prompt = prompt
        self.metadata_prompt = metadata_prompt
        self.expected_keys = expected_keys or []
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
            prompt=self.prompt,
            metadata_prompt=self.metadata_prompt,
            expected_keys=self.expected_keys,
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
        init_params = data.get("init_parameters", data)
        deserialize_chatgenerator_inplace(init_params, key="chat_generator")
        init_params.pop("metadata_key", None)
        return default_from_dict(cls, data)

    def _extract_metadata(self, llm_answer: str) -> dict[str, Any]:
        """
        Parse LLM metadata response as JSON.

        Returns parsed dict (all keys merged into document meta), or dict with "error" key on failure. Assumes
        ChatGenerator is configured for JSON output.
        """
        parsed_metadata, parse_error = _parse_json_response(llm_answer)
        if parse_error:
            logger.warning(
                "Response from the LLM is not valid JSON. Skipping metadata extraction. Received output: {response}",
                response=llm_answer,
            )
            if self.raise_on_failure:
                raise ValueError(parse_error)
            return {"error": parse_error}

        if parsed_metadata is None:
            return {}

        if not all(key in parsed_metadata for key in self.expected_keys):
            logger.warning(
                "Expected response from LLM to be a JSON with keys {expected_keys}, got {parsed_json}. "
                "Continuing extraction with received output.",
                expected_keys=self.expected_keys,
                parsed_json=parsed_metadata,
            )

        return parsed_metadata

    def _apply_metadata_response(self, response_text: str, new_meta: dict[str, Any], reply: ChatMessage) -> bool:
        """
        Parse metadata JSON and merge into new_meta, or set error keys on failure.

        Returns True if merged successfully, False if parse failed (new_meta updated with error).
        """
        parsed = self._extract_metadata(response_text)
        if "error" in parsed:
            new_meta["metadata_extraction_error"] = parsed["error"]
            new_meta["metadata_extraction_response"] = reply
            return False
        for key in parsed:
            new_meta[key] = parsed[key]
        return True

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

    def _process_single_mode_document(
        self, document: Document, result: dict[str, Any], mode: Literal["content", "metadata"]
    ) -> tuple[Document, bool]:
        """
        Process one document's LLM result in content or metadata mode.

        Returns (updated_document, True if success else False).
        """
        error_meta_key = "content_extraction_error" if mode == "content" else "metadata_extraction_error"
        if "error" in result:
            return replace(document, meta={**document.meta, error_meta_key: result["error"]}), False

        new_meta = _cleaned_document_meta(document.meta)
        response_text = result["replies"][0].text
        reply = result["replies"][0]

        if mode == "content":
            content, meta_updates, content_error = _process_content_response(response_text)
            if content_error:
                new_meta["content_extraction_error"] = content_error
                return replace(document, meta=new_meta), False
            new_meta.update(meta_updates)
            return replace(document, content=content, meta=new_meta), True

        if self._apply_metadata_response(response_text, new_meta, reply):
            return replace(document, meta=new_meta), True
        return replace(document, meta=new_meta), False

    def _process_both_mode_document(
        self, document: Document, content_result: dict[str, Any], metadata_result: dict[str, Any]
    ) -> tuple[Document, bool]:
        """
        Process one document's content + metadata LLM results in "both" mode.

        Returns (updated_document, True if at least one extraction succeeded).
        """
        new_meta = _cleaned_document_meta(document.meta)
        content_error = content_result.get("error")
        metadata_error = metadata_result.get("error")

        if content_error:
            new_meta["content_extraction_error"] = content_error
        if metadata_error:
            new_meta["metadata_extraction_error"] = metadata_error

        if content_error and metadata_error:
            return replace(document, meta=new_meta), False

        final_content = document.content
        if not content_error:
            content, meta_updates, content_err = _process_content_response(content_result["replies"][0].text)
            if content_err:
                new_meta["content_extraction_error"] = content_err
            else:
                final_content = content
                new_meta.update(meta_updates)

        if not metadata_error:
            self._apply_metadata_response(metadata_result["replies"][0].text, new_meta, metadata_result["replies"][0])

        return replace(document, content=final_content, meta=new_meta), True

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    def run(
        self, documents: list[Document], extraction_mode: ExtractionMode | None = None
    ) -> dict[str, list[Document]]:
        """
        Run extraction on image-based documents.

        One LLM call per document in "content" or "metadata" mode; two LLM calls per document in "both" mode.
        Note that the output goes to content and/or metadata according to the extraction mode.

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
        image_contents = self._document_to_image_content.run(documents=documents)["image_contents"]

        if mode == "both":
            return self._run_both(documents=documents, image_contents=image_contents)
        # mode here can only be "content" or "metadata"
        return self._run_single(
            documents=documents,
            image_contents=image_contents,
            # mode=cast(Literal["content", "metadata"], mode),
            mode=mode,
        )

    def _run_single(
        self, documents: list[Document], image_contents: list[ImageContent | None], mode: Literal["content", "metadata"]
    ) -> dict[str, list[Document]]:
        """Run extraction in content-only or metadata-only mode (one LLM call per document)."""
        prompt = self.prompt if mode == "content" else self.metadata_prompt
        all_messages = _build_messages(prompt, image_contents)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._run_on_thread, all_messages)

        successful_documents = []
        failed_documents = []
        for document, result in zip(documents, results):
            doc, success = self._process_single_mode_document(document, result, mode)
            if success:
                successful_documents.append(doc)
            else:
                failed_documents.append(doc)

        return {"documents": successful_documents, "failed_documents": failed_documents}

    def _run_both(
        self, documents: list[Document], image_contents: list[ImageContent | None]
    ) -> dict[str, list[Document]]:
        """Run content and metadata extraction (two LLM calls per document), then merge results."""
        content_messages = _build_messages(self.prompt, image_contents)
        metadata_messages = _build_messages(self.metadata_prompt, image_contents)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            content_results = list(executor.map(self._run_on_thread, content_messages))
            metadata_results = list(executor.map(self._run_on_thread, metadata_messages))

        successful_documents = []
        failed_documents = []
        for document, content_result, metadata_result in zip(documents, content_results, metadata_results):
            doc, success = self._process_both_mode_document(document, content_result, metadata_result)
            if success:
                successful_documents.append(doc)
            else:
                failed_documents.append(doc)

        return {"documents": successful_documents, "failed_documents": failed_documents}
