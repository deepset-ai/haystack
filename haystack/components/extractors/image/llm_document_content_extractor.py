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


# Reserved key in the LLM JSON response that holds the main document text.
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

No markdown, no code fence, only raw JSON.

Document:"""


@component
class LLMDocumentContentExtractor:
    """
    Extracts textual content and optionally metadata from image-based documents using a vision-enabled LLM.

    One prompt and one LLM call per document. The component converts each document to an image via
    DocumentToImageContent and sends it to the ChatGenerator. The prompt must not contain Jinja variables.

    Response handling:
    - If the LLM returns a **plain string** (non-JSON or not a JSON object), it is written to the document's content.
    - If the LLM returns a **JSON object with only the key** `document_content`, that value is written to content.
    - If the LLM returns a **JSON object with multiple keys**, the value of ``document_content`` (if present) is
      written to content and all other keys are merged into the document's metadata.

    The ChatGenerator can be configured to return JSON (e.g. ``response_format={"type": "json_object"}``
    in ``generation_kwargs``).

    Documents that fail extraction are returned in ``failed_documents`` with ``content_extraction_error`` in metadata.

    ### Usage example
    ```python
    from haystack import Document
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.extractors.image import LLMDocumentContentExtractor

    chat_generator = OpenAIChatGenerator()
    extractor = LLMDocumentContentExtractor(chat_generator=chat_generator)
    documents = [
        Document(content="", meta={"file_path": "image.jpg"}),
        Document(content="", meta={"file_path": "document.pdf", "page_number": 1})
    ]
    result = extractor.run(documents=documents)
    updated_documents = result["documents"]
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        prompt: str = DEFAULT_PROMPT_TEMPLATE,
        file_path_meta_field: str = "file_path",
        root_path: str | None = None,
        detail: Literal["auto", "high", "low"] | None = None,
        size: tuple[int, int] | None = None,
        raise_on_failure: bool = False,
        max_workers: int = 3,
    ):
        """
        Initialize the LLMDocumentContentExtractor component.

        :param chat_generator: A ChatGenerator that supports vision input. Optionally configured for JSON
            (e.g. ``response_format={"type": "json_object"}`` in ``generation_kwargs``).
        :param prompt: Prompt for extraction. Must not contain Jinja variables.
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param detail: Optional detail level of the image (only supported by OpenAI). Can be "auto", "high", or "low".
        :param size: If provided, resizes the image to fit within (width, height) while keeping aspect ratio.
        :param raise_on_failure: If True, exceptions from the LLM are raised. If False, failed documents are returned.
        :param max_workers: Maximum number of threads for parallel LLM calls.
        """
        self._chat_generator = chat_generator
        self.prompt = prompt
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.detail = detail
        self.size = size
        LLMDocumentContentExtractor._validate_prompt_no_variables(prompt, "extract the content of")
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
            prompt=self.prompt,
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

        return default_from_dict(cls, data)

    @staticmethod
    def _validate_prompt_no_variables(prompt: str, context: str) -> None:
        ast = SandboxedEnvironment().parse(prompt)
        template_variables = meta.find_undeclared_variables(ast)
        variables = list(template_variables)
        if variables:
            raise ValueError(
                f"The prompt must not have any variables, only instructions on how to {context} "
                f"the image-based document. Found {','.join(variables)} in the prompt."
            )

    @staticmethod
    def _process_response(response_text: str) -> tuple[str | None, dict[str, Any], str | None]:
        """
        Parse LLM response. Returns (content, meta_updates, error).

        - Plain string (non-JSON): use entire response as document content;
        - Valid JSON object: use key ``document_content`` for Document.content and all other keys for Document.metadata;
        - Valid JSON but not an object (e.g. array or primitive), report an error;
        """
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text, {}, None

        if not isinstance(parsed, dict):
            return None, {}, "Response must be a JSON object, not an array or primitive."

        content = parsed.get(DOCUMENT_CONTENT_KEY)
        meta_updates = {k: v for k, v in parsed.items() if k != DOCUMENT_CONTENT_KEY}
        return content, meta_updates, None

    def _run_on_thread(self, image_content: ImageContent | None) -> dict[str, Any]:
        """
        Execute the LLM inference in a separate thread for each document.

        :param image_content: The image content for one document, or None if conversion failed.
        :returns:
            The LLM response if successful, or a dictionary with an "error" key on failure.
        """
        if image_content is None:
            return {"error": "Document has no content, skipping LLM call."}

        # the prompt is the same for all documents, so we can set it up once here for each document/thread
        message = ChatMessage.from_user(content_parts=[TextContent(text=self.prompt), image_content])

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

    @staticmethod
    def _process_llm_results(document: Document, result: dict[str, Any]) -> tuple[Document, bool]:
        """
        Process one document's LLM result using the unified response logic.

        Returns (updated_document, True if success else False).
        """
        if "error" in result:
            new_meta = {**document.meta, "content_extraction_error": result["error"]}
            return replace(document, meta=new_meta), False

        # remove potentially existing error metadata from previous runs
        new_meta = {**document.meta}
        new_meta.pop("content_extraction_error", None)
        new_meta.pop("metadata_extraction_error", None)
        new_meta.pop("metadata_extraction_response", None)

        # process the LLM response considering the possible response formats
        response_text = result["replies"][0].text
        content, meta_updates, content_error = LLMDocumentContentExtractor._process_response(response_text)

        if content_error:
            new_meta["content_extraction_error"] = content_error
            return replace(document, meta=new_meta), False

        new_meta.update(meta_updates)
        final_content = document.content if content is None else content
        return replace(document, content=final_content, meta=new_meta), True

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Run extraction on image-based documents. One LLM call per document.

        :param documents: A list of image-based documents to process. Each must have a valid file path in its metadata.
        :returns:
            A dictionary with "documents" (successfully processed) and "failed_documents" (with failure metadata).
        """
        if not documents:
            return {"documents": [], "failed_documents": []}

        if not self._is_warmed_up:
            self.warm_up()

        image_contents = self._document_to_image_content.run(documents=documents)["image_contents"]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._run_on_thread, image_contents)

        successful_documents = []
        failed_documents = []
        for document, result in zip(documents, results):
            doc, success = self._process_llm_results(document, result)
            if success:
                successful_documents.append(doc)
            else:
                failed_documents.append(doc)

        return {"documents": successful_documents, "failed_documents": failed_documents}
