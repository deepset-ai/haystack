# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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


@component
class LLMDocumentContentExtractor:
    """
    Extracts textual content from image-based documents using a vision-enabled LLM (Large Language Model).

    This component converts each input document into an image using the DocumentToImageContent component,
    uses a prompt to instruct the LLM on how to extract content, and uses a ChatGenerator to extract structured
    textual content based on the provided prompt.

    The prompt must not contain variables; it should only include instructions for the LLM. Image data and the prompt
    are passed together to the LLM as a chat message.

    Documents for which the LLM fails to extract content are returned in a separate `failed_documents` list. These
    failed documents will have a `content_extraction_error` entry in their metadata. This metadata can be used for
    debugging or for reprocessing the documents later.

    ### Usage example
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
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        prompt: str = DEFAULT_PROMPT_TEMPLATE,
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None,
        raise_on_failure: bool = False,
        max_workers: int = 3,
    ):
        """
        Initialize the LLMDocumentContentExtractor component.

        :param chat_generator: A ChatGenerator instance representing the LLM used to extract text. This generator must
            support vision-based input and return a plain text response.
        :param prompt: Instructional text provided to the LLM. It must not contain Jinja variables.
            The prompt should only contain instructions on how to extract the content of the image-based document.
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
        self.prompt = prompt
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.detail = detail
        self.size = size
        # Ensure the prompt does not contain any variables.
        ast = SandboxedEnvironment().parse(prompt)
        template_variables = meta.find_undeclared_variables(ast)
        variables = list(template_variables)
        if len(variables) != 0:
            raise ValueError(
                f"The prompt must not have any variables only instructions on how to extract the content of the "
                f"image-based document. Found {','.join(variables)} in the prompt."
            )
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "LLMDocumentContentExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")
        return default_from_dict(cls, data)

    def _run_on_thread(self, message: Optional[ChatMessage]) -> Dict[str, Any]:
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

    @component.output_types(documents=List[Document], failed_documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Run content extraction on a list of image-based documents using a vision-capable LLM.

        Each document is passed to the LLM along with a predefined prompt. The response is used to update the document's
        content. If the extraction fails, the document is returned in the `failed_documents` list with metadata
        describing the failure.

        :param documents: A list of image-based documents to process. Each must have a valid file path in its metadata.
        :returns:
            A dictionary with:
            - "documents": Successfully processed documents, updated with extracted content.
            - "failed_documents": Documents that failed processing, annotated with failure metadata.
        """
        if not documents:
            return {"documents": [], "failed_documents": []}

        # Create ChatMessage prompts for each document
        image_contents = self._document_to_image_content.run(documents=documents)["image_contents"]
        all_messages: List[Union[ChatMessage, None]] = []
        for image_content in image_contents:
            if image_content is None:
                # If the image content is None, it means the document could not be converted to an image.
                # We skip this document.
                # We don't log a warning here since it is already logged in the DocumentToImageContent component.
                all_messages.append(None)
                continue
            message = ChatMessage.from_user(content_parts=[TextContent(text=self.prompt), image_content])
            all_messages.append(message)

        # Run the LLM on each message
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._run_on_thread, all_messages)

        successful_documents = []
        failed_documents = []
        for document, result in zip(documents, results):
            if "error" in result:
                new_meta = {**document.meta, "content_extraction_error": result["error"]}
                failed_documents.append(replace(document, meta=new_meta))
                continue

            # Remove content_extraction_error if present from previous runs
            new_meta = {**document.meta}
            new_meta.pop("content_extraction_error", None)

            extracted_content = result["replies"][0].text
            successful_documents.append(replace(document, content=extracted_content, meta=new_meta))

        return {"documents": successful_documents, "failed_documents": failed_documents}
