# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import AzureOpenAIChatGenerator, OpenAIChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_callable, deserialize_secrets_inplace
from haystack.utils.utils import expand_page_range

with LazyImport(message="Run 'pip install \"amazon-bedrock-haystack>=1.0.2\"'") as amazon_bedrock_generator:
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

with LazyImport(message="Run 'pip install \"google-vertex-haystack>=2.0.0\"'") as vertex_ai_gemini_generator:
    from haystack_integrations.components.generators.google_vertex.chat.gemini import VertexAIGeminiChatGenerator
    from vertexai.generative_models import GenerationConfig


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """
    Currently LLM providers supported by `LLMMetadataExtractor`.
    """

    OPENAI = "openai"
    OPENAI_AZURE = "openai_azure"
    AWS_BEDROCK = "aws_bedrock"
    GOOGLE_VERTEX = "google_vertex"

    @staticmethod
    def from_str(string: str) -> "LLMProvider":
        """
        Convert a string to a LLMProvider enum.
        """
        provider_map = {e.value: e for e in LLMProvider}
        provider = provider_map.get(string)
        if provider is None:
            msg = f"Invalid LLMProvider '{string}'Supported LLMProviders are: {list(provider_map.keys())}"
            raise ValueError(msg)
        return provider


@component
class LLMMetadataExtractor:
    """
    Extracts metadata from documents using a Large Language Model (LLM).

    The metadata is extracted by providing a prompt to an LLM that generates the metadata.

    This component expects as input a list of documents and a prompt. The prompt should have a variable called
    `document` that will point to a single document in the list of documents. So to access the content of the document,
    you can use `{{ document.content }}` in the prompt.

    The component will run the LLM on each document in the list and extract metadata from the document. The metadata
    will be added to the document's metadata field. If the LLM fails to extract metadata from a document, the document
    will be added to the `failed_documents` list. The failed documents will have the keys `metadata_extraction_error` and
    `metadata_extraction_response` in their metadata. These documents can be re-run with another extractor to
    extract metadata by using the `metadata_extraction_response` and `metadata_extraction_error` in the prompt.

    ```python
    from haystack import Document
    from haystack_experimental.components.extractors.llm_metadata_extractor import LLMMetadataExtractor

    NER_PROMPT = '''
    -Goal-
    Given text and a list of entity types, identify all entities of those types from the text.

    -Steps-
    1. Identify all entities. For each identified entity, extract the following information:
    - entity_name: Name of the entity, capitalized
    - entity_type: One of the following types: [organization, product, service, industry]
    Format each entity as a JSON like: {"entity": <entity_name>, "entity_type": <entity_type>}

    2. Return output in a single list with all the entities identified in steps 1.

    -Examples-
    ######################
    Example 1:
    entity_types: [organization, person, partnership, financial metric, product, service, industry, investment strategy, market trend]
    text: Another area of strength is our co-brand issuance. Visa is the primary network partner for eight of the top
    10 co-brand partnerships in the US today and we are pleased that Visa has finalized a multi-year extension of
    our successful credit co-branded partnership with Alaska Airlines, a portfolio that benefits from a loyal customer
    base and high cross-border usage.
    We have also had significant co-brand momentum in CEMEA. First, we launched a new co-brand card in partnership
    with Qatar Airways, British Airways and the National Bank of Kuwait. Second, we expanded our strong global
    Marriott relationship to launch Qatar's first hospitality co-branded card with Qatar Islamic Bank. Across the
    United Arab Emirates, we now have exclusive agreements with all the leading airlines marked by a recent
    agreement with Emirates Skywards.
    And we also signed an inaugural Airline co-brand agreement in Morocco with Royal Air Maroc. Now newer digital
    issuers are equally
    ------------------------
    output:
    {"entities": [{"entity": "Visa", "entity_type": "company"}, {"entity": "Alaska Airlines", "entity_type": "company"}, {"entity": "Qatar Airways", "entity_type": "company"}, {"entity": "British Airways", "entity_type": "company"}, {"entity": "National Bank of Kuwait", "entity_type": "company"}, {"entity": "Marriott", "entity_type": "company"}, {"entity": "Qatar Islamic Bank", "entity_type": "company"}, {"entity": "Emirates Skywards", "entity_type": "company"}, {"entity": "Royal Air Maroc", "entity_type": "company"}]}
    #############################
    -Real Data-
    ######################
    entity_types: [company, organization, person, country, product, service]
    text: {{ document.content }}
    ######################
    output:
    '''

    docs = [
        Document(content="deepset was founded in 2018 in Berlin, and is known for its Haystack framework"),
        Document(content="Hugging Face is a company that was founded in New York, USA and is known for its Transformers library")
    ]

    extractor = LLMMetadataExtractor(
        prompt=NER_PROMPT,
        generator_api="openai",
        generator_api_params={
            "generation_kwargs": {
                "max_tokens": 500,
                "temperature": 0.0,
                "seed": 0,
                "response_format": {"type": "json_object"},
            },
            "max_retries": 1,
            "timeout": 60.0,
        },
        expected_keys=["entities"],
        raise_on_failure=False,
    )
    extractor.warm_up()
    extractor.run(documents=docs)
    >> {'documents': [
        Document(id=.., content: 'deepset was founded in 2018 in Berlin, and is known for its Haystack framework',
        meta: {'entities': [{'entity': 'deepset', 'entity_type': 'company'}, {'entity': 'Berlin', 'entity_type': 'city'},
              {'entity': 'Haystack', 'entity_type': 'product'}]}),
        Document(id=.., content: 'Hugging Face is a company that was founded in New York, USA and is known for its Transformers library',
        meta: {'entities': [
                {'entity': 'Hugging Face', 'entity_type': 'company'}, {'entity': 'New York', 'entity_type': 'city'},
                {'entity': 'USA', 'entity_type': 'country'}, {'entity': 'Transformers', 'entity_type': 'product'}
                ]})
           ]
        'failed_documents': []
       }
    >>
    ```
    """  # noqa: E501

    def __init__(  # pylint: disable=R0917
        self,
        prompt: str,
        generator_api: Union[str, LLMProvider],
        generator_api_params: Optional[Dict[str, Any]] = None,
        expected_keys: Optional[List[str]] = None,
        page_range: Optional[List[Union[str, int]]] = None,
        raise_on_failure: bool = False,
        max_workers: int = 3,
    ):
        """
        Initializes the LLMMetadataExtractor.

        :param prompt: The prompt to be used for the LLM.
        :param generator_api: The API provider for the LLM. Currently supported providers are:
                              "openai", "openai_azure", "aws_bedrock", "google_vertex"
        :param generator_api_params: The parameters for the LLM generator.
        :param expected_keys: The keys expected in the JSON output from the LLM.
        :param page_range: A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
                           metadata from the first and third pages of each document. It also accepts printable range
                           strings, e.g.: ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,
                           11, 12. If None, metadata will be extracted from the entire document for each document in the
                           documents list.
                           This parameter is optional and can be overridden in the `run` method.
        :param raise_on_failure: Whether to raise an error on failure during the execution of the Generator or
                                 validation of the JSON output.
        :param max_workers: The maximum number of workers to use in the thread pool executor.
        """
        self.prompt = prompt
        ast = SandboxedEnvironment().parse(prompt)
        template_variables = meta.find_undeclared_variables(ast)
        variables = list(template_variables)
        if len(variables) > 1 or variables[0] != "document":
            raise ValueError(
                f"Prompt must have exactly one variable called 'document'. Found {','.join(variables)} in the prompt."
            )
        self.builder = PromptBuilder(prompt, required_variables=variables)
        self.raise_on_failure = raise_on_failure
        self.expected_keys = expected_keys or []
        self.generator_api = (
            generator_api if isinstance(generator_api, LLMProvider) else LLMProvider.from_str(generator_api)
        )
        self.generator_api_params = generator_api_params or {}
        self.llm_provider = self._init_generator(self.generator_api, self.generator_api_params)
        self.splitter = DocumentSplitter(split_by="page", split_length=1)
        self.expanded_range = expand_page_range(page_range) if page_range else None
        self.max_workers = max_workers

    @staticmethod
    def _init_generator(
        generator_api: LLMProvider, generator_api_params: Optional[Dict[str, Any]]
    ) -> Union[
        OpenAIChatGenerator, AzureOpenAIChatGenerator, "AmazonBedrockChatGenerator", "VertexAIGeminiChatGenerator"
    ]:
        """
        Initialize the chat generator based on the specified API provider and parameters.
        """

        generator_api_params = generator_api_params or {}

        if generator_api == LLMProvider.OPENAI:
            return OpenAIChatGenerator(**generator_api_params)
        elif generator_api == LLMProvider.OPENAI_AZURE:
            return AzureOpenAIChatGenerator(**generator_api_params)
        elif generator_api == LLMProvider.AWS_BEDROCK:
            amazon_bedrock_generator.check()
            return AmazonBedrockChatGenerator(**generator_api_params)
        elif generator_api == LLMProvider.GOOGLE_VERTEX:
            vertex_ai_gemini_generator.check()
            return VertexAIGeminiChatGenerator(**generator_api_params)
        else:
            raise ValueError(f"Unsupported generator API: {generator_api}")

    def warm_up(self):
        """
        Warm up the LLM provider component.
        """
        if hasattr(self.llm_provider, "warm_up"):
            self.llm_provider.warm_up()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        llm_provider = self.llm_provider.to_dict()

        return default_to_dict(
            self,
            prompt=self.prompt,
            generator_api=self.generator_api.value,
            generator_api_params=llm_provider["init_parameters"],
            expected_keys=self.expected_keys,
            page_range=self.expanded_range,
            raise_on_failure=self.raise_on_failure,
            max_workers=self.max_workers,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMetadataExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """

        init_parameters = data.get("init_parameters", {})

        if "generator_api" in init_parameters:
            data["init_parameters"]["generator_api"] = LLMProvider.from_str(data["init_parameters"]["generator_api"])

        if "generator_api_params" in init_parameters:
            # Check all the keys that need to be deserialized
            azure_openai_keys = ["azure_ad_token"]
            aws_bedrock_keys = [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_region_name",
                "aws_profile_name",
            ]
            deserialize_secrets_inplace(
                data["init_parameters"]["generator_api_params"], keys=["api_key"] + azure_openai_keys + aws_bedrock_keys
            )

            # For VertexAI
            if "generation_config" in init_parameters["generator_api_params"]:
                data["init_parameters"]["generation_config"] = GenerationConfig.from_dict(
                    init_parameters["generator_api_params"]["generation_config"]
                )

            # For all
            serialized_callback_handler = init_parameters["generator_api_params"].get("streaming_callback")
            if serialized_callback_handler:
                data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)

    def _extract_metadata(self, llm_answer: str) -> Dict[str, Any]:
        try:
            parsed_metadata = json.loads(llm_answer)
        except json.JSONDecodeError as e:
            logger.warning(
                "Response from the LLM is not valid JSON. Skipping metadata extraction. Received output: {response}",
                response=llm_answer,
            )
            if self.raise_on_failure:
                raise e
            return {"error": "Response is not valid JSON. Received JSONDecodeError: " + str(e)}

        if not all(key in parsed_metadata for key in self.expected_keys):
            logger.warning(
                "Expected response from LLM to be a JSON with keys {expected_keys}, got {parsed_json}. "
                "Continuing extraction with received output.",
                expected_keys=self.expected_keys,
                parsed_json=parsed_metadata,
            )

        return parsed_metadata

    def _prepare_prompts(
        self, documents: List[Document], expanded_range: Optional[List[int]] = None
    ) -> List[Union[ChatMessage, None]]:
        all_prompts: List[Union[ChatMessage, None]] = []
        for document in documents:
            if not document.content:
                logger.warning("Document {doc_id} has no content. Skipping metadata extraction.", doc_id=document.id)
                all_prompts.append(None)
                continue

            if expanded_range:
                doc_copy = copy.deepcopy(document)
                pages = self.splitter.run(documents=[doc_copy])
                content = ""
                for idx, page in enumerate(pages["documents"]):
                    if idx + 1 in expanded_range:
                        content += page.content
                doc_copy.content = content
            else:
                doc_copy = document

            prompt_with_doc = self.builder.run(template=self.prompt, template_variables={"document": doc_copy})

            # build a ChatMessage with the prompt
            message = ChatMessage.from_user(prompt_with_doc["prompt"])
            all_prompts.append(message)

        return all_prompts

    def _run_on_thread(self, prompt: Optional[ChatMessage]) -> Dict[str, Any]:
        # If prompt is None, return an empty dictionary
        if prompt is None:
            return {"replies": ["{}"]}

        try:
            result = self.llm_provider.run(messages=[prompt])
        except Exception as e:
            logger.error(
                "LLM {class_name} execution failed. Skipping metadata extraction. Failed with exception '{error}'.",
                class_name=self.llm_provider.__class__.__name__,
                error=e,
            )
            if self.raise_on_failure:
                raise e
            result = {"error": "LLM failed with exception: " + str(e)}
        return result

    @component.output_types(documents=List[Document], failed_documents=List[Document])
    def run(self, documents: List[Document], page_range: Optional[List[Union[str, int]]] = None):
        """
        Extract metadata from documents using a Large Language Model.

        If `page_range` is provided, the metadata will be extracted from the specified range of pages. This component
        will split the documents into pages and extract metadata from the specified range of pages. The metadata will be
        extracted from the entire document if `page_range` is not provided.

        The original documents will be returned  updated with the extracted metadata.

        :param documents: List of documents to extract metadata from.
        :param page_range: A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
                           metadata from the first and third pages of each document. It also accepts printable range
                           strings, e.g.: ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,
                           11, 12.
                           If None, metadata will be extracted from the entire document for each document in the
                           documents list.
        :returns:
            A dictionary with the keys:
            - "documents": A list of documents that were successfully updated with the extracted metadata.
            - "failed_documents": A list of documents that failed to extract metadata. These documents will have
            "metadata_extraction_error" and "metadata_extraction_response" in their metadata. These documents can be
            re-run with the extractor to extract metadata.
        """
        if len(documents) == 0:
            logger.warning("No documents provided. Skipping metadata extraction.")
            return {"documents": [], "failed_documents": []}

        expanded_range = self.expanded_range
        if page_range:
            expanded_range = expand_page_range(page_range)

        # Create ChatMessage prompts for each document
        all_prompts = self._prepare_prompts(documents=documents, expanded_range=expanded_range)

        # Run the LLM on each prompt
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._run_on_thread, all_prompts)

        successful_documents = []
        failed_documents = []
        for document, result in zip(documents, results):
            if "error" in result:
                document.meta["metadata_extraction_error"] = result["error"]
                document.meta["metadata_extraction_response"] = None
                failed_documents.append(document)
                continue

            parsed_metadata = self._extract_metadata(result["replies"][0].text)
            if "error" in parsed_metadata:
                document.meta["metadata_extraction_error"] = parsed_metadata["error"]
                document.meta["metadata_extraction_response"] = result["replies"][0]
                failed_documents.append(document)
                continue

            for key in parsed_metadata:
                document.meta[key] = parsed_metadata[key]
                # Remove metadata_extraction_error and metadata_extraction_response if present from previous runs
                document.meta.pop("metadata_extraction_error", None)
                document.meta.pop("metadata_extraction_response", None)
            successful_documents.append(document)

        return {"documents": successful_documents, "failed_documents": failed_documents}
