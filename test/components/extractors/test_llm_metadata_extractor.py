import os

import pytest
from unittest.mock import Mock
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore


from haystack.components.extractors import LLMMetadataExtractor, LLMProvider
from haystack.components.generators.chat import OpenAIChatGenerator


class TestLLMMetadataExtractor:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="prompt {{document.content}}", expected_keys=["key1", "key2"], generator_api=LLMProvider.OPENAI
        )
        assert isinstance(extractor.builder, PromptBuilder)
        assert isinstance(extractor._chat_generator, OpenAIChatGenerator)
        assert extractor.expected_keys == ["key1", "key2"]
        assert extractor.raise_on_failure is False

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="prompt {{document.content}}",
            expected_keys=["key1", "key2"],
            raise_on_failure=True,
            generator_api=LLMProvider.OPENAI,
            generator_api_params={"model": "gpt-4o-mini", "generation_kwargs": {"temperature": 0.5}},
            page_range=["1-5"],
        )
        assert isinstance(extractor.builder, PromptBuilder)
        assert extractor.expected_keys == ["key1", "key2"]
        assert extractor.raise_on_failure is True
        assert isinstance(extractor._chat_generator, OpenAIChatGenerator)
        assert extractor._chat_generator.model == "gpt-4o-mini"
        assert extractor._chat_generator.generation_kwargs == {"temperature": 0.5}
        assert extractor.expanded_range == [1, 2, 3, 4, 5]

    def test_init_with_chat_generator(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.5})

        extractor = LLMMetadataExtractor(
            prompt="prompt {{document.content}}", expected_keys=["key1", "key2"], chat_generator=chat_generator
        )
        assert isinstance(extractor._chat_generator, OpenAIChatGenerator)
        assert extractor._chat_generator.model == "gpt-4o-mini"
        assert extractor._chat_generator.generation_kwargs == {"temperature": 0.5}
        assert extractor.expected_keys == ["key1", "key2"]

    def test_init_missing_prompt_variable(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        with pytest.raises(ValueError):
            _ = LLMMetadataExtractor(
                prompt="prompt {{ wrong_variable }}", expected_keys=["key1", "key2"], generator_api=LLMProvider.OPENAI
            )

    def test_init_fails_without_generator_api_or_chat_generator(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        with pytest.raises(ValueError):
            _ = LLMMetadataExtractor(prompt="prompt {{document.content}}", expected_keys=["key1", "key2"])

    def test_init_chat_generator_overrides_generator_api(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.5})
        extractor = LLMMetadataExtractor(
            prompt="prompt {{document.content}}",
            expected_keys=["key1", "key2"],
            generator_api=LLMProvider.AWS_BEDROCK,
            chat_generator=chat_generator,
        )

        assert isinstance(extractor._chat_generator, OpenAIChatGenerator)
        assert extractor._chat_generator.model == "gpt-4o-mini"
        assert extractor._chat_generator.generation_kwargs == {"temperature": 0.5}

    def test_to_dict_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="some prompt that was used with the LLM {{document.content}}",
            expected_keys=["key1", "key2"],
            generator_api=LLMProvider.OPENAI,
            generator_api_params={"model": "gpt-4o-mini", "generation_kwargs": {"temperature": 0.5}},
            raise_on_failure=True,
        )
        extractor_dict = extractor.to_dict()

        assert extractor_dict == {
            "type": "haystack.components.extractors.llm_metadata_extractor.LLMMetadataExtractor",
            "init_parameters": {
                "chat_generator": {
                    "init_parameters": {
                        "api_base_url": None,
                        "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                        "generation_kwargs": {"temperature": 0.5},
                        "max_retries": None,
                        "model": "gpt-4o-mini",
                        "organization": None,
                        "streaming_callback": None,
                        "timeout": None,
                        "tools": None,
                        "tools_strict": False,
                    },
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                },
                "prompt": "some prompt that was used with the LLM {{document.content}}",
                "expected_keys": ["key1", "key2"],
                "raise_on_failure": True,
                "page_range": None,
                "max_workers": 3,
            },
        }

    def test_to_dict_openai_using_chat_generator(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.5})
        extractor = LLMMetadataExtractor(
            prompt="some prompt that was used with the LLM {{document.content}}",
            expected_keys=["key1", "key2"],
            chat_generator=chat_generator,
            raise_on_failure=True,
        )
        extractor_dict = extractor.to_dict()

        assert extractor_dict == {
            "type": "haystack.components.extractors.llm_metadata_extractor.LLMMetadataExtractor",
            "init_parameters": {
                "prompt": "some prompt that was used with the LLM {{document.content}}",
                "expected_keys": ["key1", "key2"],
                "raise_on_failure": True,
                "chat_generator": chat_generator.to_dict(),
                "page_range": None,
                "max_workers": 3,
            },
        }

    def test_from_dict_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        extractor_dict = {
            "type": "haystack.components.extractors.llm_metadata_extractor.LLMMetadataExtractor",
            "init_parameters": {
                "prompt": "some prompt that was used with the LLM {{document.content}}",
                "expected_keys": ["key1", "key2"],
                "raise_on_failure": True,
                "generator_api": "openai",
                "generator_api_params": {
                    "api_base_url": None,
                    "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                    "generation_kwargs": {},
                    "model": "gpt-4o-mini",
                    "organization": None,
                    "streaming_callback": None,
                },
            },
        }
        extractor = LLMMetadataExtractor.from_dict(extractor_dict)
        assert extractor.raise_on_failure is True
        assert extractor.expected_keys == ["key1", "key2"]
        assert extractor.prompt == "some prompt that was used with the LLM {{document.content}}"
        assert isinstance(extractor._chat_generator, OpenAIChatGenerator)
        assert extractor._chat_generator.model == "gpt-4o-mini"

    def test_from_dict_openai_using_chat_generator(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.5})

        extractor_dict = {
            "type": "haystack.components.extractors.llm_metadata_extractor.LLMMetadataExtractor",
            "init_parameters": {
                "prompt": "some prompt that was used with the LLM {{document.content}}",
                "expected_keys": ["key1", "key2"],
                "chat_generator": chat_generator.to_dict(),
                "raise_on_failure": True,
            },
        }
        extractor = LLMMetadataExtractor.from_dict(extractor_dict)
        assert extractor.raise_on_failure is True
        assert extractor.expected_keys == ["key1", "key2"]
        assert extractor.prompt == "some prompt that was used with the LLM {{document.content}}"
        assert extractor._chat_generator.to_dict() == chat_generator.to_dict()

    def test_warm_up_with_chat_generator(self, monkeypatch):
        mock_chat_generator = Mock()

        extractor = LLMMetadataExtractor(prompt="prompt {{document.content}}", chat_generator=mock_chat_generator)

        mock_chat_generator.warm_up.assert_not_called()

        extractor.warm_up()

        mock_chat_generator.warm_up.assert_called_once()

    def test_extract_metadata(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(prompt="prompt {{document.content}}", generator_api=LLMProvider.OPENAI)
        result = extractor._extract_metadata(llm_answer='{"output": "valid json"}')
        assert result == {"output": "valid json"}

    def test_extract_metadata_invalid_json(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="prompt {{document.content}}", generator_api=LLMProvider.OPENAI, raise_on_failure=True
        )
        with pytest.raises(ValueError):
            extractor._extract_metadata(llm_answer='{"output: "valid json"}')

    def test_extract_metadata_missing_key(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="prompt {{document.content}}", generator_api=LLMProvider.OPENAI, expected_keys=["key1"]
        )
        extractor._extract_metadata(llm_answer='{"output": "valid json"}')
        assert "Expected response from LLM to be a JSON with keys" in caplog.text

    def test_prepare_prompts(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="some_user_definer_prompt {{document.content}}", generator_api=LLMProvider.OPENAI
        )
        docs = [
            Document(content="deepset was founded in 2018 in Berlin, and is known for its Haystack framework"),
            Document(
                content="Hugging Face is a company founded in Paris, France and is known for its Transformers library"
            ),
        ]
        prompts = extractor._prepare_prompts(docs)

        assert prompts == [
            ChatMessage.from_dict(
                {
                    "_role": "user",
                    "_meta": {},
                    "_name": None,
                    "_content": [
                        {
                            "text": "some_user_definer_prompt deepset was founded in 2018 in Berlin, and is known for its Haystack framework"
                        }
                    ],
                }
            ),
            ChatMessage.from_dict(
                {
                    "_role": "user",
                    "_meta": {},
                    "_name": None,
                    "_content": [
                        {
                            "text": "some_user_definer_prompt Hugging Face is a company founded in Paris, France and is known for its Transformers library"
                        }
                    ],
                }
            ),
        ]

    def test_prepare_prompts_empty_document(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="some_user_definer_prompt {{document.content}}", generator_api=LLMProvider.OPENAI
        )
        docs = [
            Document(content=""),
            Document(
                content="Hugging Face is a company founded in Paris, France and is known for its Transformers library"
            ),
        ]
        prompts = extractor._prepare_prompts(docs)
        assert prompts == [
            None,
            ChatMessage.from_dict(
                {
                    "_role": "user",
                    "_meta": {},
                    "_name": None,
                    "_content": [
                        {
                            "text": "some_user_definer_prompt Hugging Face is a company founded in Paris, France and is known for its Transformers library"
                        }
                    ],
                }
            ),
        ]

    def test_prepare_prompts_expanded_range(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="some_user_definer_prompt {{document.content}}", generator_api=LLMProvider.OPENAI, page_range=["1-2"]
        )
        docs = [
            Document(
                content="Hugging Face is a company founded in Paris, France and is known for its Transformers library\fPage 2\fPage 3"
            )
        ]
        prompts = extractor._prepare_prompts(docs, expanded_range=[1, 2])

        assert prompts == [
            ChatMessage.from_dict(
                {
                    "_role": "user",
                    "_meta": {},
                    "_name": None,
                    "_content": [
                        {
                            "text": "some_user_definer_prompt Hugging Face is a company founded in Paris, France and is known for its Transformers library\x0cPage 2\x0c"
                        }
                    ],
                }
            )
        ]

    def test_run_no_documents(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(prompt="prompt {{document.content}}", generator_api=LLMProvider.OPENAI)
        result = extractor.run(documents=[])
        assert result["documents"] == []
        assert result["failed_documents"] == []

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run(self):
        docs = [
            Document(content="deepset was founded in 2018 in Berlin, and is known for its Haystack framework"),
            Document(
                content="Hugging Face is a company founded in Paris, France and is known for its Transformers library"
            ),
        ]

        ner_prompt = """-Goal-
Given text and a list of entity types, identify all entities of those types from the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [organization, product, service, industry]
Format each entity as {"entity": <entity_name>, "entity_type": <entity_type>}

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
"""

        doc_store = InMemoryDocumentStore()
        extractor = LLMMetadataExtractor(
            prompt=ner_prompt, expected_keys=["entities"], chat_generator=OpenAIChatGenerator()
        )
        writer = DocumentWriter(document_store=doc_store)
        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("doc_writer", writer)
        pipeline.connect("extractor.documents", "doc_writer.documents")
        pipeline.run(data={"documents": docs})

        doc_store_docs = doc_store.filter_documents()
        assert len(doc_store_docs) == 2
        assert "entities" in doc_store_docs[0].meta
        assert "entities" in doc_store_docs[1].meta
