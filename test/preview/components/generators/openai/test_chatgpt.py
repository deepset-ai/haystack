from unittest.mock import MagicMock, patch

import pytest

from haystack.preview.components.generators.openai.chatgpt import ChatGPTGenerator
from haystack.preview.components.generators.openai.chatgpt import default_streaming_callback


class TestChatGPTGenerator:
    @pytest.mark.unit
    def test_init_default(self, caplog):
        with patch("haystack.preview.components.generators.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTGenerator()
            assert component.api_key is None
            assert component.model_name == "gpt-3.5-turbo"
            assert component.system_prompt == "You are a helpful assistant."
            assert component.max_reply_tokens == 500
            assert component.temperature == 0.7
            assert component.top_p == 1
            assert component.n == 1
            assert component.stop is None
            assert component.presence_penalty == 0
            assert component.frequency_penalty == 0
            assert component.logit_bias == None
            assert component.moderate_content is True
            assert component.stream is False
            assert component.streaming_callback == default_streaming_callback
            assert component.streaming_done_marker == "[DONE]"
            assert component.api_base_url == "https://api.openai.com/v1"
            assert component.openai_organization is None
            assert component.max_tokens_limit == 2049

            tiktoken_patch.get_encoding.assert_called_once_with("cl100k_base")
            assert caplog.records[0].message == (
                "OpenAI API key is missing. You will need to provide an API key to Pipeline.run()."
            )

    @pytest.mark.unit
    def test_init_with_parameters(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.TOKENIZERS", {"test-model-name": "test-encoding"}
        )
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.TOKENIZERS_TOKEN_LIMITS", {"test-model-name": 10}
        )
        with patch("haystack.preview.components.generators.openai.chatgpt.tiktoken") as tiktoken_patch:
            callback = lambda x: x
            component = ChatGPTGenerator(
                api_key="test-api-key",
                model_name="test-model-name",
                system_prompt="test-system-prompt",
                max_reply_tokens=20,
                temperature=1,
                top_p=5,
                n=10,
                stop=["test-stop-word"],
                presence_penalty=0.5,
                frequency_penalty=0.4,
                logit_bias={"test-logit-bias": 0.3},
                moderate_content=False,
                stream=True,
                streaming_callback=callback,
                streaming_done_marker="test-marker",
                api_base_url="test-base-url",
                openai_organization="test-orga-id",
            )
            assert component.api_key == "test-api-key"
            assert component.model_name == "test-model-name"
            assert component.system_prompt == "test-system-prompt"
            assert component.max_reply_tokens == 20
            assert component.temperature == 1
            assert component.top_p == 5
            assert component.n == 10
            assert component.stop == ["test-stop-word"]
            assert component.presence_penalty == 0.5
            assert component.frequency_penalty == 0.4
            assert component.logit_bias == {"test-logit-bias": 0.3}
            assert component.moderate_content is False
            assert component.stream is True
            assert component.streaming_callback == callback
            assert component.streaming_done_marker == "test-marker"
            assert component.api_base_url == "test-base-url"
            assert component.openai_organization == "test-orga-id"
            assert component.max_tokens_limit == 10

            tiktoken_patch.get_encoding.assert_called_once_with("test-encoding")
            assert not caplog.records

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        with patch("haystack.preview.components.generators.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTGenerator()
            data = component.to_dict()
            assert data == {
                "type": "ChatGPTGenerator",
                "init_parameters": {
                    "api_key": None,
                    "model_name": "gpt-3.5-turbo",
                    "system_prompt": "You are a helpful assistant.",
                    "max_reply_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 1,
                    "n": 1,
                    "stop": None,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "logit_bias": None,
                    "moderate_content": True,
                    "stream": False,
                    # FIXME serialize callback?
                    "streaming_done_marker": "[DONE]",
                    "api_base_url": "https://api.openai.com/v1",
                    "openai_organization": None,
                },
            }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.TOKENIZERS", {"test-model-name": "test-encoding"}
        )
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.TOKENIZERS_TOKEN_LIMITS", {"test-model-name": 10}
        )
        with patch("haystack.preview.components.generators.openai.chatgpt.tiktoken") as tiktoken_patch:
            callback = lambda x: x
            component = ChatGPTGenerator(
                api_key="test-api-key",
                model_name="test-model-name",
                system_prompt="test-system-prompt",
                max_reply_tokens=20,
                temperature=1,
                top_p=5,
                n=10,
                stop=["test-stop-word"],
                presence_penalty=0.5,
                frequency_penalty=0.4,
                logit_bias={"test-logit-bias": 0.3},
                moderate_content=False,
                stream=True,
                streaming_callback=callback,
                streaming_done_marker="test-marker",
                api_base_url="test-base-url",
                openai_organization="test-orga-id",
            )
            data = component.to_dict()
            assert data == {
                "type": "ChatGPTGenerator",
                "init_parameters": {
                    "api_key": "test-api-key",
                    "model_name": "test-model-name",
                    "system_prompt": "test-system-prompt",
                    "max_reply_tokens": 20,
                    "temperature": 1,
                    "top_p": 5,
                    "n": 10,
                    "stop": ["test-stop-word"],
                    "presence_penalty": 0.5,
                    "frequency_penalty": 0.4,
                    "logit_bias": {"test-logit-bias": 0.3},
                    "moderate_content": False,
                    "stream": True,
                    # FIXME serialize callback?
                    "streaming_done_marker": "test-marker",
                    "api_base_url": "test-base-url",
                    "openai_organization": "test-orga-id",
                },
            }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.TOKENIZERS", {"test-model-name": "test-encoding"}
        )
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.TOKENIZERS_TOKEN_LIMITS", {"test-model-name": 10}
        )
        with patch("haystack.preview.components.generators.openai.chatgpt.tiktoken") as tiktoken_patch:
            data = {
                "type": "ChatGPTGenerator",
                "init_parameters": {
                    "api_key": "test-api-key",
                    "model_name": "test-model-name",
                    "system_prompt": "test-system-prompt",
                    "max_reply_tokens": 20,
                    "temperature": 1,
                    "top_p": 5,
                    "n": 10,
                    "stop": ["test-stop-word"],
                    "presence_penalty": 0.5,
                    "frequency_penalty": 0.4,
                    "logit_bias": {"test-logit-bias": 0.3},
                    "moderate_content": False,
                    "stream": True,
                    # FIXME serialize callback?
                    "streaming_done_marker": "test-marker",
                    "api_base_url": "test-base-url",
                    "openai_organization": "test-orga-id",
                },
            }
            component = ChatGPTGenerator.from_dict(data)
            assert component.api_key == "test-api-key"
            assert component.model_name == "test-model-name"
            assert component.system_prompt == "test-system-prompt"
            assert component.max_reply_tokens == 20
            assert component.temperature == 1
            assert component.top_p == 5
            assert component.n == 10
            assert component.stop == ["test-stop-word"]
            assert component.presence_penalty == 0.5
            assert component.frequency_penalty == 0.4
            assert component.logit_bias == {"test-logit-bias": 0.3}
            assert component.moderate_content is False
            assert component.stream is True
            assert component.streaming_callback == default_streaming_callback
            assert component.streaming_done_marker == "test-marker"
            assert component.api_base_url == "test-base-url"
            assert component.openai_organization == "test-orga-id"
            assert component.max_tokens_limit == 10

    # @pytest.mark.unit
    # @patch(
    #     "haystack.preview.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"
    # )
    # def test_warmup(self, mocked_factory):
    #     embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
    #     mocked_factory.get_embedding_backend.assert_not_called()
    #     embedder.warm_up()
    #     mocked_factory.get_embedding_backend.assert_called_once_with(
    #         model_name_or_path="model", device="cpu", use_auth_token=None
    #     )

    # @pytest.mark.unit
    # @patch(
    #     "haystack.preview.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"
    # )
    # def test_warmup_doesnt_reload(self, mocked_factory):
    #     embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
    #     mocked_factory.get_embedding_backend.assert_not_called()
    #     embedder.warm_up()
    #     embedder.warm_up()
    #     mocked_factory.get_embedding_backend.assert_called_once()

    # @pytest.mark.unit
    # def test_run(self):
    #     embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
    #     embedder.embedding_backend = MagicMock()
    #     embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()

    #     documents = [Document(content=f"document number {i}") for i in range(5)]

    #     result = embedder.run(documents=documents)

    #     assert isinstance(result["documents"], list)
    #     assert len(result["documents"]) == len(documents)
    #     for doc in result["documents"]:
    #         assert isinstance(doc, Document)
    #         assert isinstance(doc.embedding, list)
    #         assert isinstance(doc.embedding[0], float)

    # @pytest.mark.unit
    # def test_run_wrong_input_format(self):
    #     embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")

    #     string_input = "text"
    #     list_integers_input = [1, 2, 3]

    #     with pytest.raises(
    #         TypeError, match="SentenceTransformersDocumentEmbedder expects a list of Documents as input"
    #     ):
    #         embedder.run(documents=string_input)

    #     with pytest.raises(
    #         TypeError, match="SentenceTransformersDocumentEmbedder expects a list of Documents as input"
    #     ):
    #         embedder.run(documents=list_integers_input)

    # @pytest.mark.unit
    # def test_embed_metadata(self):
    #     embedder = SentenceTransformersDocumentEmbedder(
    #         model_name_or_path="model", metadata_fields_to_embed=["meta_field"], embedding_separator="\n"
    #     )
    #     embedder.embedding_backend = MagicMock()

    #     documents = [
    #         Document(content=f"document number {i}", metadata={"meta_field": f"meta_value {i}"}) for i in range(5)
    #     ]

    #     embedder.run(documents=documents)

    #     embedder.embedding_backend.embed.assert_called_once_with(
    #         [
    #             "meta_value 0\ndocument number 0",
    #             "meta_value 1\ndocument number 1",
    #             "meta_value 2\ndocument number 2",
    #             "meta_value 3\ndocument number 3",
    #             "meta_value 4\ndocument number 4",
    #         ],
    #         batch_size=32,
    #         show_progress_bar=True,
    #         normalize_embeddings=False,
    #     )
