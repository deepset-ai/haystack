from abc import ABC, abstractmethod
import json
import logging
import re
from typing import Any, Optional, Dict, Type, Union, List

from haystack.errors import AWSConfigurationError, AmazonBedrockConfigurationError, AmazonBedrockInferenceError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer.aws_base import AWSBaseInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import (
    DefaultPromptHandler,
    DefaultTokenStreamingHandler,
    TokenStreamingHandler,
)

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    from botocore.exceptions import ClientError


class BedrockModelAdapter(ABC):
    """
    Base class for Amazon Bedrock model adapters.
    """

    def __init__(self, model_kwargs: Dict[str, Any], max_length: Optional[int]) -> None:
        self.model_kwargs = model_kwargs
        self.max_length = max_length

    @abstractmethod
    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """Prepares the body for the Amazon Bedrock request."""

    def get_responses(self, response_body: Dict[str, Any]) -> List[str]:
        """Extracts the responses from the Amazon Bedrock response."""
        completions = self._extract_completions_from_response(response_body)
        responses = [completion.lstrip() for completion in completions]
        return responses

    def get_stream_responses(self, stream, stream_handler: TokenStreamingHandler) -> List[str]:
        tokens: List[str] = []
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                decoded_chunk = json.loads(chunk["bytes"].decode("utf-8"))
                token = self._extract_token_from_stream(decoded_chunk)
                tokens.append(stream_handler(token, event_data=decoded_chunk))
        responses = ["".join(tokens).lstrip()]
        return responses

    def _get_params(self, inference_kwargs: Dict[str, Any], default_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges the default params with the inference kwargs and model kwargs.

        Includes param if it's in kwargs or its default is not None (i.e. it is actually defined).
        """
        kwargs = self.model_kwargs.copy()
        kwargs.update(inference_kwargs)
        return {
            param: kwargs.get(param, default)
            for param, default in default_params.items()
            if param in kwargs or default is not None
        }

    @abstractmethod
    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        """Extracts the responses from the Amazon Bedrock response."""

    @abstractmethod
    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """Extracts the token from a streaming chunk."""


class AnthropicClaudeAdapter(BedrockModelAdapter):
    """
    Model adapter for the Anthropic's Claude model.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "max_tokens_to_sample": self.max_length,
            "stop_sequences": ["\n\nHuman:"],
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": f"\n\nHuman: {prompt}\n\nAssistant:", **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        return [response_body["completion"]]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("completion", "")


class CohereCommandAdapter(BedrockModelAdapter):
    """
    Model adapter for the Cohere's Command model.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "max_tokens": self.max_length,
            "stop_sequences": None,
            "temperature": None,
            "p": None,
            "k": None,
            "return_likelihoods": None,
            "stream": None,
            "logit_bias": None,
            "num_generations": None,
            "truncate": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        responses = [generation["text"] for generation in response_body["generations"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("text", "")


class AI21LabsJurassic2Adapter(BedrockModelAdapter):
    """
    Model adapter for AI21 Labs' Jurassic 2 models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "maxTokens": self.max_length,
            "stopSequences": None,
            "temperature": None,
            "topP": None,
            "countPenalty": None,
            "presencePenalty": None,
            "frequencyPenalty": None,
            "numResults": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        responses = [completion["data"]["text"] for completion in response_body["completions"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        raise NotImplementedError("Streaming is not supported for AI21 Jurassic 2 models.")


class AmazonTitanAdapter(BedrockModelAdapter):
    """
    Model adapter for Amazon's Titan models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {"maxTokenCount": self.max_length, "stopSequences": None, "temperature": None, "topP": None}
        params = self._get_params(inference_kwargs, default_params)

        body = {"inputText": prompt, "textGenerationConfig": params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        responses = [result["outputText"] for result in response_body["results"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("outputText", "")


class MetaLlama2ChatAdapter(BedrockModelAdapter):
    """
    Model adapter for Meta's Llama 2 Chat models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {"max_gen_len": self.max_length, "temperature": None, "top_p": None}
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        return [response_body["generation"]]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("generation", "")


class AmazonBedrockInvocationLayer(AWSBaseInvocationLayer):
    """
    Invocation layer for Amazon Bedrock models.
    """

    SUPPORTED_MODEL_PATTERNS: Dict[str, Type[BedrockModelAdapter]] = {
        r"amazon.titan-text.*": AmazonTitanAdapter,
        r"ai21.j2.*": AI21LabsJurassic2Adapter,
        r"cohere.command.*": CohereCommandAdapter,
        r"anthropic.claude.*": AnthropicClaudeAdapter,
        r"meta.llama2.*": MetaLlama2ChatAdapter,
    }

    def __init__(
        self,
        model_name_or_path: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        max_length: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        self.max_length = max_length

        try:
            session = self.get_aws_session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_region_name=aws_region_name,
                aws_profile_name=aws_profile_name,
            )
            self.client = session.client("bedrock-runtime")
        except Exception as exception:
            raise AmazonBedrockConfigurationError(
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            ) from exception

        model_input_kwargs = kwargs
        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        model_max_length = kwargs.get("model_max_length", 4096)

        # Truncate prompt if prompt tokens > model_max_length-max_length
        # (max_length is the length of the generated text)
        # It is hard to determine which tokenizer to use for the SageMaker model
        # so we use GPT2 tokenizer which will likely provide good token count approximation
        self.prompt_handler = DefaultPromptHandler(
            model_name_or_path="gpt2", model_max_length=model_max_length, max_length=self.max_length or 100
        )

        model_apapter_cls = self.get_model_adapter(model_name_or_path=model_name_or_path)
        if not model_apapter_cls:
            raise AmazonBedrockConfigurationError(
                f"This invocation layer doesn't support the model {model_name_or_path}."
            )
        self.model_adapter = model_apapter_cls(model_kwargs=model_input_kwargs, max_length=self.max_length)

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        if isinstance(prompt, List):
            raise ValueError(
                "The SageMaker invocation layer only supports a string as a prompt, "
                "while currently, the prompt is a dictionary."
            )

        resize_info = self.prompt_handler(prompt)
        if resize_info["prompt_length"] != resize_info["new_prompt_length"]:
            logger.warning(
                "The prompt was truncated from %s tokens to %s tokens so that the prompt length and "
                "the answer length (%s tokens) fit within the model's max token limit (%s tokens). "
                "Shorten the prompt or it will be cut off.",
                resize_info["prompt_length"],
                max(0, resize_info["model_max_length"] - resize_info["max_length"]),  # type: ignore
                resize_info["max_length"],
                resize_info["model_max_length"],
            )
        return str(resize_info["resized_prompt"])

    @classmethod
    def supports(cls, model_name_or_path, **kwargs):
        model_supported = cls.get_model_adapter(model_name_or_path) is not None
        if not model_supported or not cls.aws_configured(**kwargs):
            return False

        try:
            session = cls.get_aws_session(**kwargs)
            bedrock = session.client("bedrock")
            foundation_models_response = bedrock.list_foundation_models(byOutputModality="TEXT")
            available_model_ids = [entry["modelId"] for entry in foundation_models_response.get("modelSummaries", [])]
            model_ids_supporting_streaming = [
                entry["modelId"]
                for entry in foundation_models_response.get("modelSummaries", [])
                if entry.get("responseStreamingSupported", False)
            ]
        except AWSConfigurationError as exception:
            raise AmazonBedrockConfigurationError(message=exception.message) from exception
        except Exception as exception:
            raise AmazonBedrockConfigurationError(
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            ) from exception

        model_available = model_name_or_path in available_model_ids
        if not model_available:
            raise AmazonBedrockConfigurationError(
                f"The model {model_name_or_path} is not available in Amazon Bedrock. "
                f"Make sure the model you want to use is available in the configured AWS region and you have access."
            )

        stream: bool = kwargs.get("stream", False)
        model_supports_streaming = model_name_or_path in model_ids_supporting_streaming
        if stream and not model_supports_streaming:
            raise AmazonBedrockConfigurationError(
                f"The model {model_name_or_path} doesn't support streaming. Remove the `stream` parameter."
            )

        return model_supported

    def invoke(self, *args, **kwargs):
        kwargs = kwargs.copy()
        prompt: str = kwargs.pop("prompt", None)
        stream: bool = kwargs.get("stream", self.model_adapter.model_kwargs.get("stream", False))

        if not prompt or not isinstance(prompt, (str, list)):
            raise ValueError(
                f"The model {self.model_name_or_path} requires a valid prompt, but currently, it has no prompt. "
                f"Make sure to provide a prompt in the format that the model expects."
            )

        body = self.model_adapter.prepare_body(prompt=prompt, **kwargs)
        try:
            if stream:
                response = self.client.invoke_model_with_response_stream(
                    body=json.dumps(body),
                    modelId=self.model_name_or_path,
                    accept="application/json",
                    contentType="application/json",
                )
                response_stream = response["body"]
                handler: TokenStreamingHandler = kwargs.get(
                    "stream_handler",
                    self.model_adapter.model_kwargs.get("stream_handler", DefaultTokenStreamingHandler()),
                )
                responses = self.model_adapter.get_stream_responses(stream=response_stream, stream_handler=handler)
            else:
                response = self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model_name_or_path,
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = json.loads(response.get("body").read().decode("utf-8"))
                responses = self.model_adapter.get_responses(response_body=response_body)
        except ClientError as exception:
            raise AmazonBedrockInferenceError(
                f"Could not connect to Amazon Bedrock model {self.model_name_or_path}. "
                "Make sure your AWS environment is configured correctly, "
                "the model is available in the configured AWS region, and you have access."
            ) from exception

        return responses

    @classmethod
    def get_model_adapter(cls, model_name_or_path: str) -> Optional[Type[BedrockModelAdapter]]:
        for pattern, adapter in cls.SUPPORTED_MODEL_PATTERNS.items():
            if re.fullmatch(pattern, model_name_or_path):
                return adapter
        return None
