import json
import logging
from typing import Any, Optional, Dict, Union, List


from haystack.errors import AWSConfigurationError, AmazonBedrockConfigurationError, AmazonBedrockInferenceError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer.aws_base import AWSBaseInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    from botocore.exceptions import ClientError


class AmazonBedrockBaseInvocationLayer(AWSBaseInvocationLayer):
    """
    Base class for Amazon Bedrock based invocation layers.
    """

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
        except Exception as e:
            raise AmazonBedrockConfigurationError(
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly."
            ) from e

        self.model_input_kwargs = kwargs
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

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        if isinstance(prompt, List):
            raise ValueError("SageMaker invocation layer doesn't support a dictionary as prompt, only a string.")

        resize_info = self.prompt_handler(prompt)
        if resize_info["prompt_length"] != resize_info["new_prompt_length"]:
            logger.warning(
                "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
                "answer length (%s tokens) fit within the max token limit (%s tokens). "
                "Shorten the prompt to prevent it from being cut off.",
                resize_info["prompt_length"],
                max(0, resize_info["model_max_length"] - resize_info["max_length"]),  # type: ignore
                resize_info["max_length"],
                resize_info["model_max_length"],
            )
        return str(resize_info["resized_prompt"])

    @classmethod
    def supports(cls, model_name_or_path, **kwargs):
        supported_model_ids = [
            "amazon.titan-text-express-v1",
            "amazon.titan-text-lite-v1",
            "amazon.titan-text-agile-v1",
            "ai21.j2-ultra-v1",
            "ai21.j2-mid-v1",
            "cohere.command-text-v14",
            "anthropic.claude-v1",
            "anthropic.claude-v2",
            "anthropic.claude-instant-v1",
        ]
        model_supported = model_name_or_path in supported_model_ids
        if not model_supported or not cls.aws_configured(**kwargs):
            return False

        try:
            session = cls.get_aws_session(**kwargs)
            bedrock = session.client("bedrock")
            foundation_models_response = bedrock.list_foundation_models(byOutputModality="TEXT")
            available_model_ids = [entry["modelId"] for entry in foundation_models_response.get("modelSummaries", [])]
        except AWSConfigurationError as e:
            raise AmazonBedrockConfigurationError(message=e.message) from e
        except Exception as e:
            raise AmazonBedrockConfigurationError(
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly."
            ) from e

        model_available = model_name_or_path in available_model_ids
        if not model_available:
            raise AmazonBedrockConfigurationError(
                f"The model {model_name_or_path} is not available in Amazon Bedrock. "
                f"Please make sure the model is available in the configured AWS region and you've been granted access."
            )

        return model_supported

    def _prepare_invoke(self, prompt, **kwargs):
        del kwargs["top_k"]
        if self.model_name_or_path in ["amazon.titan-text-express-v1", "amazon.titan-text-lite-v1"]:
            kwargs["stopSequences"] = kwargs["stop_words"] or []
            kwargs["topP"] = 1 if "topP" not in kwargs else kwargs["topP"]
            kwargs["temperature"] = 0.3 if "temperature" not in kwargs else kwargs["temperature"]
            kwargs["maxTokenCount"] = self.max_length
            del kwargs["stop_words"]
            body = json.dumps({"inputText": prompt, "textGenerationConfig": {**kwargs}})
        if self.model_name_or_path in ["ai21.j2-ultra-v1", "ai21.j2-mid-v1"]:
            kwargs["topP"] = 1 if "topP" not in kwargs else kwargs["topP"]
            kwargs["temperature"] = 0.3 if "temperature" not in kwargs else kwargs["temperature"]
            kwargs["maxTokens"] = self.max_length
            del kwargs["stop_words"]
            body = json.dumps({"prompt": prompt, **kwargs})
        if self.model_name_or_path in ["cohere.command-text-v14"]:
            kwargs["temperature"] = 0.3 if "temperature" not in kwargs else kwargs["temperature"]
            kwargs["max_tokens"] = self.max_length
            del kwargs["stop_words"]
            body = json.dumps({"prompt": prompt, **kwargs})
        if self.model_name_or_path in ["anthropic.claude-v1", "anthropic.claude-v2", "anthropic.claude-instant-v1"]:
            kwargs["max_tokens_to_sample"] = self.max_length
            kwargs["temperature"] = 0 if "temperature" not in kwargs else kwargs["temperature"]
            kwargs["top_k"] = 250 if "top_k" not in kwargs else kwargs["top_k"]
            kwargs["top_p"] = 1 if "top_p" not in kwargs else kwargs["top_p"]
            kwargs["stop_sequences"] = ["\n\nHuman:"] if "stop_sequences" not in kwargs else kwargs["stop_sequences"]
            del kwargs["stop_words"]
            body = json.dumps({"prompt": f"\n\nHuman: {prompt}\n\nAssistant:", **kwargs})
        return body

    def extract_responses(self, r):
        out = json.loads(r["body"].read().decode())
        return out

    def invoke(self, *args, **kwargs):
        prompt: Any = kwargs.get("prompt")
        if not prompt or not isinstance(prompt, (str, list)):
            raise ValueError(
                f"No valid prompt provided. Model {self.model_name_or_path} requires a valid prompt."
                f"Make sure to provide a prompt in the format that the model expects."
            )

        body = self._prepare_invoke(**kwargs)
        try:
            r = self.client.invoke_model(
                body=body, modelId=self.model_name_or_path, accept="application/json", contentType="application/json"
            )
        except ClientError as e:
            raise AmazonBedrockInferenceError(
                f"Could not connect to Amazon Bedrock model {self.model_name_or_path}. Make sure the AWS environment is configured correctly."
            ) from e
        if self.model_name_or_path in ["amazon.titan-text-express-v1", "amazon.titan-text-lite-v1"]:
            responses_list = self.extract_response(r)["results"]
            responses = [responses_list[i]["outputText"] for i in range(len(responses_list))]
        if self.model_name_or_path in ["ai21.j2-ultra-v1", "ai21.j2-mid-v1"]:
            responses_list = self.extract_responses(r)["completions"]
            responses = [responses_list[i]["data"]["text"] for i in range(len(responses_list))]
        if self.model_name_or_path in ["cohere.command-text-v14"]:
            responses_list = self.extract_responses(r)["generations"]
            responses = [responses_list[i]["text"] for i in range(len(responses_list))]
        if self.model_name_or_path in ["anthropic.claude-v1", "anthropic.claude-v2", "anthropic.claude-instant-v1"]:
            responses = [self.extract_responses(r)["completion"]]
        return responses
