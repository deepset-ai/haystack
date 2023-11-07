import json
import logging
from typing import Optional, Dict, Union, List


from haystack.errors import AmazonBedrockConfigurationError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer.aws_base import AWSBaseInvocationLayer

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError


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
        max_length: Optional[str] = 2048,
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

        self.kwargs = kwargs

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        print(
            "Tokenizer for the bedrock models are not available publicly. The tokens will get truncated automatically"
        )
        return prompt

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
        return body

    def extract_responses(self, r):
        out = json.loads(r["body"].read().decode())
        return out

    def invoke(self, *args, **kwargs):
        client = self.client
        body = self._prepare_invoke(**kwargs)
        r = client.invoke_model(
            body=body, modelId=self.model_name_or_path, accept="application/json", contentType="application/json"
        )
        if self.model_name_or_path in ["amazon.titan-text-express-v1", "amazon.titan-text-lite-v1"]:
            responses_list = self.extract_response(r)["results"]
            responses = [responses_list[i]["outputText"] for i in range(len(responses_list))]
        if self.model_name_or_path in ["ai21.j2-ultra-v1", "ai21.j2-mid-v1"]:
            responses_list = self.extract_responses(r)["completions"]
            responses = [responses_list[i]["data"]["text"] for i in range(len(responses_list))]
        if self.model_name_or_path in ["cohere.command-text-v14"]:
            responses_list = self.extract_responses(r)["generations"]
            responses = [responses_list[i]["text"] for i in range(len(responses_list))]
        return responses
