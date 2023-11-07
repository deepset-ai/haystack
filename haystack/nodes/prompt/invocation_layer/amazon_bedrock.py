import json
import logging
from abc import abstractmethod, ABC
from typing import Optional, Dict, Union, List, Any


from haystack.errors import AmazonBedrockConfigurationError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError


class AmazonBedrockBaseInvocationLayer(PromptModelInvocationLayer, ABC):
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
            session = AmazonBedrockBaseInvocationLayer.create_session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_region_name=aws_region_name,
                aws_profile_name=aws_profile_name,
            )
            self.client = session.client("bedrock-runtime")
        except:
            raise AmazonBedrockConfigurationError

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        print(
            "Tokenizer for the bedrock models are not available publicly. The tokens will get truncated automatically"
        )
        return prompt

    @classmethod
    def supports(cls, model_name_or_path, **kwargs):
        model_summary = self.client.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
        model_list = [i["modelId"] for i in model_summary]
        return model_name_or_path in model_list

    @classmethod
    def create_session(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates an AWS Session with the given parameters.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :raise NoCredentialsError: If the AWS credentials are not provided or invalid.
        :return: The created AWS Session.
        """
        boto3_import.check()
        return boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region_name,
            profile_name=aws_profile_name,
        )

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

    def invoke(self, *args, **kwargs):
        client = self.client
        prompt = kwargs.get("prompt")
        body = self._prepare_invoke(**kwargs)
        r = client.invoke_model(
            body=body, modelId=self.model_name_or_path, accept="application/json", contentType="application/json"
        )
        if self.model_name_or_path in ["amazon.titan-text-express-v1", "amazon.titan-text-lite-v1"]:
            responses = json.loads(r["body"].read().decode())["results"][0]["outputText"]
        if self.model_name_or_path in ["ai21.j2-ultra-v1", "ai21.j2-mid-v1"]:
            responses = json.loads(r["body"].read().decode())["completions"][0]["data"]["text"]
        if self.model_name_or_path in ["cohere.command-text-v14"]:
            responses = json.loads(r["body"].read().decode())["generations"][0]["text"]
        return [responses]
