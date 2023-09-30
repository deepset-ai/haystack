import json
import logging
from abc import abstractmethod, ABC
from typing import Optional, Dict, Union, List, Any


from haystack.errors import SageMakerConfigurationError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError


class SageMakerBaseInvocationLayer(PromptModelInvocationLayer, ABC):
    """
    Base class for SageMaker based invocation layers.
    """

    def __init__(self, model_name_or_path: str, max_length: int = 100, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.max_length = max_length

        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        model_max_length = kwargs.get("model_max_length", 1024)

        # Truncate prompt if prompt tokens > model_max_length-max_length
        # (max_length is the length of the generated text)
        # It is hard to determine which tokenizer to use for the SageMaker model
        # so we use GPT2 tokenizer which will likely provide good token count approximation
        self.prompt_handler = DefaultPromptHandler(
            model_name_or_path="gpt2", model_max_length=model_max_length, max_length=self.max_length or 100
        )

    @classmethod
    @abstractmethod
    def get_test_payload(cls) -> Dict[str, Any]:
        """
        Return test payload for the model.
        """

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
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks whether a model_name_or_path passed down (e.g. via PromptNode) is supported by this class.

        :param model_name_or_path: The model_name_or_path to check.
        """
        aws_configuration_keys = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_profile_name",
        ]
        aws_config_provided = any(key in kwargs for key in aws_configuration_keys)
        if aws_config_provided:
            boto3_import.check()
            # attempt to create a session with the provided credentials
            session = cls.check_aws_connect(aws_configuration_keys, kwargs)
            # is endpoint in service?
            cls.check_endpoint_in_service(session, model_name_or_path)

            test_payload = cls.get_test_payload()
            # send test payload to endpoint to see if it's supported
            supported = cls.check_model_input_format(session, model_name_or_path, test_payload, **kwargs)
            return supported
        return False

    @classmethod
    def check_aws_connect(cls, aws_configuration_keys: List[str], kwargs):
        """
        Checks if the provided AWS credentials are valid and can be used to connect to SageMaker.
        :param aws_configuration_keys: The AWS configuration keys to check.
        :param kwargs: The kwargs passed down to the SageMakerClient.
        :return: The boto3 session.
        """
        boto3_import.check()
        try:
            session = cls.create_session(**kwargs)
        except BotoCoreError as e:
            provided_aws_config = {k: v for k, v in kwargs.items() if k in aws_configuration_keys}
            raise SageMakerConfigurationError(
                f"Failed to initialize the session or client with provided AWS credentials {provided_aws_config}"
            ) from e
        return session

    @classmethod
    def check_endpoint_in_service(cls, session: "boto3.Session", endpoint: str):
        """
        Checks if the SageMaker endpoint exists and is in service.
        :param session: The boto3 session.
        :param endpoint: The endpoint to check.
        """
        boto3_import.check()
        client = None
        try:
            client = session.client("sagemaker")
            response = client.describe_endpoint(EndpointName=endpoint)
            endpoint_status = response["EndpointStatus"] if "EndpointStatus" in response else None
            if endpoint_status and endpoint_status.strip() != "InService":
                raise SageMakerConfigurationError(
                    f"SageMaker endpoint {endpoint} exists but is not in service. "
                    f"Please make sure that the endpoint is in state 'InService'."
                )
        except ClientError as e:
            raise SageMakerConfigurationError(
                f"Could not connect to {endpoint} Sagemaker endpoint. "
                f"Please make sure that the endpoint exists and is accessible."
            ) from e
        finally:
            if client:
                client.close()

    @classmethod
    def format_custom_attributes(cls, attributes: dict) -> str:
        """
        Formats the custom attributes for the SageMaker endpoint.
        :param attributes: The custom attributes to format.
        :return: The formatted custom attributes.
        """
        if attributes:
            return ";".join(f"{k}={str(v).lower() if isinstance(v, bool) else str(v)}" for k, v in attributes.items())
        return ""

    @classmethod
    def check_model_input_format(cls, session: "boto3.Session", endpoint: str, test_payload: Any, **kwargs):
        """
        Checks if the SageMaker endpoint supports the test_payload model input format.
        :param session: The boto3 session.
        :param endpoint: The endpoint to hit
        :param test_payload: The payload to send to the endpoint
        :return: True if the endpoint supports the test_payload model input format, False otherwise.
        """
        boto3_import.check()
        custom_attributes = kwargs.get("aws_custom_attributes", None)
        custom_attributes = SageMakerBaseInvocationLayer.format_custom_attributes(custom_attributes)
        client = None
        try:
            client = session.client("sagemaker-runtime")
            client.invoke_endpoint(
                EndpointName=endpoint,
                Body=json.dumps(test_payload),
                ContentType="application/json",
                Accept="application/json",
                CustomAttributes=custom_attributes,
            )
        except ClientError:
            # raised if the endpoint doesn't support the test_payload model input format
            return False
        finally:
            if client:
                client.close()
        return True

    @classmethod
    def create_session(
        cls,
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
