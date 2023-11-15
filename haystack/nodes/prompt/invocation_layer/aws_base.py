import logging
from abc import ABC
from typing import Optional


from haystack.errors import AWSConfigurationError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    import boto3
    from botocore.exceptions import BotoCoreError


AWS_CONFIGURATION_KEYS = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_profile_name",
]


class AWSBaseInvocationLayer(PromptModelInvocationLayer, ABC):
    """
    Base class for AWS based invocation layers.
    """

    @classmethod
    def aws_configured(cls, **kwargs) -> bool:
        """
        Checks whether this invocation layer is active.
        :param kwargs: The kwargs passed down to the invocation layer.
        :return: True if the invocation layer is active, False otherwise.
        """
        aws_config_provided = any(key in kwargs for key in AWS_CONFIGURATION_KEYS)
        return aws_config_provided

    @classmethod
    def get_aws_session(
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
        Checks if the provided AWS credentials are valid and can be used to connect to AWS.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param kwargs: The kwargs passed down to the service client. Supported kwargs depend on the model chosen.
            See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html.
        :raises AWSConfigurationError: If the provided AWS credentials are invalid.
        :return: The created AWS session.
        """
        boto3_import.check()
        try:
            return boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=aws_region_name,
                profile_name=aws_profile_name,
            )
        except BotoCoreError as e:
            provided_aws_config = {k: v for k, v in kwargs.items() if k in AWS_CONFIGURATION_KEYS}
            raise AWSConfigurationError(
                f"Failed to initialize the session with provided AWS credentials {provided_aws_config}"
            ) from e
