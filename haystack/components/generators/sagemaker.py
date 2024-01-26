from typing import Optional, List, Dict, Any

import os
import logging
import json

import requests
from haystack.lazy_imports import LazyImport
from haystack import component, default_from_dict, default_to_dict, ComponentError

with LazyImport(message="Run 'pip install boto3'") as boto3_import:
    import boto3
    from botocore.client import BaseClient


logger = logging.getLogger(__name__)


@component
class SagemakerGenerator:
    model_generation_keys = ["generated_text", "generation"]

    """
    Enables text generation using Sagemaker. It supports Large Language Models (LLMs) hosted and deployed on a SageMaker
    Inference Endpoint. For guidance on how to deploy a model to SageMaker, refer to the
    [SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html).

    Example:

    ```bash
    export AWS_ACCESS_KEY_ID=<your_access_key_id>
    export AWS_SECRET_ACCESS_KEY=<your_secret_access_key>
    export AWS_SESSION_TOKEN=<your_session_token>  # This is optional
    export AWS_REGION_NAME=<your_region_name>
    ```

    ```python
    from haystack.components.generators.sagemaker import SagemakerGenerator
    generator = SagemakerGenerator(model="jumpstart-dft-hf-llm-falcon-7b-instruct-bf16")
    generator.warm_up()
    response = generator.run("What's Natural Language Processing? Be brief.")
    print(response)
    ```

    TODO review reply format

    >> {'replies': ['Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on
    >> the interaction between computers and human language. It involves enabling computers to understand, interpret,
    >> and respond to natural human language in a way that is both meaningful and useful.'], 'meta': [{}]}
    ```
    """

    def __init__(
        self,
        model: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        aws_custom_attributes: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Instantiates the session with SageMaker.

        :param model: The name for SageMaker Model Endpoint.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param aws_custom_attributes: Custom attributes to be passed to SageMaker, for example `{"accept_eula": True}`
            in case of Llama-2 models.
        :param generation_kwargs: Additional keyword arguments for text generation. For a list of supported parameters
            see your model's documentation page, for example here for HuggingFace models:
            https://huggingface.co/blog/sagemaker-huggingface-llm#4-run-inference-and-chat-with-our-model

            Specifically, Llama-2 models support the following inference payload parameters:

            - `max_new_tokens`: Model generates text until the output length (excluding the input context length) reaches
                `max_new_tokens`. If specified, it must be a positive integer.
            - `temperature`: Controls the randomness in the output. Higher temperature results in output sequence with
                low-probability words and lower temperature results in output sequence with high-probability words.
                If `temperature=0`, it results in greedy decoding. If specified, it must be a positive float.
            - `top_p`: In each step of text generation, sample from the smallest possible set of words with cumulative
                probability `top_p`. If specified, it must be a float between 0 and 1.
            - `return_full_text`: If `True`, input text will be part of the output generated text. If specified, it must
                be boolean. The default value for it is `False`.
        """
        self.model = model
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", None)
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_KEY", None)
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN", None)
        self.aws_region_name = aws_region_name or os.getenv("AWS_REGION_NAME", None)
        self.aws_profile_name = aws_profile_name or os.getenv("AWS_PROFILE_NAME", None)
        self.aws_custom_attributes = aws_custom_attributes or {}
        self.generation_kwargs = generation_kwargs or {"max_new_tokens": 1024}

        self.client: Optional[BaseClient] = None

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary. We must avoid serializing AWS credentials, so
        we serialize only region and profile names.

        :return: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            aws_region_name=self.aws_region_name,
            aws_profile_name=self.aws_profile_name,
            aws_custom_attributes=self.aws_custom_attributes,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SagemakerGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        return default_from_dict(cls, data)

    def warm_up(self):
        boto3_import.check()
        try:
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.aws_region_name,
                profile_name=self.aws_profile_name,
            )
            self.client = session.client("sagemaker-runtime")
        except Exception as e:
            raise ComponentError(
                f"Could not connect to SageMaker Inference Endpoint '{self.model}'."
                f"Make sure the Endpoint exists and AWS environment is configured."
            ) from e

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt: The string prompt to use for text generation.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
        potentially override the parameters passed in the __init__ method.

        :return: A list of strings containing the generated responses and a list of dictionaries containing the metadata
        for each response.
        """
        if self.client is None:
            raise ValueError("SageMaker Inference client is not initialized. Please call warm_up() first.")

        generation_kwargs = generation_kwargs or self.generation_kwargs
        custom_attributes = ";".join(
            f"{k}={str(v).lower() if isinstance(v, bool) else str(v)}" for k, v in self.aws_custom_attributes.items()
        )
        self.client: BaseClient
        try:
            body = json.dumps({"inputs": prompt, "parameters": generation_kwargs})
            response = self.client.invoke_endpoint(
                EndpointName=self.model,
                Body=body,
                ContentType="application/json",
                Accept="application/json",
                CustomAttributes=custom_attributes,
            )
            response_json = response.get("Body").read().decode("utf-8")
            output: Dict[str, Dict[str, Any]] = json.loads(response_json)

            # Find the key that contains the generated text
            # It can be any of the keys in model_generation_keys, depending on the model
            for key in self.model_generation_keys:
                if key in output[0]:
                    break

            replies = [o.pop(key, None) for o in output]
            return {"replies": replies, "meta": output * len(replies)}
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:
                raise ComponentError(f"Sagemaker model not ready: {res.text}") from err

            raise ComponentError(
                f"SageMaker Inference returned an error. Status code: {res.status_code} Response body: {res.text}",
                status_code=res.status_code,
            ) from err


"""

"""

# import os
# from haystack.nodes import PromptNode

# # We can also configure Sagemaker via AWS environment variables without AWS profile name
# pn = PromptNode(model_name_or_path="jumpstart-dft-hf-llm-falcon-7b-instruct-bf16", max_length=256,
#                 model_kwargs={"aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
#                             "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
#                             "aws_region_name": "eu-west-1"})

# response = pn("Tell me more about Berlin, be elaborate")
# print(response)
