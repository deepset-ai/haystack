import json
import logging
from typing import Optional, Dict, List, Any

import requests

from haystack.errors import SageMakerModelNotReadyError, SageMakerInferenceError, SageMakerConfigurationError
from haystack.nodes.prompt.invocation_layer.sagemaker_base import SageMakerBaseInvocationLayer

logger = logging.getLogger(__name__)


class SageMakerHFTextGenerationInvocationLayer(SageMakerBaseInvocationLayer):
    """
    SageMaker HuggingFace TextGeneration Invocation Layer


    SageMakerHFTextGenerationInvocationLayer enables the use of Large Language Models (LLMs) hosted on a SageMaker
    Inference Endpoint via PromptNode. It supports text-generation from HuggingFace, which are running on the
    SageMaker Inference Endpoint.

    For guidance on how to deploy such a model to SageMaker, refer to
    the [SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html)
    and follow the instructions provided there.

    As of June 23, this layer has been confirmed to support the following SageMaker deployed models:
    - Falcon models

    Technical Note:
    This layer is designed for models that anticipate an input format composed of the following keys/values:
    {'inputs': 'prompt_text', 'parameters': params} where 'inputs' represents the prompt and 'parameters' the
    parameters for the model.


    **Example**

     ```python
    from haystack.nodes import PromptNode

    # Pass sagemaker endpoint name and authentication details
    pn = PromptNode(model_name_or_path="falcon-40b-my-sagemaker-inference-endpoint,
    model_kwargs={"aws_profile_name": "my_aws_profile_name", "aws_region_name": "eu-central-1"})
    res = pn("what is the meaning of life?")
    print(res)
    ```

    **Example using AWS env variables**
    ```python
    import os
    from haystack.nodes import PromptNode

    # We can also configure Sagemaker via AWS environment variables without AWS profile name
    pn = PromptNode(model_name_or_path="hf-llm-falcon-7b-instruct-bf16-2023-06-22-16-22-19-811", max_length=256,
                    model_kwargs={"aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                                "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
                                "aws_region_name": "us-east-1"})

    response = pn("Tell me more about Berlin, be elaborate")
    print(response)
    ```

    Of course, in both examples your endpoints, region names and other settings will be different.
    You can find it in the SageMaker AWS console.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 100,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Instantiates the session with SageMaker using IAM based authentication via boto3.

        :param model_name_or_path: The name for SageMaker Model Endpoint.
        :param max_length: The maximum length of the output text.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        """
        super().__init__(model_name_or_path, max_length=max_length, **kwargs)
        try:
            session = self.get_aws_session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_region_name=aws_region_name,
                aws_profile_name=aws_profile_name,
            )
            self.client = session.client("sagemaker-runtime")
        except Exception as e:
            raise SageMakerInferenceError(
                f"Could not connect to SageMaker Inference Endpoint {model_name_or_path}."
                f"Make sure the Endpoint exists and AWS environment is configured."
            ) from e

        # for a list of supported parameters
        # see https://huggingface.co/blog/sagemaker-huggingface-llm#4-run-inference-and-chat-with-our-model
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "best_of",
                "details",
                "do_sample",
                "max_new_tokens",
                "repetition_penalty",
                "return_full_text",
                "seed",
                "temperature",
                "top_k",
                "top_p",
                "truncate",
                "typical_p",
                "watermark",
            ]
            if key in kwargs
        }

        # As of June 23, SageMaker does not support streaming responses.
        # However, even though it's not provided, users may attempt to use streaming responses.
        # Use stream and stream_handler for warning and future use
        self.stream_handler = kwargs.get("stream_handler", None)
        self.stream = kwargs.get("stream", False)

    def invoke(self, *args, **kwargs) -> List[str]:
        """
        Sends the prompt to the remote model and returns the generated response(s).
        You can pass all parameters supported by the Huggingface Transformers `generate` method
        here via **kwargs (e.g. "temperature", "stop" ...).

        :return: The generated responses from the model as a list of strings.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )

        stream = kwargs.get("stream", self.stream)
        stream_handler = kwargs.get("stream_handler", self.stream_handler)
        streaming_requested = stream or stream_handler is not None
        if streaming_requested:
            raise SageMakerConfigurationError("SageMaker model response streaming is not supported yet")

        stop_words = kwargs.pop("stop_words", None) or []
        kwargs_with_defaults = self.model_input_kwargs
        kwargs_with_defaults.update(kwargs)

        # For the list of supported parameters and the docs
        # see https://huggingface.co/blog/sagemaker-huggingface-llm#4-run-inference-and-chat-with-our-model
        # these parameters were valid in June 23, make sure to check the docs for updates
        params = {
            "best_of": kwargs_with_defaults.get("best_of", None),
            "details": kwargs_with_defaults.get("details", False),
            "do_sample": kwargs_with_defaults.get("do_sample", False),
            "max_new_tokens": kwargs_with_defaults.get("max_new_tokens", self.max_length),
            "repetition_penalty": kwargs_with_defaults.get("repetition_penalty", None),
            "return_full_text": kwargs_with_defaults.get("return_full_text", False),
            "seed": kwargs_with_defaults.get("seed", None),
            "stop": kwargs_with_defaults.get("stop", stop_words),
            "temperature": kwargs_with_defaults.get("temperature", 1.0),
            "top_k": kwargs_with_defaults.get("top_k", None),
            "top_p": kwargs_with_defaults.get("top_p", None),
            "truncate": kwargs_with_defaults.get("truncate", None),
            "typical_p": kwargs_with_defaults.get("typical_p", None),
            "watermark": kwargs_with_defaults.get("watermark", False),
        }
        generated_texts = self._post(prompt=prompt, params=params)
        return generated_texts

    def _post(self, prompt: str, params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Post data to the SageMaker inference model. It takes in a prompt and returns a list of responses using model invocation.
        :param prompt: The prompt text to be sent to the model.
        :param params: The parameters to be sent to the Hugging Face model (see https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task)
        :return: The generated responses as a list of strings.
        """

        try:
            body = {"inputs": prompt, "parameters": params}
            response = self.client.invoke_endpoint(
                EndpointName=self.model_name_or_path,
                Body=json.dumps(body),
                ContentType="application/json",
                Accept="application/json",
            )
            response_json = response.get("Body").read().decode("utf-8")
            output = json.loads(response_json)
            generated_texts = [o["generated_text"] for o in output if "generated_text" in o]
            return generated_texts
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:  # type: ignore[union-attr]
                raise SageMakerModelNotReadyError(f"Model not ready: {res.text}") from err  # type: ignore[union-attr]
            raise SageMakerInferenceError(
                f"SageMaker Inference returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",  # type: ignore[union-attr]
                status_code=res.status_code,  # type: ignore[union-attr]
            ) from err

    @classmethod
    def get_test_payload(cls) -> Dict[str, Any]:
        """
        Returns a payload used for testing if the current endpoint supports the JSON payload format used by
        this class.

        As of June 23, Sagemaker endpoints support the JSON payload format from the
        https://github.com/huggingface/text-generation-inference project. At the time of writing this docstring,
        only Falcon models were deployed using this format. See python client implementation from the
        https://github.com/huggingface/text-generation-inference for more details.

        :return: A payload used for testing if the current endpoint is working.
        """
        return {"inputs": "Hello world", "parameters": {}}
