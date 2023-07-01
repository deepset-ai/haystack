import json
import logging
from typing import Optional, Dict, List, Any

import requests

from haystack.errors import SageMakerInferenceError, SageMakerConfigurationError, SageMakerModelNotReadyError
from haystack.nodes.prompt.invocation_layer.ai21_jurrasic_2 import (
    AI21TaskSpecificContextualAnswers,
    AI21J2CompleteInput,
)
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler
from haystack.nodes.prompt.invocation_layer.sagemaker_base import SageMakerBaseInvocationLayer

logger = logging.getLogger(__name__)


class SageMakerJumpStartAi21J2CompleteInferenceInvocationLayer(SageMakerBaseInvocationLayer, AI21J2CompleteInput):
    def _filter_kwargs(self, kwargs: dict, params: dict):
        result = {}
        for paramName, paramType in params.items():
            if paramName in kwargs:
                if isinstance(paramType, dict) and isinstance(kwargs[paramName], dict):
                    child = self._filter_kwargs(kwargs[paramName], paramType)
                    if child:
                        result[paramName] = child
                elif isinstance(kwargs[paramName], paramType):
                    result[paramName] = kwargs[paramName]
        return result

    """
    SageMaker HuggingFace Inference Invocation Layer

    SageMakerHFInferenceInvocationLayer enables the use of Large Language Models (LLMs) hosted on a SageMaker Inference
    Endpoint via PromptNode. It supports text-generation and text2text-generation models from HuggingFace, which are
    running on the SageMaker Inference Endpoint.

    As of June 23, this layer has been confirmed to support the following SageMaker deployed models:
    - MPT
    - Dolly V2
    - Flan-U2
    - Flan-T5
    - RedPajama
    - Open Llama
    - GPT-J-6B

    For guidance on how to deploy such a model to SageMaker, refer to
    the [SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html)
    and follow the instructions provided there.

    Technical Note:

    This layer is designed for models that anticipate an input format composed of the following keys/values:
    {'text_inputs': 'prompt_text', **(params or {})}
    The text_inputs key represents the prompt text, with all additional parameters for the model being added at
    the same dictionary level as text_inputs.


    **Example**

     ```python
    from haystack.nodes import PromptNode

    # Pass sagemaker endpoint name and authentication details
    pn = PromptNode(model_name_or_path="jumpstart-dft-hf-textgeneration-dolly-v2-3b-bf16",
    model_kwargs={"aws_profile_name": "my_aws_profile_name", "aws_region_name": "eu-central-1"})
    res = pn("what is the meaning of life?")
    print(res)
    ```

    **Example using AWS env variables**
    ```python
    import os
    from haystack.nodes import PromptNode

    # We can also configure Sagemaker via AWS environment variables without AWS profile name
    pn = PromptNode(model_name_or_path="jumpstart-dft-hf-textgeneration-dolly-v2-3b-bf16", max_length=128,
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
        SageMakerBaseInvocationLayer.__init__(self, model_name_or_path)
        AI21J2CompleteInput.__init__(self)

        try:
            session = SageMakerJumpStartAi21J2CompleteInferenceInvocationLayer.create_session(
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
                f"Make sure the Endpoint exists and AWS environment is configured. {e}"
            )
        self.max_length = max_length

        # for a list of supported parameters
        # see https://docs.ai21.com/reference/contextual-answers-ref
        self.model_input_kwargs = self._filter_kwargs(kwargs, self.params)

        # As of June 23, SageMaker does not support streaming responses.
        # However, even though it's not provided, users may attempt to use streaming responses.
        # Use stream and stream_handler for warning and future use
        self.stream_handler = kwargs.get("stream_handler", None)
        self.stream = kwargs.get("stream", False)

        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        model_max_length = kwargs.get("model_max_length", 8191)

        # Truncate prompt if prompt tokens > model_max_length-max_length
        # (max_length is the length of the generated text)
        # It is hard to determine which tokenizer to use for the SageMaker model
        # so we use GPT2 tokenizer which will likely provide good token count approximation
        self.prompt_handler = DefaultPromptHandler(
            model_name_or_path="gpt2", model_max_length=model_max_length, max_length=self.max_length or 100
        )

    def invoke(self, *args, **kwargs) -> List[str]:
        """
        Sends the prompt to the remote model and returns the generated response(s).
        You can pass all parameters supported by the SageMaker model
        here via **kwargs (e.g. "temperature", "do_sample" ...).

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

        kwargs_with_defaults = self.model_input_kwargs
        kwargs_with_defaults.update(kwargs)

        # these parameters were valid in June 23, make sure to check the Sagemaker docs for updates
        default_params = self.default_params

        # put the param in the params if it's in kwargs and not None (e.g. it is actually defined)
        # endpoint doesn't tolerate None values, send only the params that are defined
        params = {
            param: kwargs_with_defaults.get(param, default)
            for param, default in default_params.items()
            if param in kwargs_with_defaults or default is not None
        }
        generated_texts = self._post(params=params)

        return generated_texts

    def _post(self, params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Post data to the SageMaker inference model. It takes in a prompt and returns a list of responses using model invocation.
        :param prompt: The prompt text to be sent to the model.
        :param params: The parameters to be sent to the Hugging Face model (see https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task)
        :return: The generated responses as a list of strings.
        """

        try:
            body = {**(params or {})}
            response = self.client.invoke_endpoint(
                EndpointName=self.model_name_or_path,
                Body=json.dumps(body),
                ContentType="application/json",
                Accept="application/json",
            )
            response_json = response.get("Body").read().decode("utf-8")
            output = json.loads(response_json)
            return self._extract_response(output)
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:
                raise SageMakerModelNotReadyError(f"Model not ready: {res.text}")
            raise SageMakerInferenceError(
                f"SageMaker Inference returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",
                status_code=res.status_code,
            )

    def _extract_response(self, json_response: Any) -> List[str]:
        """
        Extracts generated list of texts from the JSON response.
        :param json_response: The JSON response to process.
        :return: A list of generated texts.
        """
        return [completion["data"]["text"].strip() for completion in json_response["completions"]]

    @classmethod
    def get_test_payload(cls) -> Dict[str, str]:
        return cls.test_payload


class SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(
    SageMakerJumpStartAi21J2CompleteInferenceInvocationLayer, AI21TaskSpecificContextualAnswers
):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 0,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        **kwargs,
    ):
        SageMakerJumpStartAi21J2CompleteInferenceInvocationLayer.__init__(
            self,
            model_name_or_path,
            max_length,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            aws_region_name,
            aws_profile_name,
            **kwargs,
        )
        AI21TaskSpecificContextualAnswers.__init__(self)

    def _ensure_token_limit(self, prompt: str) -> str:
        return prompt

    def invoke(self, *args, **kwargs) -> List[str]:
        """
        Sends the prompt to the remote model and returns the generated response(s).
        You can pass all parameters supported by the SageMaker model
        here via **kwargs (e.g. "temperature", "do_sample" ...).

        :return: The generated responses from the model as a list of strings.
        """
        prompt = kwargs.get("prompt")
        question = kwargs.get("question")

        if prompt and not question:
            kwargs["question"] = prompt
        question = kwargs.get("question")
        context = kwargs.get("context")
        if not context:
            raise ValueError(
                f"No context provided. Model {self.model_name_or_path} requires context."
                f"Make sure to provide context in kwargs."
            )
        if not question:
            raise ValueError(
                f"No question provided. Model {self.model_name_or_path} requires question."
                f"Make sure to provide question in kwargs."
            )

        stream = kwargs.get("stream", self.stream)
        stream_handler = kwargs.get("stream_handler", self.stream_handler)
        streaming_requested = stream or stream_handler is not None
        if streaming_requested:
            raise SageMakerConfigurationError("SageMaker model response streaming is not supported yet")

        kwargs_with_defaults = self.model_input_kwargs
        kwargs_with_defaults.update(kwargs)

        # these parameters were valid in June 23, make sure to check the Sagemaker docs for updates
        default_params = self.default_params

        # put the param in the params if it's in kwargs and not None (e.g. it is actually defined)
        # endpoint doesn't tolerate None values, send only the params that are defined
        params = {
            param: kwargs_with_defaults.get(param, default)
            for param, default in default_params.items()
            if param in kwargs_with_defaults or default is not None
        }
        generated_texts = self._post(params=params)

        return generated_texts

    def _extract_response(self, json_response: Any) -> List[str]:
        return [json_response["answer"]]
