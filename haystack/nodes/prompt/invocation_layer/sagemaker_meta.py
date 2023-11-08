import json
import logging
from typing import Optional, Dict, List, Any, Union

import requests

from haystack.errors import (
    AWSConfigurationError,
    SageMakerModelNotReadyError,
    SageMakerInferenceError,
    SageMakerConfigurationError,
)
from haystack.nodes.prompt.invocation_layer.sagemaker_base import SageMakerBaseInvocationLayer

logger = logging.getLogger(__name__)


class SageMakerMetaInvocationLayer(SageMakerBaseInvocationLayer):
    """
        SageMaker Meta Invocation Layer


        SageMakerMetaInvocationLayer enables the use of Meta Large Language Models (LLMs) hosted on a SageMaker
        Inference Endpoint via PromptNode. It primarily focuses on LLama-2 models and it supports both the chat and
        instruction following models. Other Meta models have not been tested.

        For guidance on how to deploy such a model to SageMaker, refer to
        the [SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html)
        and follow the instructions provided there.

        As of July 24, this layer has been confirmed to support the following SageMaker deployed models:
        - Llama-2 models

        Technical Note:
        This layer is designed for models that anticipate an input format composed of the following keys/values:
        {'inputs': 'prompt_text', 'parameters': params} where 'inputs' represents the prompt and 'parameters' the
        parameters for the model.


        **Examples**

         ```python
        from haystack.nodes import PromptNode

        # Pass sagemaker endpoint name and authentication details
        pn = PromptNode(model_name_or_path="llama-2-7b",
        model_kwargs={"aws_profile_name": "my_aws_profile_name"})
        res = pn("Berlin is the capital of")
        print(res)
        ```

        **Example using AWS env variables**
        ```python
        import os
        from haystack.nodes import PromptNode

        # We can also configure Sagemaker via AWS environment variables without AWS profile name
        pn = PromptNode(model_name_or_path="llama-2-7b", max_length=512,
                        model_kwargs={"aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                                    "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
                                    "aws_region_name": "us-east-1"})

        response = pn("The secret for a good life is")
        print(response)
        ```

        LLama-2 also supports chat format.
        **Example using chat format**
        ```python
        from haystack.nodes.prompt import PromptNode
        pn = PromptNode(model_name_or_path="llama-2-7b-chat", max_length=512, model_kwargs={"aws_profile_name": "default",
                                                                            "aws_custom_attributes": {"accept_eula": True}})
        pn_input = [[{"role": "user", "content": "what is the recipe of mayonnaise?"}]]
        response = pn(pn_input)
        print(response)
        ```

        Note that in the chat examples we can also include multiple turns between the user and the assistant. See the
        Llama-2 chat documentation for more details.

        **Example using chat format with multiple turns**
        ```python
        from haystack.nodes.prompt import PromptNode
        pn = PromptNode(model_name_or_path="llama-2-7b-chat", max_length=512, model_kwargs={"aws_profile_name": "default",
                                                                            "aws_custom_attributes": {"accept_eula": True}})
        pn_input = [[
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {"role": "assistant", "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n
    1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n
    2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n
    3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n
    These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.",},
        {"role": "user", "content": "What is so great about #1?"}]]
        response = pn(pn_input)
        print(response)
        ```



        Llama-2 models support the following inference payload parameters:

        max_new_tokens: Model generates text until the output length (excluding the input context length) reaches
        max_new_tokens. If specified, it must be a positive integer.
        temperature: Controls the randomness in the output. Higher temperature results in output sequence with
        low-probability words and lower temperature results in output sequence with high-probability words.
        If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.
        top_p: In each step of text generation, sample from the smallest possible set of words with cumulative
        probability top_p. If specified, it must be a float between 0 and 1.
        return_full_text: If True, input text will be part of the output generated text. If specified, it must be
        boolean. The default value for it is False.

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
        # We set the default max_length to 4096 as this is the context window size supported by the LLama-2 model
        kwargs.setdefault("model_max_length", 4096)
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

        # save the kwargs for the model invocation
        self.model_input_kwargs = kwargs

        # As of July 24, SageMaker Meta layer does not support streaming responses.
        # However, even though it's not provided, users may attempt to use streaming responses.
        # Use stream and stream_handler for warning and future use
        self.stream_handler = kwargs.get("stream_handler", None)
        self.stream = kwargs.get("stream", False)

    def invoke(self, *args, **kwargs) -> List[str]:
        """
        Sends the prompt to the remote model and returns the generated response(s).

        :return: The generated responses from the model as a list of strings.
        """
        prompt: Any = kwargs.get("prompt")
        if not prompt or not isinstance(prompt, (str, list)):
            raise ValueError(
                f"No valid prompt provided. Model {self.model_name_or_path} requires a valid prompt."
                f"Make sure to provide a prompt in the format that the model expects."
            )

        if not (isinstance(prompt, str) or self.is_proper_chat_conversation_format(prompt)):
            raise ValueError(
                f"The prompt format is different than what the model expects. "
                f"The model {self.model_name_or_path} requires either a string or messages in the specific chat format. "
                f"For more details, see https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L213)."
            )

        stream = kwargs.get("stream", self.stream)
        stream_handler = kwargs.get("stream_handler", self.stream_handler)
        streaming_requested = stream or stream_handler is not None
        if streaming_requested:
            raise SageMakerConfigurationError("SageMaker model response streaming is not supported yet")

        kwargs_with_defaults = self.model_input_kwargs
        kwargs_with_defaults.update(kwargs)

        default_params = {
            "max_new_tokens": self.max_length,
            "return_full_text": None,
            "temperature": None,
            "top_p": None,
        }
        # put the param in the params if it's in kwargs and not None (e.g. it is actually defined)
        # endpoint doesn't tolerate None values, send only the params that are defined
        params = {
            param: kwargs_with_defaults.get(param, default)
            for param, default in default_params.items()
            if param in kwargs_with_defaults or default is not None
        }
        generated_texts = self._post(prompt=prompt, params=params)
        return generated_texts

    def _post(self, prompt: Any, params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Post data to the SageMaker inference model. It takes in a prompt and returns a list of responses using model
        invocation.
        :param prompt: The prompt text/messages to be sent to the model.
        :param params: The parameters to be sent to the Meta model.
        :return: The generated responses as a list of strings.
        """
        custom_attributes = SageMakerBaseInvocationLayer.format_custom_attributes(
            self.model_input_kwargs.get("aws_custom_attributes", {})
        )
        try:
            body = {"inputs": prompt, "parameters": params}
            response = self.client.invoke_endpoint(
                EndpointName=self.model_name_or_path,
                Body=json.dumps(body),
                ContentType="application/json",
                Accept="application/json",
                CustomAttributes=custom_attributes,
            )
            response_json = response.get("Body").read().decode("utf-8")
            output = json.loads(response_json)
            generated_texts = [o["generation"] for o in output if "generation" in o]
            return generated_texts
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:  # type: ignore[union-attr]
                raise SageMakerModelNotReadyError(f"Model not ready: {res.text}") from err  # type: ignore[union-attr]
            raise SageMakerInferenceError(
                f"SageMaker Inference returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",  # type: ignore[union-attr]
                status_code=res.status_code,  # type: ignore[union-attr]
            ) from err

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        if isinstance(prompt, str):
            return super()._ensure_token_limit(prompt)
        else:
            # TODO: implement truncation for the chat format
            return prompt

    def _is_proper_chat_message_format(self, chat_message: Dict[str, str]) -> bool:
        """
        Checks whether a chat message is in the proper format.
        :param chat_message: The chat message to be checked.
        :return: True if the chat message is in the proper format, False otherwise.
        """
        allowed_roles = {"user", "assistant", "system"}
        return (
            isinstance(chat_message, dict)
            and "role" in chat_message
            and "content" in chat_message
            and chat_message["role"] in allowed_roles
        )

    def is_proper_chat_conversation_format(self, prompt: List[Any]) -> bool:
        """
        Checks whether a chat conversation is in the proper format.
        :param prompt: The chat conversation to be checked.
        :return: True if the chat conversation is in the proper format, False otherwise.
        """
        if not isinstance(prompt, list) or len(prompt) == 0:
            return False

        return all(
            isinstance(message_list, list)
            and all(self._is_proper_chat_message_format(chat_message) for chat_message in message_list)
            for message_list in prompt
        )

    @classmethod
    def get_test_payload(cls) -> Dict[str, Any]:
        """
        Return test payload for the model.
        """
        # implement the abstract method to fulfill the contract, but it won't be used
        # because we override the supports method to check support
        # for the chat and instruction following format manually
        return {}

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks whether a model_name_or_path passed down (e.g. via PromptNode) is supported by this class.

        :param model_name_or_path: The model_name_or_path to check.
        """
        accept_eula = False
        if "aws_custom_attributes" in kwargs and isinstance(kwargs["aws_custom_attributes"], dict):
            accept_eula = kwargs["aws_custom_attributes"].get("accept_eula", False)
        if cls.aws_configured(**kwargs) and accept_eula:
            # attempt to create a session with the provided credentials
            try:
                session = cls.get_aws_session(**kwargs)
            except AWSConfigurationError as e:
                raise SageMakerConfigurationError(message=e.message) from e
            # is endpoint in service?
            cls.check_endpoint_in_service(session, model_name_or_path)

            # let's test both formats as we want to support both chat and instruction following models:
            # 1. instruction following format
            # send test payload to endpoint to see if it's supported
            instruction_test_payload: Dict[str, Any] = {
                "inputs": "Hello world",
                # don't remove max_new_tokens param, if we don't specify it, the model will generate 4k tokens
                "parameters": {"max_new_tokens": 10},
            }
            supported = cls.check_model_input_format(session, model_name_or_path, instruction_test_payload, **kwargs)
            if supported:
                return True

            # 2. chat format
            chat_test_payload: Dict[str, Any] = {
                "inputs": [[{"role": "user", "content": "what is the recipe of mayonnaise?"}]],
                # don't remove max_new_tokens param, if we don't specify it, the model will generate 4k tokens
                "parameters": {"max_new_tokens": 10},
            }
            supported = cls.check_model_input_format(session, model_name_or_path, chat_test_payload, **kwargs)
            return supported
        return False
