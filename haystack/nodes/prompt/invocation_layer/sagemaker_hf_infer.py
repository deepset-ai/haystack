import json
import logging
from typing import Optional, Dict, List, Any

import requests

from haystack.errors import SageMakerInferenceError, SageMakerConfigurationError, SageMakerModelNotReadyError
from haystack.nodes.prompt.invocation_layer.sagemaker_base import SageMakerBaseInvocationLayer

logger = logging.getLogger(__name__)


class SageMakerHFInferenceInvocationLayer(SageMakerBaseInvocationLayer):
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
    - GPT NEO
    - BloomZ

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

        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "max_length",
                "max_time",
                "num_return_sequences",
                "num_beams",
                "no_repeat_ngram_size",
                "temperature",
                "early_stopping",
                "do_sample",
                "top_k",
                "top_p",
                "seed",
                "return_full_text",
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

        stop_words = kwargs.pop("stop_words", None)  # doesn't tolerate empty list
        kwargs_with_defaults = self.model_input_kwargs
        kwargs_with_defaults.update(kwargs)

        # these parameters were valid in June 23, make sure to check the Sagemaker docs for updates
        default_params = {
            "max_length": self.max_length,
            "max_time": None,
            "num_return_sequences": None,
            "num_beams": None,
            "no_repeat_ngram_size": None,
            "temperature": None,
            "early_stopping": None,
            "do_sample": None,
            "top_k": None,
            "top_p": None,
            "seed": None,
            "stopping_criteria": stop_words,
            "return_full_text": None,  # not used by all models (e.g. MPT, Flan)
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

    def _post(self, prompt: str, params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Post data to the SageMaker inference model. It takes in a prompt and returns a list of responses using model invocation.
        :param prompt: The prompt text to be sent to the model.
        :param params: The parameters to be sent to the Hugging Face model (see https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task)
        :return: The generated responses as a list of strings.
        """

        try:
            body = {"text_inputs": prompt, **(params or {})}
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
            if res.status_code == 429:  # type: ignore[union-attr]
                raise SageMakerModelNotReadyError(f"Model not ready: {res.text}")  # type: ignore[union-attr]
            raise SageMakerInferenceError(
                f"SageMaker Inference returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",  # type: ignore[union-attr]
                status_code=res.status_code,  # type: ignore[union-attr]
            )

    def _extract_response(self, json_response: Any) -> List[str]:
        """
        Extracts generated list of texts from the JSON response.
        :param json_response: The JSON response to process.
        :return: A list of generated texts.
        """
        generated_texts = []
        for response in self._unwrap_response(json_response):
            for key in ["generated_texts", "generated_text"]:
                raw_response = response.get(key)
                if raw_response:
                    if isinstance(raw_response, list):
                        generated_texts.extend(raw_response)
                    else:
                        generated_texts.append(raw_response)
        return generated_texts

    def _unwrap_response(self, response: Any):
        """
        Recursively unwrap the JSON response to get to the dictionary level where the generated text is.

        If the response is a list, it recursively calls this method for each sublist.
        If the response is a dict and contains either "generated_text" or "generated_texts" key,
        it yields the dictionary.

        :param response: The response to process.
        :yield: dictionary containing either "generated_text" or "generated_texts" key with a
        string or list of strings as value.
        """
        if isinstance(response, list):
            for sublist in response:
                yield from self._unwrap_response(sublist)
        elif isinstance(response, dict) and ("generated_text" in response or "generated_texts" in response):
            yield response

    @classmethod
    def get_test_payload(cls) -> Dict[str, str]:
        """
        Returns a payload used for testing if the current endpoint supports the JSON payload format used by
        this class.

        As of June 23, Sagemaker endpoints support the format where the payload is a JSON object with:
        "text_inputs" used as the key and the prompt as the value. All other parameters are passed as key/value
        pairs on the same level. See _post method for more details.

        :return: A payload used for testing if the current endpoint is working.
        """
        return {"text_inputs": "Hello world!"}
