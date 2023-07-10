from abc import ABC, abstractmethod
import json
import logging
from typing import Optional, Dict, List, Any
from enum import Enum

import requests

from haystack.errors import SageMakerInferenceError, SageMakerConfigurationError, SageMakerModelNotReadyError
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler
from haystack.nodes.prompt.invocation_layer.sagemaker_base import SageMakerBaseInvocationLayer
import botocore

logger = logging.getLogger(__name__)


class Ai21LabsApis(str, Enum):
    """
    The APIs of AI21 models for different tasks.

    For all the APIs and their capabilities see https://docs.ai21.com/reference
    """

    J2_COMPLETE = "j2-complete"
    PARAPHRASE = "paraphrase"
    GRAMMATICAL_ERROR_CORRECTIONS = "gec"
    TEXT_IMPROVEMENTS = "improvements"
    SUMMARIZE = "summarize"
    TEXT_SEGMENTATION = "segmentation"
    CONTEXTUAL_ANSWERS = "answer"


class AI21ResponseExtractor(ABC):
    @abstractmethod
    def extract_response(self, json_response: Any) -> List[str]:
        """
        Method that extracts the response as text from the model's JSON response.

        :param json_response: Response from model in JSON format.
        :return: List of text responses
        """
        pass


class ContextualAnswersResponseExtractor(AI21ResponseExtractor):
    def extract_response(self, json_response: Any) -> List[str]:
        return [json_response["answer"]]


class J2CompleteResponseExtractor(AI21ResponseExtractor):
    def extract_response(self, json_response: Any) -> List[str]:
        return [completion["data"]["text"].strip() for completion in json_response["completions"]]


class TestPayloadProvider(ABC):
    @abstractmethod
    def get_test_payload(self) -> Dict[str, any]:
        """
        Provides a test payload for the API of this model.

        :return: Json payload in the format of this models API
        """
        pass


class ContextualAnswersTestPayloadProvider(TestPayloadProvider):
    def get_test_payload(self) -> Dict[str, any]:
        return {
            "context": "The tower is 330 metres (1,083 ft) tall,[6] about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest human-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure in the world to surpass both the 200-metre and 300-metre mark in height. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
            "question": "What is the height of the Eiffel tower?",
        }


class J2CompleteTestPayloadProvider(TestPayloadProvider):
    def get_test_payload(self) -> Dict[str, any]:
        return {
            "prompt": """
            ### Instruction ###
            Translate the text below to Spanish:
            Text: "hello!"
            """,
            "numResults": 1,
            "maxTokens": 16,
            "minTokens": 0,
            "temperature": 0.7,
            "topP": 1,
            "stopSequences": None,
            "topKReturn": 0,
            "frequencyPenalty": {
                "scale": 1,
                "applyToWhitespaces": True,
                "applyToPunctuation": True,
                "applyToNumbers": True,
                "applyToStopwords": True,
                "applyToEmojis": True,
            },
            "presencePenalty": {
                "scale": 0,
                "applyToWhitespaces": True,
                "applyToPunctuation": True,
                "applyToNumbers": True,
                "applyToStopwords": True,
                "applyToEmojis": True,
            },
            "countPenalty": {
                "scale": 0,
                "applyToWhitespaces": True,
                "applyToPunctuation": True,
                "applyToNumbers": True,
                "applyToStopwords": True,
                "applyToEmojis": True,
            },
        }


class SageMakerAi21LabsInferenceInvocationLayer(SageMakerBaseInvocationLayer):
    """
    SageMaker AI21 Labs Inference Invocation Layer

    SageMakerAi21LabsInferenceInvocationLayer enables the use of AI21 Labs models hosted on a SageMaker Inference
    Endpoint via PromptNode.


    For guidance on how to deploy a AI21 Labs model to SageMaker, refer to
    the [SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html)
    and follow the instructions provided there.

    Technical Note:

    This layer passes the input without to the model without any validation.
    The AI21 Labs documentation describes how you should format the input for specific model: https://docs.ai21.com/reference
    For example the Contextual Answers API Model expects the following input:
    {
        "context": "In 2020 and 2021, enormous QE — approximately $4.4 trillion, or 18%, of 2021 gross domestic product (GDP) — and enormous fiscal stimulus (which has been and always will be inflationary) — approximately $5 trillion, or 21%, of 2021 GDP — stabilized markets and allowed companies to raise enormous amounts of capital. In addition, this infusion of capital saved many small businesses and put more than $2.5 trillion in the hands of consumers and almost $1 trillion into state and local coffers. These actions led to a rapid decline in unemployment, dropping from 15% to under 4% in 20 months — the magnitude and speed of which were both unprecedented. Additionally, the economy grew 7% in 2021 despite the arrival of the Delta and Omicron variants and the global supply chain shortages, which were largely fueled by the dramatic upswing in consumer spending and the shift in that spend from services to goods. Fortunately, during these two years, vaccines for COVID-19 were also rapidly developed and distributed.In today's economy, the consumer is in excellent financial shape (on average), with leverage among the lowest on record, excellent mortgage underwriting (even though we've had home price appreciation), plentiful jobs with wage increases and more than $2 trillion in excess savings, mostly due to government stimulus. Most consumers and companies (and states) are still flush with the money generated in 2020 and 2021, with consumer spending over the last several months 12% above pre-COVID-19 levels. (But we must recognize that the account balances in lower-income households, smaller to begin with, are going down faster and that income for those households is not keeping pace with rising inflation.) Today's economic landscape is completely different from the 2008 financial crisis when the consumer was extraordinarily overleveraged, as was the financial system as a whole — from banks and investment banks to shadow banks, hedge funds, private equity, Fannie Mae and many other entities. In addition, home price appreciation, fed by bad underwriting and leverage in the mortgage system, led to excessive speculation, which was missed by virtually everyone — eventually leading to nearly $1 trillion in actual losses.",
        "question": "Did the economy shrink after the Omicron variant arrived?"
    }


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
    pn = PromptNode(model_name_or_path="j2-mid", max_length=128,
                    model_kwargs={"aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                                "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
                                "aws_region_name": "us-east-1",
                                "ai21_labs_api":"j2-complete"})

    prompt = \"""
            ### Instruction ###
            Translate the text below to Spanish:
            Text: "hello!"
            \"""
    response = pn(prompt, temperature=0.1)
    print(response)
    ```

    Of course, in the examples your endpoints, region names and other settings will be different.
    You can find it in the SageMaker AWS console.
    """

    def __init__(
        self,
        model_name_or_path: str,
        ai21_labs_api: Ai21LabsApis,
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

        :param ai21_labs_api: Required parameter that specifies which AI21 Labs API the model implements. Should be one of the following "j2-complete", "paraphrase",= "gec", "improvements", "summarize", "segmentation", or "answer".
        :param model_name_or_path: The name for SageMaker Model Endpoint.
        :param max_length: The maximum length of the output text.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        """
        SageMakerBaseInvocationLayer.__init__(self, model_name_or_path)

        try:
            session = SageMakerBaseInvocationLayer.create_session(
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

        # Supported APIs from https://docs.ai21.com/reference
        self.ai21_labs_api = ai21_labs_api

        self.response_extractor = self.response_extractor_factory(self.ai21_labs_api)

        # for a list of supported parameters
        # see https://docs.ai21.com/reference
        self.model_input_kwargs = kwargs

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
        You can pass all parameters supported by the AI21 Labs model
        here via **kwargs (e.g. "temperature", "numResults" ...).

        :return: The generated responses from the model as a list of strings.
        """
        # prompt = kwargs.get("prompt")

        # if not prompt:
        #    raise ValueError(
        #        f"No prompt provided. Model {self.model_name_or_path} requires prompt."
        #        f"Make sure to provide prompt in kwargs."
        #    )

        stream = kwargs.get("stream", self.stream)
        stream_handler = kwargs.get("stream_handler", self.stream_handler)
        streaming_requested = stream or stream_handler is not None
        if streaming_requested:
            raise SageMakerConfigurationError("SageMaker model response streaming is not supported yet")

        kwargs_with_defaults = self.model_input_kwargs
        kwargs_with_defaults.update(kwargs)
        try:
            generated_texts = self._post(params=kwargs_with_defaults)
            return generated_texts

        except self.client.exceptions.ModelError as e:
            message = e.response["Error"]["Message"]
            mightBeWrongParameter = "validation error for Request\\nbody" in message
            raise SageMakerInferenceError(
                f"SageMaker Inference endpoint returned an error. {'This error is most likely caused by a parameter that the model does not support. Please make sure that you are only sending supported parameters: https://docs.ai21.com/reference.' if mightBeWrongParameter else ''}\nError message: {message}",
                status_code=e.response["OriginalStatusCode"],
            )

    def _post(self, params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Post data to the SageMaker inference model. It takes in the parameters to send to the model and returns a list of responses using model invocation.
        :param params: The parameters to be sent to the AI21 Labs model (see https://docs.ai21.com/reference)
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
        return self.response_extractor.extract_response(json_response)

    def response_extractor_factory(self, task: Ai21LabsApis = Ai21LabsApis.J2_COMPLETE) -> AI21ResponseExtractor:
        """Factory method for AI21ResponseExtractor"""
        extractors = {
            Ai21LabsApis.CONTEXTUAL_ANSWERS: ContextualAnswersResponseExtractor,
            Ai21LabsApis.J2_COMPLETE: J2CompleteResponseExtractor,
        }

        return extractors[task]()

    @classmethod
    def test_payload_provider_factory(cls, task: Ai21LabsApis = Ai21LabsApis.J2_COMPLETE) -> TestPayloadProvider:
        """Factory method for TestPayloadProvider"""
        providers = {
            Ai21LabsApis.CONTEXTUAL_ANSWERS: ContextualAnswersTestPayloadProvider,
            Ai21LabsApis.J2_COMPLETE: J2CompleteTestPayloadProvider,
        }

        return providers[task]()

    @classmethod
    def get_test_payload(cls, ai21_labs_api: Ai21LabsApis = Ai21LabsApis.J2_COMPLETE, **kwargs) -> Dict[str, str]:
        test_payload_provider = cls.test_payload_provider_factory(ai21_labs_api)
        return test_payload_provider.get_test_payload()
