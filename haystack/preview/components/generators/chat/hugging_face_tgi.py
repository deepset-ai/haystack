import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Iterable, Callable
from urllib.parse import urlparse

from haystack.preview import component, default_to_dict, default_from_dict
from haystack.preview.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.preview.dataclasses import ChatMessage, StreamingChunk
from haystack.preview.components.generators.hf_utils import check_valid_model, check_generation_params
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import InferenceClient
    from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, TextGenerationResponse, Token
    from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceTGIChatGenerator:
    """
    Enables text generation using HuggingFace Hub hosted chat-based LLMs. This component is designed to seamlessly
    inference chat-based models deployed on the Text Generation Inference (TGI) backend.

    You can use this component for chat LLMs hosted on Hugging Face inference endpoints, the rate-limited
    Inference API tier:

    ```python
    from haystack.preview.components.generators.chat import HuggingFaceTGIChatGenerator
    from haystack.preview.dataclasses import ChatMessage

    messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]


    client = HuggingFaceTGIChatGenerator(model="meta-llama/Llama-2-70b-chat-hf", token="<your-token>")
    client.warm_up()
    response = client.run(messages, generation_kwargs={"max_new_tokens": 120})
    print(response)
    ```

    For chat LLMs hosted on paid https://huggingface.co/inference-endpoints endpoint and/or your own custom TGI
    endpoint, you'll need to provide the URL of the endpoint as well as a valid token:

    ```python
    from haystack.preview.components.generators.chat import HuggingFaceTGIChatGenerator
    from haystack.preview.dataclasses import ChatMessage

    messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]

    client = HuggingFaceTGIChatGenerator(model="meta-llama/Llama-2-70b-chat-hf",
                                         url="<your-tgi-endpoint-url>",
                                         token="<your-token>")
    client.warm_up()
    response = client.run(messages, generation_kwargs={"max_new_tokens": 120})
    print(response)
    ```

     Key Features and Compatibility:
         - **Primary Compatibility**: Designed to work seamlessly with any chat-based model deployed using the TGI
           framework. For more information on TGI, visit https://github.com/huggingface/text-generation-inference.
         - **Hugging Face Inference Endpoints**: Supports inference of TGI chat LLMs deployed on Hugging Face
           inference endpoints. For more details, refer to https://huggingface.co/inference-endpoints.
         - **Inference API Support**: Supports inference of TGI chat LLMs hosted on the rate-limited Inference
           API tier. Learn more about the Inference API at https://huggingface.co/inference-api.
           Discover available chat models using the following command:
           ```
           wget -qO- https://api-inference.huggingface.co/framework/text-generation-inference | grep chat
           ```
           and simply use the model ID as the model parameter for this component. You'll also need to provide a valid
           Hugging Face API token as the token parameter.
         - **Custom TGI Endpoints**: Supports inference of TGI chat LLMs deployed on custom TGI endpoints. Anyone can
           deploy their own TGI endpoint using the TGI framework. For more details, refer
           to https://huggingface.co/inference-endpoints.

     Input and Output Format:
         - **ChatMessage Format**: This component uses the ChatMessage format to structure both input and output,
           ensuring coherent and contextually relevant responses in chat-based text generation scenarios. Details on the
           ChatMessage format can be found at https://github.com/openai/openai-python/blob/main/chatml.md.

    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-13b-chat-hf",
        url: Optional[str] = None,
        token: Optional[str] = None,
        chat_template: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initialize the HuggingFaceTGIChatGenerator instance.

        :param model: A string representing the model path or URL. Default is "meta-llama/Llama-2-13b-chat-hf".
        :param url: An optional string representing the URL of the TGI endpoint.
        :param chat_template: This optional parameter allows you to specify a Jinja template for formatting chat
            messages. While high-quality and well-supported chat models typically include their own chat templates
            accessible through their tokenizer, there are models that do not offer this feature. For such scenarios,
            or if you wish to use a custom template instead of the model's default, you can use this parameter to
            set your preferred chat template.
        :param token: The Hugging Face token for HTTP bearer authorization.
            You can find your HF token at https://huggingface.co/settings/tokens.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's [documentation](https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters)
            for more information.
        :param stop_words: An optional list of strings representing the stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        transformers_import.check()

        if url:
            r = urlparse(url)
            is_valid_url = all([r.scheme in ["http", "https"], r.netloc])
            if not is_valid_url:
                raise ValueError(f"Invalid TGI endpoint URL provided: {url}")

        check_valid_model(model, token)

        # handle generation kwargs setup
        generation_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        check_generation_params(generation_kwargs, ["n"])
        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", [])
        generation_kwargs["stop_sequences"].extend(stop_words or [])

        self.model = model
        self.url = url
        self.chat_template = chat_template
        self.token = token
        self.generation_kwargs = generation_kwargs
        self.client = InferenceClient(url or model, token=token)
        self.streaming_callback = streaming_callback
        self.tokenizer = None

    def warm_up(self) -> None:
        """
        Load the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, token=self.token)
        # mypy can't infer that chat_template attribute exists on the object returned by AutoTokenizer.from_pretrained
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if not chat_template and not self.chat_template:
            logger.warning(
                "The model '%s' doesn't have a default chat_template, and no chat_template was supplied during "
                "this component's initialization. It’s possible that the model doesn't support ChatML inference "
                "format, potentially leading to unexpected behavior.",
                self.model,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :return: A dictionary containing the serialized component.
        """
        callback_name = serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            chat_template=self.chat_template,
            token=self.token if not isinstance(self.token, str) else None,  # don't serialize valid tokens
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceTGIChatGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        # Don't send URL as it is sensitive information
        return {"model": self.model}

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A list containing the generated responses as ChatMessage instances.
        """

        # check generation kwargs given as parameters to override the default ones
        additional_params = ["n", "stop_words"]
        check_generation_params(generation_kwargs, additional_params)

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        num_responses = generation_kwargs.pop("n", 1)

        # merge stop_words and stop_sequences into a single list
        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", [])
        generation_kwargs["stop_sequences"].extend(generation_kwargs.pop("stop_words", []))

        if self.tokenizer is None:
            raise RuntimeError("Please call warm_up() before running LLM inference.")

        # apply either model's chat template or the user-provided one
        prepared_prompt: str = self.tokenizer.apply_chat_template(
            conversation=messages, chat_template=self.chat_template, tokenize=False
        )
        prompt_token_count: int = len(self.tokenizer.encode(prepared_prompt, add_special_tokens=False))

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            return self._run_streaming(prepared_prompt, prompt_token_count, generation_kwargs)

        return self._run_non_streaming(prepared_prompt, prompt_token_count, num_responses, generation_kwargs)

    def _run_streaming(
        self, prepared_prompt: str, prompt_token_count: int, generation_kwargs: Dict[str, Any]
    ) -> Dict[str, List[ChatMessage]]:
        res: Iterable[TextGenerationStreamResponse] = self.client.text_generation(
            prepared_prompt, stream=True, details=True, **generation_kwargs
        )
        chunk = None
        # pylint: disable=not-an-iterable
        for chunk in res:
            token: Token = chunk.token
            if token.special:
                continue
            chunk_metadata = {**asdict(token), **(asdict(chunk.details) if chunk.details else {})}
            stream_chunk = StreamingChunk(token.text, chunk_metadata)
            self.streaming_callback(stream_chunk)  # type: ignore # streaming_callback is not None (verified in the run method)

        message = ChatMessage.from_assistant(chunk.generated_text)
        message.metadata.update(
            {
                "finish_reason": chunk.details.finish_reason.value,
                "index": 0,
                "model": self.client.model,
                "usage": {
                    "completion_tokens": chunk.details.generated_tokens,
                    "prompt_tokens": prompt_token_count,
                    "total_tokens": prompt_token_count + chunk.details.generated_tokens,
                },
            }
        )
        return {"replies": [message]}

    def _run_non_streaming(
        self, prepared_prompt: str, prompt_token_count: int, num_responses: int, generation_kwargs: Dict[str, Any]
    ) -> Dict[str, List[ChatMessage]]:
        chat_messages: List[ChatMessage] = []
        for _i in range(num_responses):
            tgr: TextGenerationResponse = self.client.text_generation(
                prepared_prompt, details=True, **generation_kwargs
            )
            message = ChatMessage.from_assistant(tgr.generated_text)
            message.metadata.update(
                {
                    "finish_reason": tgr.details.finish_reason.value,
                    "index": _i,
                    "model": self.client.model,
                    "usage": {
                        "completion_tokens": len(tgr.details.tokens),
                        "prompt_tokens": prompt_token_count,
                        "total_tokens": prompt_token_count + len(tgr.details.tokens),
                    },
                }
            )
            chat_messages.append(message)
        return {"replies": chat_messages}
