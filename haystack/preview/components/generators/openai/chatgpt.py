from typing import Optional, List, Callable, Dict, Any

import sys
import builtins
import logging
from dataclasses import asdict

import openai

from haystack.preview import component, default_from_dict, default_to_dict, DeserializationError

from haystack.preview.dataclasses.chat_message import ChatMessage


logger = logging.getLogger(__name__)


TOKENS_PER_MESSAGE_OVERHEAD = 4


def default_streaming_callback(chunk):
    """
    Default callback function for streaming responses from OpenAI API.
    Prints the tokens of the first completion to stdout as soon as they are received and returns the chunk unchanged.
    """
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, flush=True, end="")
    return chunk


@component
class ChatGPTGenerator:
    """
    ChatGPT LLM Generator.

    Queries ChatGPT using OpenAI's GPT-3 ChatGPT API. Invocations are made using REST API.
    See [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    # TODO support function calling!

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = "https://api.openai.com/v1",
    ):
        """
        Creates an instance of ChatGPTGenerator for OpenAI's GPT-3.5 model.

        :param api_key: The OpenAI API key.
        :param model_name: The name of the model to use.
        :param system_prompt: An additional message to be sent to the LLM at the beginning of each conversation.
            Typically, a conversation is formatted with a system message first, followed by alternating messages from
            the 'user' (the "queries") and the 'assistant' (the "responses"). The system message helps set the behavior
            of the assistant. For example, you can modify the personality of the assistant or provide specific
            instructions about how it should behave throughout the conversation.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param model_parameters: A dictionary of parameters to use for the model. See OpenAI
            [documentation](https://platform.openai.com/docs/api-reference/chat) for more details. Some of the supported
            parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
                values are the bias to add to that token.
            - `openai_organization`: The OpenAI organization ID.
        """
        if not api_key:
            logger.warning("OpenAI API key is missing. You need to provide an API key to Pipeline.run().")

        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model_parameters = model_parameters
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        if self.streaming_callback:
            module = self.streaming_callback.__module__
            if module == "builtins":
                callback_name = self.streaming_callback.__name__
            else:
                callback_name = f"{module}.{self.streaming_callback.__name__}"
        else:
            callback_name = None

        return default_to_dict(
            self,
            api_key=self.api_key,
            model_name=self.model_name,
            model_parameters=self.model_parameters,
            system_prompt=self.system_prompt,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatGPTGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        streaming_callback = None
        if "streaming_callback" in init_params:
            parts = init_params["streaming_callback"].split(".")
            module_name = ".".join(parts[:-1])
            function_name = parts[-1]
            module = sys.modules.get(module_name, None)
            if not module:
                raise DeserializationError(f"Could not locate the module of the streaming callback: {module_name}")
            streaming_callback = getattr(module, function_name, None)
            if not streaming_callback:
                raise DeserializationError(f"Could not locate the streaming callback: {function_name}")
            data["init_parameters"]["streaming_callback"] = streaming_callback
        return default_from_dict(cls, data)

    @component.output_types(replies=List[List[str]], metadata=List[Dict[str, Any]])
    def run(
        self,
        prompts: List[str],
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = "https://api.openai.com/v1",
    ):
        """
        Queries the LLM with the prompts to produce replies.

        :param prompts: The prompts to be sent to the generative model.
        :param api_key: The OpenAI API key.
        :param model_name: The name of the model to use.
        :param system_prompt: An additional message to be sent to the LLM at the beginning of each conversation.
            Typically, a conversation is formatted with a system message first, followed by alternating messages from
            the 'user' (the "queries") and the 'assistant' (the "responses"). The system message helps set the behavior
            of the assistant. For example, you can modify the personality of the assistant or provide specific
            instructions about how it should behave throughout the conversation.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param model_parameters: A dictionary of parameters to use for the model. See OpenAI
            [documentation](https://platform.openai.com/docs/api-reference/chat) for more details. Some of the supported
            parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
                values are the bias to add to that token.
            - `openai_organization`: The OpenAI organization ID.

        See OpenAI documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
        """
        api_key = api_key if api_key is not None else self.api_key
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please provide an API key.")

        model_name = model_name or self.model_name
        system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        model_parameters = model_parameters if model_parameters is not None else self.model_parameters
        streaming_callback = streaming_callback or self.streaming_callback
        api_base_url = api_base_url or self.api_base_url

        if system_prompt:
            system_message = ChatMessage(content=system_prompt, role="system")
        chats = []
        for prompt in prompts:
            message = ChatMessage(content=prompt, role="user")
            if system_prompt:
                chats.append([system_message, message])
            else:
                chats.append([message])

        all_replies, all_metadata = [], []
        for chat in chats:
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                api_key=self.api_key,
                messages=[asdict(message) for message in chat],
                stream=streaming_callback is not None,
                **(self.model_parameters or model_parameters or {}),
            )

            replies: List[str]
            metadata: List[Dict[str, Any]]
            if streaming_callback:
                replies_dict = {}
                metadata_dict: Dict[str, Dict[str, Any]] = {}
                for chunk in completion:
                    chunk = streaming_callback(chunk)
                    for choice in chunk.choices:
                        if choice.index not in replies_dict:
                            replies_dict[choice.index] = ""
                            metadata_dict[choice.index] = {}

                        if hasattr(choice.delta, "content"):
                            replies_dict[choice.index] += choice.delta.content
                        metadata_dict[choice.index] = {
                            "model": chunk.model,
                            "index": choice.index,
                            "finish_reason": choice.finish_reason,
                        }
                all_replies.append(list(replies_dict.values()))
                all_metadata.append(list(metadata_dict.values()))
                self._check_truncated_answers(list(metadata_dict.values()))

            else:
                metadata = [
                    {
                        "model": completion.model,
                        "index": choice.index,
                        "finish_reason": choice.finish_reason,
                        "usage": dict(completion.usage.items()),
                    }
                    for choice in completion.choices
                ]
                replies = [choice.message.content.strip() for choice in completion.choices]
                all_replies.append(replies)
                all_metadata.append(metadata)
                self._check_truncated_answers(metadata)

        return {"replies": all_replies, "metadata": all_metadata}

    def _check_truncated_answers(self, metadata: List[Dict[str, Any]]):
        """
        Check the `finish_reason` returned with the OpenAI completions.
        If the `finish_reason` is `length`, log a warning to the user.

        :param result: The result returned from the OpenAI API.
        :param payload: The payload sent to the OpenAI API.
        """
        truncated_completions = sum(1 for meta in metadata if meta.get("finish_reason") != "stop")
        if truncated_completions > 0:
            logger.warning(
                "%s out of the %s completions have been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                truncated_completions,
                len(metadata),
            )
