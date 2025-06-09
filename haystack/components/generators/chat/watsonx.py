import json
import os
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall, ToolCallResult
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class WatsonxChatGenerator:
    """
    Enables chat completions using IBM's watsonx.ai foundation models.

    This component interacts with IBM's watsonx.ai platform to generate chat responses
    using various foundation models. It supports the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage)
    format for both input and output.

    The generator works with IBM's foundation models including:
    - granite-13b-chat-v2
    - llama-2-70b-chat
    - llama-3-70b-instruct
    - Other watsonx.ai chat models

    You can customize the generation behavior by passing parameters to the
    watsonx.ai API through the `generation_kwargs` argument. These parameters
    are passed directly to the watsonx.ai inference endpoint.

    For details on watsonx.ai API parameters, see
    [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

    ### Usage example

    ```python
    from haystack.components.generators.chat import WatsonxChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("Explain quantum computing in simple terms")]

    client = WatsonxChatGenerator(
        model="ibm/granite-13b-chat-v2",
        project_id="your-project-id"
    )
    response = client.run(messages)
    print(response)
    ```
    Output:
    ```
    {'replies':
        [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=
        [TextContent(text="Quantum computing uses quantum-mechanical phenomena like ....")],
         _name=None,
         _meta={'model': 'ibm/granite-13b-chat-v2', 'project_id': 'your-project-id',
         'usage': {'prompt_tokens': 12, 'completion_tokens': 45, 'total_tokens': 57}})
        ]
    }
    ```

    The component also supports streaming responses and function calling through
    watsonx.ai's tools parameter.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),
        model: str = "ibm/granite-13b-chat-v2",
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        api_base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        verify: Optional[Union[bool, str]] = None,
    ):
        """
        Initializes the WatsonxChatGenerator with connection and generation parameters.

        Before initializing the component, you can set environment variables:
        - `WATSONX_TIMEOUT` to override the default timeout
        - `WATSONX_MAX_RETRIES` to override the default retry count

        :param api_key: IBM Cloud API key for watsonx.ai access.
            Can be set via `WATSONX_API_KEY` environment variable or passed directly.
        :param model: The model ID to use for completions. Defaults to "ibm/granite-13b-chat-v2".
            Available models can be found in your IBM Cloud account.
        :param project_id: IBM Cloud project ID (required if space_id is not provided).
        :param space_id: watsonx.ai deployment space ID (required if project_id is not provided).
        :param api_base_url: Custom base URL for the API endpoint.
            Defaults to "https://us-south.ml.cloud.ibm.com".
        :param generation_kwargs: Additional parameters to control text generation.
            These parameters are passed directly to the watsonx.ai inference endpoint.
            Supported parameters include:
            - `temperature`: Controls randomness (lower = more deterministic)
            - `max_new_tokens`: Maximum number of tokens to generate
            - `min_new_tokens`: Minimum number of tokens to generate
            - `top_p`: Nucleus sampling probability threshold
            - `top_k`: Number of highest probability tokens to consider
            - `repetition_penalty`: Penalty for repeated tokens
            - `length_penalty`: Penalty based on output length
            - `stop_sequences`: List of sequences where generation should stop
            - `random_seed`: Seed for reproducible results
        :param timeout: Timeout in seconds for API requests.
            Defaults to environment variable `WATSONX_TIMEOUT` or 30 seconds.
        :param max_retries: Maximum number of retry attempts for failed requests.
            Defaults to environment variable `WATSONX_MAX_RETRIES` or 5.
        :param tools: List of tools in Watsonx format for function calling.
            Each tool should be a dictionary with:
            - `name`: Tool name
            - `description`: Tool description
            - `parameters`: JSON schema for parameters
        :param verify: SSL verification setting. Can be:
            - True: Verify SSL certificates (default)
            - False: Skip verification (insecure)
            - Path to CA bundle for custom certificates
        """
        self.api_key = api_key
        self.model = model
        self.project_id = project_id
        self.space_id = space_id
        self.api_base_url = api_base_url or "https://us-south.ml.cloud.ibm.com"
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout or float(os.environ.get("WATSONX_TIMEOUT", "30.0"))
        self.max_retries = max_retries or int(os.environ.get("WATSONX_MAX_RETRIES", "5"))
        self.tools = tools
        self.verify = verify

        if not project_id and not space_id:
            raise ValueError("Either project_id or space_id must be provided")

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Watsonx client with configured credentials."""
        credentials = Credentials(api_key=self.api_key.resolve_value(), url=self.api_base_url)

        self.client = ModelInference(
            model_id=self.model,
            credentials=credentials,
            project_id=self.project_id,
            space_id=self.space_id,
            verify=self.verify,
            max_retries=self.max_retries,
            delay_time=0.5,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            model=self.model,
            project_id=self.project_id,
            space_id=self.space_id,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            tools=self.tools,
            verify=self.verify,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatsonxChatGenerator":
        """Deserialize the component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None, stream: bool = False
    ):
        """
        Generate chat completions synchronously.

        :param messages: List of ChatMessage objects
        :param generation_kwargs: Additional generation parameters
        :param stream: Enable streaming response
        :return: Dictionary with generated replies
        """
        if not messages:
            return {"replies": []}

        api_args = self._prepare_api_call(messages=messages, generation_kwargs=generation_kwargs, stream=stream)

        if stream:
            return self._handle_streaming(api_args)
        return self._handle_standard(api_args)

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None, stream: bool = False
    ):
        """
        Generate chat completions asynchronously.

        :param messages: List of ChatMessage objects
        :param generation_kwargs: Additional generation parameters
        :param stream: Enable streaming response
        :return: Dictionary with generated replies
        """
        if not messages:
            return {"replies": []}

        api_args = self._prepare_api_call(messages=messages, generation_kwargs=generation_kwargs, stream=stream)

        if stream:
            return await self._handle_async_streaming(api_args)
        return await self._handle_async_standard(api_args)

    def _prepare_api_call(
        self, *, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None, stream: bool = False
    ) -> Dict[str, Any]:
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        watsonx_messages = []
        for msg in messages:
            if msg.is_from("user"):
                content = msg.text
            elif msg.is_from("assistant"):
                content = msg.text
                if msg.tool_calls:
                    merged_kwargs["tools"] = [
                        {"name": tc.tool_name, "description": "", "parameters": tc.arguments} for tc in msg.tool_calls
                    ]
            elif msg.is_from("tool"):
                content = {
                    "tool_call_id": msg.tool_call_results[0].origin.id,
                    "content": msg.tool_call_results[0].result,
                }
            else:
                content = msg.text

            watsonx_msg = {"role": msg.role.value, "content": content}
            if msg.name:
                watsonx_msg["name"] = msg.name
            watsonx_messages.append(watsonx_msg)

        if "stream" in merged_kwargs:
            del merged_kwargs["stream"]

        return {"messages": watsonx_messages, "params": merged_kwargs}

    def _handle_streaming(self, api_args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronous streaming response."""
        chunks: List[StreamingChunk] = []
        full_text = ""

        stream = self.client.chat_stream(messages=api_args["messages"], params=api_args["params"])

        for chunk in stream:
            if not isinstance(chunk, dict) or not chunk.get("choices"):
                continue

            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                full_text += content
                chunk_meta = {
                    "model": self.model,
                    "index": chunk["choices"][0].get("index", 0),
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                    "received_at": datetime.now().isoformat(),
                }
                chunks.append(StreamingChunk(content=content, meta=chunk_meta))

        return {
            "replies": [
                ChatMessage.from_assistant(
                    text=full_text,
                    meta={
                        "model": self.model,
                        "finish_reason": chunks[-1].meta["finish_reason"] if chunks else "completed",
                    },
                )
            ],
            "chunks": chunks,
        }

    def _handle_standard(self, api_args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronous standard response."""
        response = self.client.chat(messages=api_args["messages"], params=api_args["params"])
        return self._process_response(response)

    async def _handle_async_streaming(self, api_args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle asynchronous streaming response."""
        chunks: List[StreamingChunk] = []
        full_text = ""

        stream = await self.client.achat_stream(messages=api_args["messages"], params=api_args["params"])

        async for chunk in stream:
            if not isinstance(chunk, dict) or not chunk.get("choices"):
                continue

            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                full_text += content
                chunk_meta = {
                    "model": self.model,
                    "index": chunk["choices"][0].get("index", 0),
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                    "received_at": datetime.now().isoformat(),
                }
                chunks.append(StreamingChunk(content=content, meta=chunk_meta))

        return {
            "replies": [
                ChatMessage.from_assistant(
                    text=full_text,
                    meta={
                        "model": self.model,
                        "finish_reason": chunks[-1].meta["finish_reason"] if chunks else "completed",
                    },
                )
            ],
            "chunks": chunks,
        }

    async def _handle_async_standard(self, api_args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle asynchronous standard response."""
        response = await self.client.achat(**api_args)
        return self._process_response(response)

    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process standard response into Haystack format."""
        if not response.get("choices"):
            return {"replies": []}

        choice = response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls = []

        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                if tc["type"] == "function":
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                        tool_calls.append(
                            ToolCall(id=tc.get("id"), tool_name=tc["function"]["name"], arguments=arguments)
                        )
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool call arguments: %s", tc["function"]["arguments"])

        return {
            "replies": [
                ChatMessage.from_assistant(
                    text=content,
                    tool_calls=tool_calls,
                    meta={
                        "model": self.model,
                        "index": choice.get("index", 0),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": response.get("usage", {}),
                    },
                )
            ]
        }
