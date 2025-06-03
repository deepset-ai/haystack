import os
from typing import Any, Dict, List, Optional, Union, Generator
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

@component
class WatsonxGenerator:
    """
    Generates text using IBM's watsonx.ai foundational models.

    This component supports IBM's Granite and other watsonx.ai models with features like:
    - Single prompt text generation
    - Streaming responses
    - Content moderation (guardrails)
    - Custom generation parameters

    You can customize the text generation by passing parameters to the watsonx API through
    the `generation_kwargs` argument during initialization or at runtime. Any parameter that
    works with `ibm_watsonx_ai.foundation_models.ModelInference.generate_text()` will work here.

    For details on watsonx API parameters, see:
    [IBM watsonx.ai documentation](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html)

    ### Supported Models
    Works with IBM's foundation models including:
    - granite-13b-instruct-v2 (default)
    - granite-20b-instruct-v1
    - llama-2-70b-chat
    - Other watsonx.ai models

    ### Usage example

    ```python
    from haystack.components.generators import WatsonxGenerator
    from haystack.utils import Secret

    # Initialize with default model
    generator = WatsonxGenerator(
        api_key=Secret.from_token("your-api-key"),
        project_id="your-project-id",
        generation_kwargs={"max_new_tokens": 100}
    )

    # Generate text
    response = generator.run("Explain quantum computing in simple terms")
    print(response)

    # Example output:
    # {'replies': ['Quantum computing uses quantum bits that can exist in multiple states...'],
    #  'meta': [{'model': 'ibm/granite-13b-instruct-v2', 'finish_reason': 'completed'}],
    #  'chunks': []}
    ```

    ### Streaming Support
    The component supports streaming responses through watsonx's `generate_text_stream` API.

    ### Streaming example
    ```python
    # Initialize with streaming
    streaming_generator = WatsonxGenerator(
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        project_id="your-project-id"
    )

    # Generate with streaming
    response = streaming_generator.run("Write a poem about AI", stream=True)
    for chunk in response["chunks"]:
        print(chunk.content, end="", flush=True)
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),
        model: str = "ibm/granite-13b-instruct-v2",
        project_id: str = None,
        space_id: str = None,
        api_base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        verify: Optional[Union[bool, str]] = None,
    ):
        """
        Creates an instance of WatsonxGenerator. Uses IBM's granite-13b-instruct-v2 model by default.

        Authentication requires either:
        - A project_id (for IBM Cloud projects)
        - A space_id (for watsonx.ai deployments)

        :param api_key: The IBM Cloud API key for watsonx.ai access.
            Can be provided directly or via environment variable WATSONX_API_KEY.
        :param model: The model ID to use for generation. Defaults to "ibm/granite-13b-instruct-v2".
            Other options include:
            - "ibm/granite-20b-instruct-v1"
            - "ibm/llama-2-70b-chat"
            - Other watsonx.ai foundation models
        :param project_id: The IBM Cloud project ID. Required if space_id is not provided.
        :param space_id: The watsonx.ai deployment space ID. Required if project_id is not provided.
        :param api_base_url: Custom base URL for the API endpoint.
            Defaults to "https://us-south.ml.cloud.ibm.com".
        :param generation_kwargs: Additional parameters for text generation. These parameters are
            sent directly to the watsonx.ai API. Common parameters include:
            - `max_new_tokens`: Maximum number of tokens to generate (default: 20)
            - `temperature`: Value between 0-2 controlling randomness (1.0 is neutral)
            - `top_p`: Nucleus sampling parameter (0.0-1.0)
            - `decoding_method`: "greedy" or "sample"
            - `repetition_penalty`: Penalty for repeated tokens (1.0 is neutral)
            See IBM documentation for full parameter list.
        :param verify: SSL verification setting. Can be:
            - True: Verify SSL certificates (default)
            - False: Skip verification
            - Path to CA bundle for custom certificates

        ### Environment Variables
        The component respects these environment variables:
        - WATSONX_API_KEY: API key if not provided directly
        - WATSONX_PROJECT_ID: Project ID if not provided directly
        - WATSONX_SPACE_ID: Space ID if not provided directly

        """
        self.api_key = api_key
        self.model = model
        self.project_id = project_id
        self.space_id = space_id
        self.api_base_url = api_base_url
        self.verify = verify
        self.generation_kwargs = generation_kwargs or {}

        if not project_id and not space_id:
            raise ValueError("Either project_id or space_id must be provided")

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Watsonx client with the configured credentials."""
        credentials = Credentials(
            api_key=self.api_key.resolve_value(),
            url=self.api_base_url or "https://us-south.ml.cloud.ibm.com"
        )
        self.client = ModelInference(
            model_id=self.model,
            credentials=credentials,
            project_id=self.project_id,
            space_id=self.space_id,
            params=self.generation_kwargs,
            verify=self.verify
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            model=self.model,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            project_id=self.project_id,
            space_id=self.space_id,
            verify=self.verify,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatsonxGenerator":
        """Deserialize the component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]], chunks=List[StreamingChunk])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        guardrails: bool = False,
        stream: bool = False,
    ):
        """
        Generate text using the watsonx.ai model.
        
        Args:
            prompt: The input prompt string
            generation_kwargs: Additional generation parameters
            guardrails: Enable HAP content filtering
            stream: Enable streaming response
            
        Returns:
            Dictionary containing:
            - replies: List of generated texts
            - meta: List of metadata dictionaries
            - chunks: List of streaming chunks (if streaming)
        """
        # Handle empty prompt
        if not prompt.strip():
            return self._create_empty_response()

        # Merge generation parameters
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        try:
            if stream:
                return self._handle_streaming(prompt, merged_kwargs, guardrails)
            return self._handle_standard(prompt, merged_kwargs, guardrails)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_streaming(
        self,
        prompt: str,
        generation_kwargs: Dict[str, Any],
        guardrails: bool
    ) -> Dict[str, Any]:
        """Handle streaming generation."""
        try:
            stream = self.client.generate_text_stream(
                prompt=prompt,
                params=generation_kwargs,
                guardrails=guardrails,
                raw_response=True
            )

            chunks = []
            full_text = ""

            for chunk in stream:
                if not isinstance(chunk, dict):
                    continue
                
                chunk_text = chunk.get('results', [{}])[0].get('generated_text', '')
                if chunk_text:
                    full_text += chunk_text
                    chunks.append(StreamingChunk(
                        content=chunk_text,
                        meta={
                            "model": self.model,
                            "finish_reason": chunk.get("stop_reason", "streaming")
                        }
                    ))

            return {
                "replies": [full_text] if full_text else ["[No content]"],
                "meta": [{
                    "model": self.model,
                    "finish_reason": "completed",
                    "chunk_count": len(chunks)
                }],
                "chunks": chunks
            }
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_standard(
        self,
        prompt: str,
        generation_kwargs: Dict[str, Any],
        guardrails: bool
    ) -> Dict[str, Any]:
        """Handle standard (non-streaming) generation."""
        try:
            response = self.client.generate_text(
                prompt=prompt,
                params=generation_kwargs,
                guardrails=guardrails,
                raw_response=True
            )

            reply = response.get('results', [{}])[0].get('generated_text', '')
            return {
                "replies": [reply] if reply.strip() else ["[Empty response]"],
                "meta": [{
                    "model": self.model,
                    "finish_reason": response.get("stop_reason", "completed")
                }],
                "chunks": []
            }
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return self._create_error_response(str(e))

    def _create_empty_response(self) -> Dict[str, Any]:
        """Create response for empty input."""
        return {
            "replies": [""],
            "meta": [{"model": self.model, "finish_reason": "empty_input"}],
            "chunks": []
        }

    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create response for error cases."""
        return {
            "replies": [f"[Error: {error_msg}]"],
            "meta": [{
                "model": self.model,
                "finish_reason": "error",
                "error": error_msg
            }],
            "chunks": []
        }