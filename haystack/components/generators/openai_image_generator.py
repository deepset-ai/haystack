# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI
from openai.types.image import Image

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components._openai_client_mixin import OpenAIClientMixin
from haystack.utils import Secret

logger = logging.getLogger(__name__)


@component
class OpenAIImageGenerator(OpenAIClientMixin):
    """
    Generates images using OpenAI's image generation models such as `gpt-image-2`.

    For details on OpenAI API parameters, see
    [OpenAI documentation](https://developers.openai.com/api/reference/resources/images/methods/generate).

    ### Usage example
    ```python
    from haystack.components.generators import OpenAIImageGenerator
    image_generator = OpenAIImageGenerator()
    response = image_generator.run("Show me a picture of a black cat.")
    print(response)
    ```
    """

    def __init__(
        self,
        model: str = "gpt-image-2",
        quality: Literal["auto", "high", "medium", "low"] = "auto",
        size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] = "1024x1024",
        response_format: Literal["b64_json"] = "b64_json",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of OpenAIImageGenerator. Unless specified otherwise in `model`, uses OpenAI's gpt-image-2.

        :param model: The model to use for image generation. Model names can be found in the
            [OpenAI documentation](https://developers.openai.com/api/docs/models/all).
        :param quality: The quality of the generated image. Can be "auto", "high", "medium", or "low".
        :param size: The size of the generated images. One of 1024x1024, 1024x1536, 1536x1024, or "auto".
            `gpt-image-2` also supports arbitrary sizes. You can find more information about supported sizes in
            the [OpenAI documentation](https://developers.openai.com/api/reference/resources/images/methods/generate).
        :param response_format: This parameter is ignored and only kept for backward compatibility.
        :param api_key: The OpenAI API key to connect to OpenAI.
        :param api_base_url: An optional base URL.
        :param organization: The Organization ID, defaults to `None`.
        :param timeout:
            Timeout for OpenAI Client calls. If not set, it is inferred from the `OPENAI_TIMEOUT` environment variable
            or set to 30.
        :param max_retries:
            Maximum retries to establish contact with OpenAI if it returns an internal error. If not set, it is inferred
            from the `OPENAI_MAX_RETRIES` environment variable or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        self.model = model
        if quality not in ["auto", "high", "medium", "low"]:
            logger.warning("Invalid quality: {quality}. Defaulting to 'auto'.", quality=quality)
            quality = "auto"
        self.quality = quality
        self.size = size
        if response_format != "b64_json":
            logger.warning("response_format is ignored. A base64-encoded image will be returned.")
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.organization = organization

        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs

        self.client: OpenAI | None = None
        self.async_client: AsyncOpenAI | None = None

    @component.output_types(images=list[str], revised_prompt=str)
    def run(
        self,
        prompt: str,
        size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = None,
        quality: Literal["auto", "high", "medium", "low"] | None = None,
        response_format: Literal["b64_json"] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Invokes the image generation inference based on the provided prompt and generation parameters.

        :param prompt: The prompt to generate the image.
        :param size: If provided, overrides the size provided during initialization.
        :param quality: If provided, overrides the quality provided during initialization.
        :param response_format: This parameter is ignored and only kept for backward compatibility.

        :returns:
            A dictionary containing the generated list of images as base64 encoded JSON strings and the revised prompt.
            The revised prompt is the prompt that was used to generate the image, if there was any revision
            to the prompt made by OpenAI.
        """
        self.warm_up()

        # at this point the client is initialized, but mypy doesn't know that
        assert self.client is not None

        size = size or self.size
        quality = quality or self.quality
        response = self.client.images.generate(model=self.model, prompt=prompt, size=size, quality=quality, n=1)
        image_str = ""
        revised_prompt = ""
        if response.data is not None:
            image: Image = response.data[0]
            image_str = image.b64_json or ""
            revised_prompt = image.revised_prompt or ""

        return {"images": [image_str], "revised_prompt": revised_prompt}

    @component.output_types(images=list[str], revised_prompt=str)
    async def run_async(
        self,
        prompt: str,
        size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = None,
        quality: Literal["auto", "high", "medium", "low"] | None = None,
        response_format: Literal["b64_json"] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Asynchronously invokes the image generation inference based on the provided prompt and generation parameters.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in an async code.

        :param prompt: The prompt to generate the image.
        :param size: If provided, overrides the size provided during initialization.
        :param quality: If provided, overrides the quality provided during initialization.
        :param response_format: This parameter is ignored and only kept for backward compatibility.

        :returns:
            A dictionary containing the generated list of images as base64 encoded JSON strings and the revised prompt.
            The revised prompt is the prompt that was used to generate the image, if there was any revision
            to the prompt made by OpenAI.
        """
        await self.warm_up_async()

        # at this point the client is initialized, but mypy doesn't know that
        assert self.async_client is not None

        size = size or self.size
        quality = quality or self.quality
        response = await self.async_client.images.generate(
            model=self.model, prompt=prompt, size=size, quality=quality, n=1
        )
        image_str = ""
        revised_prompt = ""
        if response.data is not None:
            image: Image = response.data[0]
            image_str = image.b64_json or ""
            revised_prompt = image.revised_prompt or ""

        return {"images": [image_str], "revised_prompt": revised_prompt}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            quality=self.quality,
            size=self.size,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            organization=self.organization,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAIImageGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        return default_from_dict(cls, data)
