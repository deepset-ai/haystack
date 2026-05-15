# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Literal

from openai import OpenAI
from openai.types.image import Image

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret
from haystack.utils.http_client import init_http_client

logger = logging.getLogger(__name__)


@component
class DALLEImageGenerator:
    """
    Generates images using OpenAI's image generation models such as `gpt-image-2`.

    For details on OpenAI API parameters, see
    [OpenAI documentation](https://developers.openai.com/api/reference/resources/images/methods/generate).

    ### Usage example
    ```python
    from haystack.components.generators import DALLEImageGenerator
    image_generator = DALLEImageGenerator()
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
        Creates an instance of DALLEImageGenerator. Unless specified otherwise in `model`, uses OpenAI's gpt-image-2.

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

        self.timeout = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        self.http_client_kwargs = http_client_kwargs

        self.client: OpenAI | None = None

    def warm_up(self) -> None:
        """
        Warm up the OpenAI client.
        """
        if self.client is None:
            self.client = OpenAI(
                api_key=self.api_key.resolve_value(),
                organization=self.organization,
                base_url=self.api_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=init_http_client(self.http_client_kwargs, async_client=False),
            )

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
        if self.client is None:
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
    def from_dict(cls, data: dict[str, Any]) -> "DALLEImageGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        return default_from_dict(cls, data)
