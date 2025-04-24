# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI
from openai.types.image import Image

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.http_client import init_http_client


@component
class DALLEImageGenerator:
    """
    Generates images using OpenAI's DALL-E model.

    For details on OpenAI API parameters, see
    [OpenAI documentation](https://platform.openai.com/docs/api-reference/images/create).

    ### Usage example

    ```python
    from haystack.components.generators import DALLEImageGenerator
    image_generator = DALLEImageGenerator()
    response = image_generator.run("Show me a picture of a black cat.")
    print(response)
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model: str = "dall-e-3",
        quality: Literal["standard", "hd"] = "standard",
        size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        response_format: Literal["url", "b64_json"] = "url",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of DALLEImageGenerator. Unless specified otherwise in `model`, uses OpenAI's dall-e-3.

        :param model: The model to use for image generation. Can be "dall-e-2" or "dall-e-3".
        :param quality: The quality of the generated image. Can be "standard" or "hd".
        :param size: The size of the generated images.
            Must be one of 256x256, 512x512, or 1024x1024 for dall-e-2.
            Must be one of 1024x1024, 1792x1024, or 1024x1792 for dall-e-3 models.
        :param response_format: The format of the response. Can be "url" or "b64_json".
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
        self.quality = quality
        self.size = size
        self.response_format = response_format
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.organization = organization

        self.timeout = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        self.http_client_kwargs = http_client_kwargs

        self.client: Optional[OpenAI] = None

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

    @component.output_types(images=List[str], revised_prompt=str)
    def run(
        self,
        prompt: str,
        size: Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] = None,
        quality: Optional[Literal["standard", "hd"]] = None,
        response_format: Optional[Optional[Literal["url", "b64_json"]]] = None,
    ):
        """
        Invokes the image generation inference based on the provided prompt and generation parameters.

        :param prompt: The prompt to generate the image.
        :param size: If provided, overrides the size provided during initialization.
        :param quality: If provided, overrides the quality provided during initialization.
        :param response_format: If provided, overrides the response format provided during initialization.

        :returns:
            A dictionary containing the generated list of images and the revised prompt.
            Depending on the `response_format` parameter, the list of images can be URLs or base64 encoded JSON strings.
            The revised prompt is the prompt that was used to generate the image, if there was any revision
            to the prompt made by OpenAI.
        """
        if self.client is None:
            raise RuntimeError(
                "The component DALLEImageGenerator wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        size = size or self.size
        quality = quality or self.quality
        response_format = response_format or self.response_format
        response = self.client.images.generate(
            model=self.model, prompt=prompt, size=size, quality=quality, response_format=response_format, n=1
        )
        if response.data is not None:
            image: Image = response.data[0]
            image_str = image.url or image.b64_json or ""
            revised_prompt = image.revised_prompt or ""
        else:
            image_str = ""
            revised_prompt = ""
        return {"images": [image_str], "revised_prompt": revised_prompt}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(  # type: ignore
            self,
            model=self.model,
            quality=self.quality,
            size=self.size,
            response_format=self.response_format,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
            organization=self.organization,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DALLEImageGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, keys=["api_key"])
        return default_from_dict(cls, data)  # type: ignore
