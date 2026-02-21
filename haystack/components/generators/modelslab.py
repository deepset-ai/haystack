# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""ModelsLab image generation component for Haystack pipelines."""

import time
from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

MODELSLAB_BASE_URL = "https://modelslab.com/api/v6"
MODELSLAB_TEXT2IMG_URL = f"{MODELSLAB_BASE_URL}/images/text2img"
MODELSLAB_FETCH_URL = f"{MODELSLAB_BASE_URL}/images/fetch"
MODELSLAB_POLLING_INTERVAL = 3  # seconds
MODELSLAB_POLLING_TIMEOUT = 300  # seconds (5 minutes)


@component
class ModelsLabImageGenerator:
    """
    Generates images using ModelsLab's text-to-image API.

    ModelsLab provides access to 200+ AI models including Flux, SDXL, Stable Diffusion,
    and thousands of community fine-tunes via a single unified API.

    For details on ModelsLab API parameters, see
    [ModelsLab documentation](https://docs.modelslab.com/image-generation/community-models/text2img).

    ### Usage example

    ```python
    from haystack.components.generators import ModelsLabImageGenerator

    generator = ModelsLabImageGenerator(model="flux")
    response = generator.run("A breathtaking sunset over snow-capped mountains.")
    print(response["images"])  # list of image URLs
    ```

    ### Pipeline usage

    ```python
    from haystack import Pipeline
    from haystack.components.generators import ModelsLabImageGenerator

    pipeline = Pipeline()
    pipeline.add_component("image_gen", ModelsLabImageGenerator(model="flux", width=1024, height=1024))
    result = pipeline.run({"image_gen": {"prompt": "A futuristic city at night"}})
    print(result["image_gen"]["images"])
    ```
    """

    def __init__(
        self,
        model: str = "flux",
        width: int = 512,
        height: int = 512,
        samples: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        api_key: Secret = Secret.from_env_var("MODELSLAB_API_KEY"),
        api_base_url: Optional[str] = None,
    ):
        """
        Creates an instance of ModelsLabImageGenerator.

        :param model: The model to use for image generation. Defaults to ``"flux"``.
            Supports any ModelsLab community model ID, e.g. ``"sdxl"``, ``"realistic-vision-v51"``.
        :param width: Width of the generated image in pixels (must be divisible by 8). Defaults to ``512``.
        :param height: Height of the generated image in pixels (must be divisible by 8). Defaults to ``512``.
        :param samples: Number of images to generate per request (1–4). Defaults to ``1``.
        :param num_inference_steps: Number of denoising steps (1–50). Defaults to ``30``.
        :param guidance_scale: Classifier-free guidance scale (1–20). Higher values follow the prompt
            more closely. Defaults to ``7.5``.
        :param negative_prompt: Optional negative prompt to steer generation away from undesired content.
        :param seed: Optional random seed for reproducible results.
        :param api_key: The ModelsLab API key. Reads from the ``MODELSLAB_API_KEY`` environment variable
            by default.
        :param api_base_url: Optional custom base URL for the ModelsLab API. Defaults to
            ``https://modelslab.com/api/v6``.
        """
        self.model = model
        self.width = width
        self.height = height
        self.samples = samples
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.api_key = api_key
        self.api_base_url = api_base_url or MODELSLAB_BASE_URL

    def _build_url(self, path: str) -> str:
        return f"{self.api_base_url.rstrip('/')}/{path.lstrip('/')}"

    def _poll_for_result(self, generation_id: int, api_key: str) -> list[str]:
        """
        Poll the ModelsLab fetch endpoint until image generation completes.

        :param generation_id: The ModelsLab generation task ID.
        :param api_key: The resolved API key.
        :returns: List of generated image URLs.
        :raises TimeoutError: If polling exceeds MODELSLAB_POLLING_TIMEOUT seconds.
        :raises RuntimeError: If the API returns an error status.
        """
        try:
            import requests  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Could not import `requests`. Install it with `pip install requests`."
            ) from e

        fetch_url = self._build_url(f"images/fetch/{generation_id}")
        start = time.time()

        while True:
            if time.time() - start > MODELSLAB_POLLING_TIMEOUT:
                raise TimeoutError(
                    f"ModelsLab image generation timed out after {MODELSLAB_POLLING_TIMEOUT}s "
                    f"(generation_id={generation_id})."
                )

            response = requests.post(
                fetch_url,
                json={"key": api_key},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            status = data.get("status", "")

            if status == "success":
                return data.get("output", [])
            elif status == "error":
                raise RuntimeError(
                    f"ModelsLab image generation failed: {data.get('message', 'unknown error')}"
                )
            elif status == "processing":
                logger.debug(
                    f"ModelsLab: generation {generation_id} still processing, retrying in {MODELSLAB_POLLING_INTERVAL}s"
                )
                time.sleep(MODELSLAB_POLLING_INTERVAL)
            else:
                raise RuntimeError(f"ModelsLab: unexpected status '{status}' for generation {generation_id}")

    @component.output_types(images=list[str], metadata=list[dict])
    def run(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        samples: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Generate images from a text prompt using the ModelsLab API.

        :param prompt: The text prompt describing the image to generate.
        :param model: Overrides the model set at initialization.
        :param width: Overrides the width set at initialization.
        :param height: Overrides the height set at initialization.
        :param samples: Overrides the number of samples set at initialization.
        :param negative_prompt: Overrides the negative prompt set at initialization.
        :param seed: Overrides the seed set at initialization.
        :returns:
            A dictionary with:
            - ``images``: A list of image URLs (one per sample).
            - ``metadata``: A list of dicts with generation metadata (model, width, height, generation_time).
        """
        try:
            import requests  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Could not import `requests`. Install it with `pip install requests`."
            ) from e

        resolved_model = model or self.model
        resolved_width = width or self.width
        resolved_height = height or self.height
        resolved_samples = samples or self.samples
        resolved_negative_prompt = negative_prompt or self.negative_prompt
        resolved_seed = seed or self.seed
        api_key = self.api_key.resolve_value() or ""

        payload: dict[str, Any] = {
            "key": api_key,
            "prompt": prompt,
            "model_id": resolved_model,
            "width": resolved_width,
            "height": resolved_height,
            "samples": resolved_samples,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "safety_checker": "no",
        }
        if resolved_negative_prompt:
            payload["negative_prompt"] = resolved_negative_prompt
        if resolved_seed is not None:
            payload["seed"] = resolved_seed

        text2img_url = self._build_url("images/text2img")
        response = requests.post(
            text2img_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "")

        if status == "error":
            raise RuntimeError(f"ModelsLab error: {data.get('message', 'unknown error')}")

        if status == "processing":
            generation_id = data.get("id")
            if not generation_id:
                raise RuntimeError("ModelsLab returned 'processing' status without a generation ID.")
            logger.info(f"ModelsLab: generation {generation_id} is processing, polling for result...")
            image_urls = self._poll_for_result(generation_id=generation_id, api_key=api_key)
        elif status == "success":
            image_urls = data.get("output", [])
        else:
            raise RuntimeError(f"ModelsLab: unexpected response status '{status}'.")

        generation_time = data.get("generationTime")
        metadata = [
            {
                "model": resolved_model,
                "width": resolved_width,
                "height": resolved_height,
                "generation_time": generation_time,
            }
            for _ in image_urls
        ]

        return {"images": image_urls, "metadata": metadata}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            width=self.width,
            height=self.height,
            samples=self.samples,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelsLabImageGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        if api_key := data.get("init_parameters", {}).get("api_key"):
            data["init_parameters"]["api_key"] = Secret.from_dict(api_key)
        return default_from_dict(cls, data)
