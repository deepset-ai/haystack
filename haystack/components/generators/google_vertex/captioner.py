from typing import List, Dict, Optional, Any
import logging

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.core.component import component
from haystack.dataclasses.byte_stream import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install google-cloud-aiplatform'") as vertexai_import:
    import vertexai
    from vertexai.vision_models import ImageTextModel, Image

logger = logging.getLogger(__name__)


@component
class VertexAIImageCaptioner:
    def __init__(self, *, model: str = "imagetext", project_id: str, location: Optional[str] = None, **kwargs):
        """
        Generate image captions using a Google Vertex AI model.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official Google documentation:
        https://cloud.google.com/docs/authentication/provide-credentials-adc

        :param project_id: ID of the GCP project to use.
        :param model: Name of the model to use, defaults to "imagetext".
        :param location: The default location to use when making API calls, if not set uses us-central-1.
            Defaults to None.
        :param kwargs: Additional keyword arguments to pass to the model.
            For a list of supported arguments see the `ImageTextModel.get_captions()` documentation.
        """
        vertexai_import.check()

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai.init(project=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._location = location
        self._kwargs = kwargs

        self._model = ImageTextModel.from_pretrained(self._model_name)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self, model=self._model_name, project_id=self._project_id, location=self._location, **self._kwargs
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIImageCaptioner":
        return default_from_dict(cls, data)

    @component.output_types(captions=List[str])
    def run(self, image: ByteStream):
        captions = self._model.get_captions(image=Image(image.data), **self._kwargs)
        return {"captions": captions}
