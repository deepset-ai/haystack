from typing import Any, Dict
from pathlib import Path
import logging
import canals

from haystack.preview.telemetry import send_event


logger = logging.getLogger(__name__)


class Pipeline(canals.Pipeline):
    def run(self, data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """
        Runs the pipeline.

        :params data: the inputs to give to the input components of the Pipeline.
        :params debug: whether to collect and return debug information.

        :returns: A dictionary with the outputs of the output components of the Pipeline.

        :raises PipelineRuntimeError: if the any of the components fail or return unexpected output.
        """
        try:
            pipeline_description = self.to_dict()
            components = {}
            for component_name, component in pipeline_description["components"].items():
                components[component_name] = component["type"]
            send_event("Pipeline (2.x)", {"components": components})
        except Exception as e:
            logger.warning(f"Error sending telemetry event: {e}")

        return super().run(data=data, debug=debug)
