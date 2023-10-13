from typing import Any, Dict, Optional, Union
from pathlib import Path
import datetime
import logging
import canals

from haystack.preview.telemetry import pipeline_running


logger = logging.getLogger(__name__)


class Pipeline(canals.Pipeline):
    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        max_loops_allowed: int = 100,
        debug_path: Union[Path, str] = Path(".haystack_debug/"),
    ):
        """
        Creates the Pipeline.

        Args:
            metadata: arbitrary dictionary to store metadata about this pipeline. Make sure all the values contained in
                this dictionary can be serialized and deserialized if you wish to save this pipeline to file with
                `save_pipelines()/load_pipelines()`.
            max_loops_allowed: how many times the pipeline can run the same node before throwing an exception.
            debug_path: when debug is enabled in `run()`, where to save the debug data.
        """
        self._telemetry_runs = 0
        self._last_telemetry_sent: Optional[datetime.datetime] = None
        super().__init__(metadata=metadata, max_loops_allowed=max_loops_allowed, debug_path=debug_path)

    def run(self, data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """
        Runs the pipeline.

        :params data: the inputs to give to the input components of the Pipeline.
        :params debug: whether to collect and return debug information.

        :returns: A dictionary with the outputs of the output components of the Pipeline.

        :raises PipelineRuntimeError: if the any of the components fail or return unexpected output.
        """
        pipeline_running(self)
        return super().run(data=data, debug=debug)
