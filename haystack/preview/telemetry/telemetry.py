import os
from typing import Any, Dict, Optional, TYPE_CHECKING
from pathlib import Path
import logging
import uuid
import json
import platform
import sys

import yaml
import posthog

import haystack
from haystack.preview.telemetry.environment import is_containerized

if TYPE_CHECKING:
    from haystack.preview.pipeline import Pipeline


HAYSTACK_TELEMETRY_ENABLED = "HAYSTACK_TELEMETRY_ENABLED"
CONFIG_PATH = Path("~/.haystack/config.yaml").expanduser()


logger = logging.getLogger(__name__)


class Telemetry:
    """
    Haystack reports anonymous usage statistics to support continuous software improvements for all its users.

    You can opt-out of sharing usage statistics by manually setting the environment
    variable `HAYSTACK_TELEMETRY_ENABLED` as described for different operating systems on the
    [documentation page](https://docs.haystack.deepset.ai/docs/telemetry#how-can-i-opt-out).

    Check out the documentation for more details: [Telemetry](https://docs.haystack.deepset.ai/docs/telemetry).
    """

    def __init__(self):
        """
        Initializes the telemetry. Loads the user_id from the config file,
        or creates a new id and saves it if the file is not found.

        It also collects system information which cannot change across the lifecycle
        of the process (for example `is_containerized()`).
        """

        # disable posthog logging
        for module_name in ["posthog", "backoff"]:
            logging.getLogger(module_name).setLevel(logging.CRITICAL)
            # Prevent module from sending errors to stderr when an exception is encountered during an emit() call
            logging.getLogger(module_name).addHandler(logging.NullHandler())
            logging.getLogger(module_name).propagate = False

        self.user_id = None

        if CONFIG_PATH.exists():
            # Load the config file
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file)
                    if "user_id" in config:
                        self.user_id = config["user_id"]
            except Exception as e:
                logger.debug("Telemetry could not read the config file %s", CONFIG_PATH, exc_info=e)
        else:
            # Create the config file
            logger.info(
                "Haystack sends anonymous usage data to understand the actual usage and steer dev efforts "
                "towards features that are most meaningful to users. You can opt-out at anytime by manually "
                "setting the environment variable HAYSTACK_TELEMETRY_ENABLED as described for different "
                "operating systems in the [documentation page](https://docs.haystack.deepset.ai/docs/telemetry#how-can-i-opt-out). "
                "More information at [Telemetry](https://docs.haystack.deepset.ai/docs/telemetry)."
            )
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
            self.user_id = str(uuid.uuid4())
            try:
                with open(CONFIG_PATH, "w") as outfile:
                    yaml.dump({"user_id": self.user_id}, outfile, default_flow_style=False)
            except Exception as e:
                logger.debug("Telemetry could not write config file to %s", CONFIG_PATH, exc_info=e)

        self.event_properties = self.collect_static_system_specs()

    def collect_static_system_specs(self) -> Dict[str, Any]:
        """
        Collects meta data about the setup that is used with Haystack, such as:
        operating system, python version, Haystack version, transformers version,
        pytorch version, number of GPUs, execution environment.
        """
        specs = {
            "libraries.haystack": haystack.__version__,
            "os.containerized": is_containerized(),
            "os.version": platform.release(),
            "os.family": platform.system(),
            "os.machine": platform.machine(),
            "python.version": platform.python_version(),
            "hardware.cpus": os.cpu_count(),
            "libraries.transformers": False,
            "libraries.torch": False,
            "libraries.cuda": False,
            "hardware.gpus": 0,
        }

        # Try to find out transformer's version
        try:
            import transformers

            specs["libraries.transformers"] = transformers.__version__
        except ImportError:
            pass

        # Try to find out torch's version and info on potential GPU(s)
        try:
            import torch

            specs["libraries.torch"] = torch.__version__
            if torch.cuda.is_available():
                specs["libraries.cuda"] = torch.version.cuda
                specs["libraries.gpus"] = torch.cuda.device_count()
        except ImportError:
            pass
        return specs

    def collect_dynamic_system_specs(self) -> Dict[str, Any]:
        """
        Collects meta data about the setup that is used with Haystack, such as:
        operating system, python version, Haystack version, transformers version,
        pytorch version, number of GPUs, execution environment.
        """
        return {
            "libraries.pytest": sys.modules["pytest"].__version__ if "pytest" in sys.modules.keys() else False,
            "libraries.ipython": sys.modules["ipython"].__version__ if "ipython" in sys.modules.keys() else False,
            "libraries.colab": sys.modules["google.colab"].__version__
            if "google.colab" in sys.modules.keys()
            else False,
        }

    def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]] = None):
        """
        Sends a telemetry event.

        :param event_name: The name of the event to show in PostHog.
        :param event_properties: Additional event metadata. These are merged with the
            system metadata collected in __init__, so take care not to overwrite them.
        """
        event_properties = event_properties or {}
        dynamic_specs = self.collect_dynamic_system_specs()
        try:
            posthog.capture(
                distinct_id=self.user_id,
                event=event_name,
                # loads/dumps to sort the keys
                properties=json.loads(
                    json.dumps({**self.event_properties, **dynamic_specs, **event_properties}, sort_keys=True)
                ),
            )
        except Exception as e:
            logger.debug("Telemetry couldn't make a POST request to PostHog.", exc_info=e)


def send_event(event_name: str, event_properties: Optional[Dict[str, Any]] = None):
    """
    Send a telemetry event, if telemetry is enabled.
    """
    try:
        if telemetry:
            telemetry.send_event(event_name=event_name, event_properties=event_properties)
    except Exception as e:
        # Never let telemetry break things
        logger.debug("There was an issue sending a '%s' telemetry event", event_name, exc_info=e)


def send_pipeline_run_event(pipeline: "Pipeline"):
    """
    Send a telemetry event for Pipeline.run(), if telemetry is enabled.
    """
    try:
        if telemetry:
            pipeline._telemetry_runs += 1
            if pipeline._telemetry_runs in [1, 10, 100, 1000] or pipeline._telemetry_runs % 10_000 == 0:
                pipeline_description = pipeline.to_dict()
                components = {}
                for component_name, component in pipeline_description["components"].items():
                    components[component_name] = component["type"]
                send_event("Pipeline run (2.x)", {"components": components, "runs": pipeline._telemetry_runs})
    except Exception as e:
        # Never let telemetry break things
        logger.debug("There was an issue sending a 'Pipeline run (2.x)' telemetry event", exc_info=e)


def tutorial_running(tutorial_id: str):
    """
    Send a telemetry event for a tutorial, if telemetry is enabled.
    :param tutorial_id: identifier of the tutorial
    """
    send_event(event_name="Tutorial", event_properties={"tutorial.id": tutorial_id})


if os.environ.get(HAYSTACK_TELEMETRY_ENABLED, "True") == "False":
    telemetry = None  # type: ignore
else:
    telemetry = Telemetry()
