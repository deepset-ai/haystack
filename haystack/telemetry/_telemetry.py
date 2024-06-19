# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import posthog
import yaml

from haystack import logging as haystack_logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.telemetry._environment import collect_system_specs

if TYPE_CHECKING:
    from haystack.core.pipeline import Pipeline


HAYSTACK_TELEMETRY_ENABLED = "HAYSTACK_TELEMETRY_ENABLED"
CONFIG_PATH = Path("~/.haystack/config.yaml").expanduser()

#: Telemetry sends at most one event every number of seconds specified in this constant
MIN_SECONDS_BETWEEN_EVENTS = 60


logger = haystack_logging.getLogger(__name__)


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
        Initializes the telemetry.

        Loads the user_id from the config file, or creates a new id and saves it if the file is not found.

        It also collects system information which cannot change across the lifecycle
        of the process (for example `is_containerized()`).
        """
        posthog.api_key = "phc_C44vUK9R1J6HYVdfJarTEPqVAoRPJzMXzFcj8PIrJgP"
        posthog.host = "https://eu.posthog.com"

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
                logger.debug(
                    "Telemetry could not read the config file {config_path}", config_path=CONFIG_PATH, exc_info=e
                )
        else:
            # Create the config file
            logger.info(
                "Haystack sends anonymous usage data to understand the actual usage and steer dev efforts "
                "towards features that are most meaningful to users. You can opt-out at anytime by manually "
                "setting the environment variable HAYSTACK_TELEMETRY_ENABLED as described for different "
                "operating systems in the "
                "[documentation page](https://docs.haystack.deepset.ai/docs/telemetry#how-can-i-opt-out). "
                "More information at [Telemetry](https://docs.haystack.deepset.ai/docs/telemetry)."
            )
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
            self.user_id = str(uuid.uuid4())
            try:
                with open(CONFIG_PATH, "w") as outfile:
                    yaml.dump({"user_id": self.user_id}, outfile, default_flow_style=False)
            except Exception as e:
                logger.debug(
                    "Telemetry could not write config file to {config_path}", config_path=CONFIG_PATH, exc_info=e
                )

        self.event_properties = collect_system_specs()

    def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]] = None):
        """
        Sends a telemetry event.

        :param event_name: The name of the event to show in PostHog.
        :param event_properties: Additional event metadata. These are merged with the
            system metadata collected in __init__, so take care not to overwrite them.
        """
        event_properties = event_properties or {}
        try:
            posthog.capture(
                distinct_id=self.user_id, event=event_name, properties={**self.event_properties, **event_properties}
            )
        except Exception as e:
            logger.debug("Telemetry couldn't make a POST request to PostHog.", exc_info=e)


def send_telemetry(func):
    """
    Decorator that sends the output of the wrapped function to PostHog.

    The wrapped function is actually called only if telemetry is enabled.
    """

    # FIXME? Somehow, functools.wraps makes `telemetry` out of scope. Let's take care of it later.
    def send_telemetry_wrapper(*args, **kwargs):
        try:
            if telemetry:
                output = func(*args, **kwargs)
                if output:
                    telemetry.send_event(*output)
        except Exception as e:
            # Never let telemetry break things
            logger.debug("There was an issue sending a telemetry event", exc_info=e)

    return send_telemetry_wrapper


@send_telemetry
def pipeline_running(pipeline: "Pipeline") -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Collects telemetry data for a pipeline run and sends it to Posthog.

    Collects name, type and the content of the _telemetry_data attribute, if present, for each component in the
    pipeline and sends such data to Posthog.

    :param pipeline: the pipeline that is running.
    """
    pipeline._telemetry_runs += 1
    if (
        pipeline._last_telemetry_sent
        and (datetime.datetime.now() - pipeline._last_telemetry_sent).seconds < MIN_SECONDS_BETWEEN_EVENTS
    ):
        return None

    pipeline._last_telemetry_sent = datetime.datetime.now()

    # Collect info about components
    components: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for component_name, instance in pipeline.walk():
        component_qualified_class_name = generate_qualified_class_name(type(instance))
        if hasattr(instance, "_get_telemetry_data"):
            telemetry_data = getattr(instance, "_get_telemetry_data")()
            if not isinstance(telemetry_data, dict):
                raise TypeError(
                    f"Telemetry data for component {component_name} must be a dictionary but is {type(telemetry_data)}."
                )
            components[component_qualified_class_name].append({"name": component_name, **telemetry_data})
        else:
            components[component_qualified_class_name].append({"name": component_name})

    # Data sent to Posthog
    return "Pipeline run (2.x)", {
        "pipeline_id": str(id(pipeline)),
        "runs": pipeline._telemetry_runs,
        "components": components,
    }


@send_telemetry
def tutorial_running(tutorial_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Send a telemetry event for a tutorial, if telemetry is enabled.

    :param tutorial_id: identifier of the tutorial
    """
    return "Tutorial", {"tutorial.id": tutorial_id}


telemetry = None
if os.getenv("HAYSTACK_TELEMETRY_ENABLED", "true").lower() in ("true", "1"):
    telemetry = Telemetry()
