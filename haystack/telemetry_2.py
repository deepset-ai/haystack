import os
from typing import Any, Dict, Optional
import uuid
import logging
from logging import CRITICAL
from pathlib import Path

import yaml
import posthog

from haystack.environment import collect_system_specs

posthog.api_key = "phc_C44vUK9R1J6HYVdfJarTEPqVAoRPJzMXzFcj8PIrJgP"
posthog.host = "https://eu.posthog.com"
HAYSTACK_TELEMETRY_ENABLED = "HAYSTACK_TELEMETRY_ENABLED"
HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED = "HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED"
CONFIG_PATH = Path("~/.haystack/config.yaml").expanduser()
LOG_PATH = Path("~/.haystack/telemetry.log").expanduser()


logger = logging.getLogger(__name__)


# disable posthog logging
for module_name in ["posthog", "backoff"]:
    logging.getLogger(module_name).setLevel(CRITICAL)
    # Prevent module from sending errors to stderr when an exception is encountered during an emit() call
    logging.getLogger(module_name).addHandler(logging.NullHandler())
    logging.getLogger(module_name).propagate = False


class Telemetry:
    """
    Haystack reports anonymous usage statistics to support continuous software improvements for all its users.
    An example report can be inspected via calling print_report().

    You can opt-out of sharing usage statistics by calling disable() or by manually setting the environment
    variable HAYSTACK_TELEMETRY_ENABLED as described for different operating systems on the documentation page.
    You can log all events to the local file specified in LOG_PATH for inspection by setting the environment
    variable HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED to "True".

    Check out the documentation for more details: https://docs.haystack.deepset.ai/docs/telemetry
    """

    def __init__(self):
        """
        Initializes the telemetry. Loads the user_id from the config file,
        or creates a new id and saves it if the file is not found.

        It also collects system information which cannot change across the lifecycle
        of the process (for example `is_containerized()`)
        """
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
                "towards features that are most meaningful to users. You can opt-out at anytime by calling "
                "disable() or by manually setting the environment variable  "
                "HAYSTACK_TELEMETRY_ENABLED as described for different operating systems on the documentation "
                "page. More information at https://docs.haystack.deepset.ai/docs/telemetry"
            )
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
            self.user_id = str(uuid.uuid4())
            try:
                with open(CONFIG_PATH, "w") as outfile:
                    yaml.dump({"user_id": self.user_id}, outfile, default_flow_style=False)
            except Exception as e:
                logger.debug("Telemetry could not write config file to %s", CONFIG_PATH, exc_info=e)

        self.event_properties = collect_system_specs()

    def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]] = None):
        """
        Sends an event.

        :param event_name: the name of the event to show in PostHog
        :param event_properties: additional event metadata. These will be merged with the
            system metadata collected in __init__, so take care not to overwrite them.
        """
        event_properties = event_properties or {}
        if self.is_enabled():
            try:
                posthog.capture(
                    distinct_id=self.user_id, event=event_name, properties={**self.event_properties, **event_properties}
                )
            except Exception as e:
                logger.debug("Telemetry was not able to make a post request to posthog.", exc_info=e)

            if self.is_logging_to_file_enabled():
                self.write_event_to_log_file(user_id=self.user_id, event=event_name, properties=event_properties)

    def print_report(self):
        """
        Prints the user id and the meta data that are sent in events
        """
        if self.is_enabled():
            user_id = self.user_id
            print({"user_id": user_id})
        else:
            print("Telemetry is disabled.")

    def enable(self):
        """
        Enables telemetry so that a limited amount of anonymous usage data is sent as events.
        """
        os.environ[HAYSTACK_TELEMETRY_ENABLED] = "True"
        logger.info("Telemetry has been enabled.")

    def disable(self):
        """
        Disables telemetry so that no events are sent anymore, except for one final event.
        """
        os.environ[HAYSTACK_TELEMETRY_ENABLED] = "False"
        logger.info("Telemetry has been disabled.")

    def enable_writing_events_to_file(self):
        """
        Enables writing each event that is sent to the log file specified in LOG_PATH
        """
        os.environ[HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED] = "True"
        logger.info("Writing events to log file %s has been enabled.", LOG_PATH)

    def disable_writing_events_to_file(self):
        """
        Disables writing each event that is sent to the log file specified in LOG_PATH
        """
        os.environ[HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED] = "False"
        logger.info("Writing events to log file %s has been disabled.", LOG_PATH)

    def is_enabled(self) -> bool:
        """
        Returns False if telemetry is disabled via an environment variable, otherwise True.
        """
        telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_ENABLED, "True")
        return telemetry_environ.lower() != "false"

    def is_logging_to_file_enabled(self) -> bool:
        """
        Returns False if logging telemetry events to a file is disabled via an environment variable, otherwise True.
        """
        telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED, "False")
        return telemetry_environ.lower() != "false"

    def write_event_to_log_file(self, user_id: str, event: str, properties: Dict[str, Any]):
        try:
            with open(LOG_PATH, "a") as file_object:
                file_object.write(f"{user_id}: {event}, {properties}\n")
        except Exception as e:
            logger.debug("Telemetry was not able to write event to log file %s", LOG_PATH, exc_info=e)


telemetry = Telemetry()
print("Telemetry 2!")
