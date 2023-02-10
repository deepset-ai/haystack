import os
from typing import Any, Dict, Optional, List, Union
import uuid
import logging
from pathlib import Path
import json
import yaml

import posthog

from haystack.environment import collect_static_system_specs, collect_dynamic_system_specs

HAYSTACK_TELEMETRY_ENABLED = "HAYSTACK_TELEMETRY_ENABLED"
HAYSTACK_EXECUTION_CONTEXT = "HAYSTACK_EXECUTION_CONTEXT"
HAYSTACK_DOCKER_CONTAINER = "HAYSTACK_DOCKER_CONTAINER"
CONFIG_PATH = Path("~/.haystack/config.yaml").expanduser()
LOG_PATH = Path("~/.haystack/telemetry.log").expanduser()


logger = logging.getLogger(__name__)


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

        self.event_properties = collect_static_system_specs()

    def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]] = None):
        """
        Sends an event.

        :param event_name: the name of the event to show in PostHog
        :param event_properties: additional event metadata. These will be merged with the
            system metadata collected in __init__, so take care not to overwrite them.
        """
        event_properties = event_properties or {}
        dynamic_specs = collect_dynamic_system_specs()
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
            logger.debug("Telemetry was not able to make a POST request to posthog.", exc_info=e)


def sent_pipeline_run_event(
    event_name: str,
    pipeline: "Pipeline",
    query: Optional[str] = None,
    queries: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
    labels: Optional[Union["MultiLabel", List["MultiLabel"]]] = None,
    documents: Optional[Union[List["Document"], List[List["Document"]]]] = None,
    meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    params: Optional[dict] = None,
    debug: Optional[bool] = None,
):
    try:
        global telemetry
        if telemetry:
            event_properties = {}

            # Check if it's the public demo
            exec_context = os.environ.get(HAYSTACK_EXECUTION_CONTEXT, "")
            if exec_context == "public_demo":
                event_properties["pipeline.is_public_demo"] = True
                event_properties["pipeline.run_parameters.query"] = query
                event_properties["pipeline.run_parameters.params"] = params
                telemetry.send_event(event_name=event_name, event_properties=event_properties)
                return

            # Collect pipeline profile
            event_properties["pipeline.classname"] = pipeline.__class__.__name__
            event_properties["pipeline.fingerprint"] = pipeline.fingerprint
            if pipeline.yaml_hash:
                event_properties["pipeline.yaml_hash"] = pipeline.yaml_hash

            # Add document store
            docstore = pipeline.get_document_store()
            if docstore:
                event_properties["pipeline.document_store"] = docstore.__class__.__name__

            # Add an entry for each node class and classify the pipeline by its root node
            for node in pipeline.graph.nodes:
                node_type = pipeline.graph.nodes.get(node)["component"].__class__.__name__
                if node_type == "RootNode":
                    event_properties["pipeline.type"] = node
                else:
                    event_properties["pipeline.nodes." + node_type] = (
                        event_properties.get("pipeline.nodes." + node_type, 0) + 1
                    )

            # Inputs of the run() or run_batch() call
            if isinstance(labels, list):
                labels_len = len(labels)
            else:
                labels_len = 1 if labels else 0
            if documents and isinstance(documents, list) and isinstance(documents[0], list):
                documents_len = [len(docs) if isinstance(docs, list) else 0 for docs in documents]
            elif isinstance(documents, list):
                documents_len = [len(documents)]
            else:
                documents_len = [0]
            if meta and isinstance(meta, list):
                meta_len = len(meta)
            else:
                meta_len = 1
            event_properties["pipeline.run_parameters.queries"] = len(queries) if queries else bool(query)
            event_properties["pipeline.run_parameters.file_paths"] = len(file_paths or [])
            event_properties["pipeline.run_parameters.labels"] = labels_len
            event_properties["pipeline.run_parameters.documents"] = documents_len
            event_properties["pipeline.run_parameters.meta"] = meta_len
            event_properties["pipeline.run_parameters.params"] = bool(params)
            event_properties["pipeline.run_parameters.debug"] = bool(debug)

            telemetry.send_event(event_name=event_name, event_properties=event_properties)
    except Exception as e:
        # Never let telemetry break things
        logger.debug("There was an issue sending a %s telemetry event", event_name, exc_info=e)


def sent_pipeline_event(pipeline: "Pipeline", event_name: str):
    try:
        global telemetry
        if telemetry:
            telemetry.send_event(
                event_name=event_name,
                event_properties={
                    "pipeline.classname": pipeline.__class__.__name__,
                    "pipeline.fingerprint": pipeline.fingerprint,
                    "pipeline.yaml_hash": pipeline.yaml_hash,
                },
            )
    except Exception as e:
        # Never let telemetry break things
        logger.debug("There was an issue sending a '%s' telemetry event", event_name, exc_info=e)


def send_event(event_name: str, event_properties: Optional[Dict[str, Any]] = None):
    try:
        global telemetry
        if telemetry:
            telemetry.send_event(event_name=event_name, event_properties=event_properties)
    except Exception as e:
        # Never let telemetry break things
        logger.debug("There was an issue sending a '%s' telemetry event", event_name, exc_info=e)


if os.environ.get("HAYSTACK_TELEMETRY_VERSION", "2") == "2":
    telemetry = Telemetry()
else:
    telemetry = None  # type: ignore
