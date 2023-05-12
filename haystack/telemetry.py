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
SEND_EVENT_EVERY_N_RUNS = 100


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

        self.event_properties = collect_static_system_specs()

    def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]] = None):
        """
        Sends a telemetry event.

        :param event_name: The name of the event to show in PostHog.
        :param event_properties: Additional event metadata. These are merged with the
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
            logger.debug("Telemetry couldn't make a POST request to PostHog.", exc_info=e)


def tutorial_running(tutorial_id: int):
    """
    Can be called when a tutorial is executed so that the tutorial_id is used to identify the tutorial and send an event.
    :param tutorial_id: ID number of the tutorial
    """
    send_event(event_name="Tutorial", event_properties={"tutorial.id": tutorial_id})


def send_pipeline_event(  # type: ignore
    pipeline: "Pipeline",  # type: ignore
    query: Optional[str] = None,
    queries: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
    labels: Optional[Union["MultiLabel", List["MultiLabel"]]] = None,  # type: ignore
    documents: Optional[Union[List["Document"], List[List["Document"]]]] = None,  # type: ignore
    meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    params: Optional[dict] = None,
    debug: Optional[bool] = None,
):
    """
    Sends a telemetry event about the execution of a pipeline, if telemetry is enabled.

    :param pipeline: the pipeline that is running
    :param query: the value of the `query` input of the pipeline, if any
    :param queries: the value of the `queries` input of the pipeline, if any
    :param file_paths: the value of the `file_paths` input of the pipeline, if any
    :param labels: the value of the `labels` input of the pipeline, if any
    :param documents: the value of the `documents` input of the pipeline, if any
    :param meta: the value of the `meta` input of the pipeline, if any
    :param params: the value of the `params` input of the pipeline, if any
    :param debug: the value of the `debug` input of the pipeline, if any
    """
    try:
        if telemetry:
            # Check if it's the public demo
            exec_context = os.environ.get(HAYSTACK_EXECUTION_CONTEXT, "")
            if exec_context == "public_demo":
                event_properties: Dict[str, Optional[Union[str, bool, int, Dict[str, Any]]]] = {
                    "pipeline.is_public_demo": True,
                    "pipeline.run_parameters.query": query,
                    "pipeline.run_parameters.params": params,
                }
                telemetry.send_event(event_name="Public Demo", event_properties=event_properties)
                return

            # If pipeline config has not changed, send an event every SEND_EVENT_EVERY_N_RUNS runs
            if pipeline.last_config_hash == pipeline.config_hash and pipeline.runs % SEND_EVENT_EVERY_N_RUNS == 0:
                event_properties = {"pipeline.config_hash": pipeline.config_hash, "pipeline.runs": pipeline.runs}
                telemetry.send_event(event_name="Pipeline", event_properties=event_properties)
                return
            pipeline.last_config_hash = pipeline.config_hash
            pipeline.runs = 1

            event_properties = {
                "pipeline.classname": pipeline.__class__.__name__,
                "pipeline.config_hash": pipeline.config_hash,
                "pipeline.runs": pipeline.runs,
            }

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
                        event_properties.get("pipeline.nodes." + node_type, 0) + 1  # type: ignore
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
            event_properties["pipeline.run_parameters.documents"] = documents_len  # type: ignore
            event_properties["pipeline.run_parameters.meta"] = meta_len
            event_properties["pipeline.run_parameters.params"] = bool(params)
            event_properties["pipeline.run_parameters.debug"] = bool(debug)

            telemetry.send_event(event_name="Pipeline", event_properties=event_properties)
    except Exception as e:
        # Never let telemetry break things
        logger.debug("There was an issue sending a 'Pipeline' telemetry event", exc_info=e)


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


if os.environ.get("HAYSTACK_TELEMETRY_ENABLED", "True") == "False":
    telemetry = None  # type: ignore
else:
    telemetry = Telemetry()
