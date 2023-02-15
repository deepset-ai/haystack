"""
    Telemetry
    Haystack reports anonymous usage statistics to support continuous software improvements for all its users.
    An example report can be inspected via calling print_telemetry_report(). Check out the documentation for more details: https://docs.haystack.deepset.ai/docs/telemetry
    You can opt-out of sharing usage statistics by calling disable_telemetry() or by manually setting the environment variable HAYSTACK_TELEMETRY_ENABLED as described for different operating systems on the documentation page.
    You can log all events to the local file specified in LOG_PATH for inspection by setting the environment variable HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED to "True".
"""
import os
from typing import Any, Dict, List, Optional
import uuid
import logging
from logging import CRITICAL
from enum import Enum
from functools import wraps
from pathlib import Path

import yaml
import posthog

from haystack.environment import HAYSTACK_EXECUTION_CONTEXT, get_or_create_env_meta_data

posthog.api_key = "phc_F5v11iI2YHkoP6Er3cPILWSrLhY3D6UY4dEMga4eoaa"
posthog.host = "https://tm.hs.deepset.ai"
HAYSTACK_TELEMETRY_ENABLED = "HAYSTACK_TELEMETRY_ENABLED"
HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED = "HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED"
CONFIG_PATH = Path("~/.haystack/config.yaml").expanduser()
LOG_PATH = Path("~/.haystack/telemetry.log").expanduser()

user_id: Optional[str] = None

logger = logging.getLogger(__name__)

# disable posthog logging
for module_name in ["posthog", "backoff"]:
    logging.getLogger(module_name).setLevel(CRITICAL)
    # Prevent module from sending errors to stderr when an exception is encountered during an emit() call
    logging.getLogger(module_name).addHandler(logging.NullHandler())
    logging.getLogger(module_name).propagate = False


class TelemetryFileType(Enum):
    LOG_FILE: str = "LOG_FILE"
    CONFIG_FILE: str = "CONFIG_FILE"


def print_telemetry_report():
    """
    Prints the user id and the meta data that are sent in events
    """
    if is_telemetry_enabled():
        user_id = _get_or_create_user_id()
        meta_data = get_or_create_env_meta_data()
        print({**{"user_id": user_id}, **meta_data})
    else:
        print("Telemetry is disabled.")


def enable_telemetry():
    """
    Enables telemetry so that a limited amount of anonymous usage data is sent as events.
    """
    os.environ[HAYSTACK_TELEMETRY_ENABLED] = "True"
    logger.info("Telemetry has been enabled.")


def disable_telemetry():
    """
    Disables telemetry so that no events are sent anymore, except for one final event.
    """
    os.environ[HAYSTACK_TELEMETRY_ENABLED] = "False"
    logger.info("Telemetry has been disabled.")


def enable_writing_events_to_file():
    """
    Enables writing each event that is sent to the log file specified in LOG_PATH
    """
    os.environ[HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED] = "True"
    logger.info("Writing events to log file %s has been enabled.", LOG_PATH)


def disable_writing_events_to_file():
    """
    Disables writing each event that is sent to the log file specified in LOG_PATH
    """
    os.environ[HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED] = "False"
    logger.info("Writing events to log file %s has been disabled.", LOG_PATH)


def is_telemetry_enabled() -> bool:
    """
    Returns False if telemetry is disabled via an environment variable, otherwise True.
    """
    telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_ENABLED, "True")
    return telemetry_environ.lower() != "false"


def is_telemetry_logging_to_file_enabled() -> bool:
    """
    Returns False if logging telemetry events to a file is disabled via an environment variable, otherwise True.
    """
    telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED, "False")
    return telemetry_environ.lower() != "false"


def send_event_if_public_demo(func):
    """
    Can be used as a decorator to send an event only if HAYSTACK_EXECUTION_CONTEXT is "public_demo"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        exec_context = os.environ.get(HAYSTACK_EXECUTION_CONTEXT, "")
        if exec_context == "public_demo":
            send_custom_event(event="demo query executed", payload=kwargs)
        return func(*args, **kwargs)

    return wrapper


def send_event(func):
    """
    Can be used as a decorator to send an event formatted like 'Pipeline.eval executed'
    with additional parameters as defined in TrackedParameters ('add_isolated_node_eval') and
    metadata, such as os_version
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        send_custom_event(event=f"{type(args[0]).__name__}.{func.__name__} executed", payload=kwargs)
        return func(*args, **kwargs)

    return wrapper


def send_custom_event(event: str = "", payload: Optional[Dict[str, Any]] = None):
    """
    This method can be called directly from anywhere in Haystack to send an event.
    Enriches the given event with metadata and sends it to the posthog server if telemetry is enabled.
    If telemetry has just been disabled, a final event is sent and the config file and the log file are deleted

    :param event: Name of the event. Use a noun and a verb, e.g., "evaluation started", "component created"
    :param payload: A dictionary containing event meta data, e.g., parameter settings
    """
    global user_id  # pylint: disable=global-statement
    if payload is None:
        payload = {}
    try:

        def send_request(payload: Dict[str, Any]):
            """
            Prepares and sends an event in a post request to a posthog server
            Sending the post request within posthog.capture is non-blocking.

            :param payload: A dictionary containing event meta data, e.g., parameter settings
            """
            event_properties = {**(NonPrivateParameters.apply_filter(payload)), **get_or_create_env_meta_data()}
            if user_id is None:
                raise RuntimeError("User id was not initialized")
            try:
                posthog.capture(distinct_id=user_id, event=event, properties=event_properties)
            except Exception as e:
                logger.debug("Telemetry was not able to make a post request to posthog.", exc_info=e)
            if is_telemetry_enabled() and is_telemetry_logging_to_file_enabled():
                _write_event_to_telemetry_log_file(distinct_id=user_id, event=event, properties=event_properties)

        user_id = _get_or_create_user_id()
        if is_telemetry_enabled():
            send_request(payload=payload)
        elif CONFIG_PATH.exists():
            # if telemetry has just been disabled but the config file has not been deleted yet,
            # then send a final event instead of the triggered event and delete config file and log file afterward
            event = "telemetry disabled"
            send_request(payload={})
            _delete_telemetry_file(TelemetryFileType.CONFIG_FILE)
            _delete_telemetry_file(TelemetryFileType.LOG_FILE)
        else:
            # return without sending any event, not even a final event
            return

    except Exception as e:
        logger.debug("Telemetry was not able to send an event.", exc_info=e)


def send_tutorial_event(url: str):
    """
    Can be called when a tutorial dataset is downloaded so that the dataset URL is used to identify the tutorial and send an event.
    :param url: URL of the dataset that is loaded in the tutorial.
    """
    dataset_url_to_tutorial = {
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip": "1",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/squad_small.json.zip": "2",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip": "3",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/small_faq_covid.csv.zip": "4",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip": "5",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip": "6",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip": "7",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial8.zip": "8",
        # "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz":"9",
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz": "9",
        "https://fandom-qa.s3-eu-west-1.amazonaws.com/saved_models/hp_v3.4.zip": "10",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt11.zip": "11",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt12.zip": "12",
        # Tutorial 13: no dataset available yet
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt14.zip": "14",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/table_text_dataset.zip": "15",
        # "https://nlp.stanford.edu/data/glove.6B.zip": "16",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial16.zip": "16",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt17.zip": "17",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/spirit-animals.zip": "19",
    }
    send_custom_event(event=f"tutorial {dataset_url_to_tutorial.get(url, '?')} executed")


def tutorial_running(tutorial_id: int):
    """
    Can be called when a tutorial is executed so that the tutorial_id is used to identify the tutorial and send an event.
    :param tutorial_id: ID number of the tutorial
    """
    send_custom_event(event=f"tutorial {tutorial_id} executed")


def _get_or_create_user_id() -> Optional[str]:
    """
    Randomly generates a user id or loads the id defined in the config file and returns it.
    Returns None if no id has been set previously and a new one cannot be stored because telemetry is disabled
    """
    global user_id  # pylint: disable=global-statement
    if user_id is None:
        # if user_id is not set, read it from config file
        _read_telemetry_config()
        if user_id is None and is_telemetry_enabled():
            # if user_id cannot be read from config file, create new user_id and write it to config file
            user_id = str(uuid.uuid4())
            _write_telemetry_config()
    return user_id


def _read_telemetry_config():
    """
    Loads the config from the file specified in CONFIG_PATH
    """
    global user_id  # pylint: disable=global-statement
    try:
        if not CONFIG_PATH.is_file():
            return
        with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
            if "user_id" in config and user_id is None:
                user_id = config["user_id"]
    except Exception as e:
        logger.debug("Telemetry was not able to read the config file %s", CONFIG_PATH, exc_info=e)


def _write_telemetry_config():
    """
    Writes a config file storing the randomly generated user id and whether to write events to a log file.
    This method logs an info to inform the user about telemetry when it is used for the first time.
    """
    global user_id  # pylint: disable=global-statement
    try:
        # show a log message if telemetry config is written for the first time
        if not CONFIG_PATH.is_file():
            logger.info(
                "Haystack sends anonymous usage data to understand the actual usage and steer dev efforts "
                "towards features that are most meaningful to users. You can opt-out at anytime by calling "
                "disable_telemetry() or by manually setting the environment variable  "
                "HAYSTACK_TELEMETRY_ENABLED as described for different operating systems on the documentation "
                "page. More information at https://docs.haystack.deepset.ai/docs/telemetry"
            )
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
        user_id = _get_or_create_user_id()
        config = {"user_id": user_id}

        with open(CONFIG_PATH, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
    except Exception:
        logger.debug("Could not write config file to %s", CONFIG_PATH)
        send_custom_event(event="config saving failed")


def _write_event_to_telemetry_log_file(distinct_id: str, event: str, properties: Dict[str, Any]):
    try:
        with open(LOG_PATH, "a") as file_object:
            file_object.write(f"{event}, {properties}, {distinct_id}\n")
    except Exception as e:
        logger.debug("Telemetry was not able to write event to log file %s", LOG_PATH, exc_info=e)


def _delete_telemetry_file(file_type_to_delete: TelemetryFileType):
    """
    Deletes the telemetry config file or log file if it exists.
    """
    if not isinstance(file_type_to_delete, TelemetryFileType):
        logger.debug("File type to delete must be either TelemetryFileType.LOG_FILE or TelemetryFileType.CONFIG_FILE.")
    path = LOG_PATH if file_type_to_delete is TelemetryFileType.LOG_FILE else CONFIG_PATH
    try:
        path.unlink()  # todo add missing_ok=True to the unlink() call when upgrading to python>3.7
    except Exception as e:
        logger.debug("Telemetry was not able to delete the %s at %s", file_type_to_delete, path, exc_info=e)


class NonPrivateParameters:
    param_names: List[str] = [
        "top_k",
        "model_name_or_path",
        "add_isolated_node_eval",
        "fingerprint",
        "type",
        "uptime",
        "run_total",
        "run_total_window",
        "message",
    ]

    @classmethod
    def apply_filter(cls, param_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures that only the values of non-private parameters are sent in events. All other parameter values are filtered out before sending an event.
        If model_name_or_path is a local file path, it will be reduced to the name of the file. The directory names are not sent.

        :param param_dicts: the keyword arguments that need to be filtered before sending an event
        """
        tracked_params = {k: param_dicts[k] for k in cls.param_names if k in param_dicts}

        # if model_name_or_path is a local file path, we reduce it to the model name
        if "model_name_or_path" in tracked_params:
            if (
                Path(tracked_params["model_name_or_path"]).is_file()
                or tracked_params["model_name_or_path"].count(os.path.sep) > 1
            ):
                # if model_name_or_path points to an existing file or contains more than one / it is a path
                tracked_params["model_name_or_path"] = Path(tracked_params["model_name_or_path"]).name
        return tracked_params
