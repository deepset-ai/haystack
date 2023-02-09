import logging
import os
import platform
import sys
from typing import Any, Dict, Optional

import torch
import transformers

from haystack import __version__

# Any remote API (OpenAI, Cohere etc.)
HAYSTACK_REMOTE_API_BACKOFF_SEC = "HAYSTACK_REMOTE_API_BACKOFF_SEC"
HAYSTACK_REMOTE_API_MAX_RETRIES = "HAYSTACK_REMOTE_API_MAX_RETRIES"
HAYSTACK_REMOTE_API_TIMEOUT_SEC = "HAYSTACK_REMOTE_API_TIMEOUT_SEC"

env_meta_data: Dict[str, Any] = {}

logger = logging.getLogger(__name__)


def set_pytorch_secure_model_loading(flag_val="1"):
    # To load secure only model pytorch requires value of
    # TORCH_FORCE_WEIGHTS_ONLY_LOAD to be ["1", "y", "yes", "true"]
    os_flag_val = os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
    if os_flag_val is None:
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = flag_val
    else:
        logger.info("TORCH_FORCE_WEIGHTS_ONLY_LOAD is already set to %s, Haystack will use the same.", os_flag_val)


def is_containerized() -> Optional[bool]:
    # https://www.baeldung.com/linux/is-process-running-inside-container
    # Using CPU scheduling info as I found it to be the only one usable on my machine.
    path = "/proc/1/sched"
    try:
        with open(path, "r") as cgroupfile:
            first_line = cgroupfile.readline()
            if first_line.startswith("systemd") or first_line.startswith("init"):
                return False
            return True
    except Exception:
        logger.debug("Failed to detect if Haystack is running in a container (for telemetry purposes).")
        return None


def collect_static_system_specs() -> Dict[str, Any]:
    """
    Collects meta data about the setup that is used with Haystack, such as:
    operating system, python version, Haystack version, transformers version,
    pytorch version, number of GPUs, execution environment.
    """
    return {
        "libraries.haystack": __version__,
        "libraries.transformers": transformers.__version__ if "transformers" in sys.modules.keys() else False,
        "libraries.torch": torch.__version__ if "torch" in sys.modules.keys() else False,
        "libraries.cuda": torch.version.cuda if "torch" in sys.modules.keys() and torch.cuda.is_available() else False,
        "os.containerized": is_containerized(),
        # FIXME review these
        "os.version": platform.release(),
        "os.family": platform.system(),
        "os.machine": platform.machine(),
        "python.version": platform.python_version(),  # FIXME verify
        "hardware.cpus": os.cpu_count(),  # FIXME verify
        "hardware.gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,  # probably ok
    }


def collect_dynamic_system_specs() -> Dict[str, Any]:
    return {
        "libraries.pytest": sys.modules["pytest"].__version__ if "pytest" in sys.modules.keys() else False,
        "libraries.ray": sys.modules["ray"].__version__ if "ray" in sys.modules.keys() else False,
        "libraries.ipython": sys.modules["ipython"].__version__ if "ipython" in sys.modules.keys() else False,
        "libraries.colab": sys.modules["pytest"].__version__ if "google.colab" in sys.modules.keys() else False,
    }
