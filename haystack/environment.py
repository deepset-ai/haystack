import os
import platform
import sys
from typing import Any, Dict
import torch
import transformers

from haystack import __version__


HAYSTACK_EXECUTION_CONTEXT = "HAYSTACK_EXECUTION_CONTEXT"
HAYSTACK_DOCKER_CONTAINER = "HAYSTACK_DOCKER_CONTAINER"


env_meta_data: Dict[str, Any] = {}


def get_or_create_env_meta_data() -> Dict[str, Any]:
    """
    Collects meta data about the setup that is used with Haystack, such as: operating system, python version, Haystack version, transformers version, pytorch version, number of GPUs, execution environment, and the value stored in the env variable HAYSTACK_EXECUTION_CONTEXT.
    """
    global env_meta_data  # pylint: disable=global-statement
    if not env_meta_data:
        env_meta_data = {
            "os_version": platform.release(),
            "os_family": platform.system(),
            "os_machine": platform.machine(),
            "python_version": platform.python_version(),
            "haystack_version": __version__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda if torch.cuda.is_available() else 0,
            "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "n_cpu": os.cpu_count(),
            "context": os.environ.get(HAYSTACK_EXECUTION_CONTEXT),
            "execution_env": _get_execution_environment(),
        }
    return env_meta_data


def _get_execution_environment():
    """
    Identifies the execution environment that Haystack is running in.
    Options are: colab notebook, kubernetes, CPU/GPU docker container, test environment, jupyter notebook, python script
    """
    if os.environ.get("CI", "False").lower() == "true":
        execution_env = "ci"
    elif "google.colab" in sys.modules:
        execution_env = "colab"
    elif "KUBERNETES_SERVICE_HOST" in os.environ:
        execution_env = "kubernetes"
    elif HAYSTACK_DOCKER_CONTAINER in os.environ:
        execution_env = os.environ.get(HAYSTACK_DOCKER_CONTAINER)
    # check if pytest is imported
    elif "pytest" in sys.modules:
        execution_env = "test"
    else:
        try:
            execution_env = get_ipython().__class__.__name__  # pylint: disable=undefined-variable
        except NameError:
            execution_env = "script"
    return execution_env
