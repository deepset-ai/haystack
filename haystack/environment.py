# pylint: disable=global-statement
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

# IS_DOCKER cache copy
IS_DOCKER_CACHE = None


def set_pytorch_secure_model_loading(flag_val="1"):
    # To load secure only model pytorch requires value of
    # TORCH_FORCE_WEIGHTS_ONLY_LOAD to be ["1", "y", "yes", "true"]
    os_flag_val = os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
    if os_flag_val is None:
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = flag_val
    else:
        logger.info("TORCH_FORCE_WEIGHTS_ONLY_LOAD is already set to %s, Haystack will use the same.", os_flag_val)


def in_podman() -> bool:
    """
    Podman run would create the file /run/.containernv, see:
    https://github.com/containers/podman/blob/main/docs/source/markdown/podman-run.1.md.in#L31
    """
    return os.path.exists("/run/.containerenv")


def has_dockerenv() -> bool:
    """
    This might not work anymore at some point (even if it's been a while now), see:
    https://github.com/moby/moby/issues/18355#issuecomment-220484748
    """
    return os.path.exists("/.dockerenv")


def has_docker_cgroup_v1() -> bool:
    """
    This only works with cgroups v1
    """
    path = "/proc/self/cgroup"  # 'self' should be always symlinked to the actual PID
    return os.path.isfile(path) and any("docker" in line for line in open(path))


def has_docker_cgroup_v2() -> bool:
    """
    cgroups v2 version, inspired from
    https://github.com/jenkinsci/docker-workflow-plugin/blob/master/src/main/java/org/jenkinsci/plugins/docker/workflow/client/DockerClient.java
    """
    path = "/proc/self/mountinfo"  # 'self' should be always symlinked to the actual PID
    return os.path.isfile(path) and any("/docker/containers/" in line for line in open(path))


def is_containerized() -> Optional[bool]:
    """
    This code is based on the popular 'is-docker' package for node.js
    """
    global IS_DOCKER_CACHE

    if IS_DOCKER_CACHE is None:
        IS_DOCKER_CACHE = in_podman() or has_dockerenv() or has_docker_cgroup_v1() or has_docker_cgroup_v2()

    return IS_DOCKER_CACHE


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
        "libraries.colab": sys.modules["google.colab"].__version__ if "google.colab" in sys.modules.keys() else False,
    }


#
# Old telemetry
#


def get_or_create_env_meta_data() -> Dict[str, Any]:
    """
    Collects meta data about the setup that is used with Haystack, such as: operating system, python version, Haystack version, transformers version, pytorch version, number of GPUs, execution environment, and the value stored in the env variable HAYSTACK_EXECUTION_CONTEXT.
    """
    from haystack.telemetry import HAYSTACK_EXECUTION_CONTEXT

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
    from haystack.telemetry import HAYSTACK_DOCKER_CONTAINER

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
