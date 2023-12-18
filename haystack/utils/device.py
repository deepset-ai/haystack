import logging
import os

from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport(message="Using cpu, to use cuda or mps backends run 'pip install transformers[torch]'") as torch_import:
    import torch


def get_device() -> str:
    """
    Detect and return an available torch device as string. Priority is given as follows if the device is available:
    1) GPU, 2) MPS, 3) CPU
    """
    try:
        torch_import.check()
    except ImportError as e:
        logger.warning(e.msg)
        return "cpu:0"

    if torch.cuda.is_available():
        device = "cuda:0"
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and os.getenv("HAYSTACK_MPS_ENABLED", "true") != "false"
    ):
        device = "mps:0"
    else:
        device = "cpu:0"

    return device
