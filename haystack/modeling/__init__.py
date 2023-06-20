import logging

logger = logging.getLogger(__name__)

try:
    import torch
except (ModuleNotFoundError, ImportError) as iexc:
    raise ImportError(
        "torch not installed, haystack.modeling won't work. Run 'pip install transformers[torch]' to fix this problem."
    )
