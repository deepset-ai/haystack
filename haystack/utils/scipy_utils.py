import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import njit  # pylint: disable=import-error
except (ImportError, ModuleNotFoundError):
    logger.debug("Numba not found, replacing njit() with no-op implementation. Enable it with 'pip install numba'.")

    def njit(f):
        return f


@njit  # (fastmath=True)
def expit(x: float) -> float:
    return 1 / (1 + np.exp(-x))
