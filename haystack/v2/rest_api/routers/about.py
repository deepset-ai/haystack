import logging

from fastapi import APIRouter

from haystack import __version__


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/ready")
def check_status():
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@router.get("/version")
def version():
    """
    Get the running version of Haystack.
    """
    return {"haystack": __version__}
