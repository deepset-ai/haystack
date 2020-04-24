from fastapi import APIRouter

from haystack.api.controller import model, feedback

router = APIRouter()

router.include_router(model.router, tags=["model"])
router.include_router(feedback.router, tags=["feedback"])
