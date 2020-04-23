from fastapi import APIRouter

from haystack.api.controller import query, feedback

router = APIRouter()

router.include_router(query.router, tags=["model"])
router.include_router(feedback.router, tags=["feedback"])

