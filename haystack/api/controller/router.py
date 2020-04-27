from fastapi import APIRouter

from haystack.api.controller import search, feedback

router = APIRouter()

router.include_router(search.router, tags=["search"])
router.include_router(feedback.router, tags=["feedback"])
