from fastapi import APIRouter

from haystack.api.controller import search, feedback, file_upload

router = APIRouter()

router.include_router(search.router, tags=["search"])
router.include_router(feedback.router, tags=["feedback"])
router.include_router(file_upload.router, tags=["file-upload"])
