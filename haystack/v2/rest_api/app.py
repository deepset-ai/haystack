from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from canals import load_pipelines

from haystack import __version__
from haystack.v2.rest_api.config import DEFAULT_PIPELINES


APP = None
OPENAPI_TAGS = [
    {"name": "about", "description": "Check the app's status"},
    {"name": "pipelines", "description": "Operations on Pipelines: list, warmup, run, etc..."},
    {"name": "files", "description": "Operations on files: upload, dowload, list, etc..."},
]


async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"errors": [exc.detail]}, status_code=exc.status_code)


def get_app() -> FastAPI:
    global APP  # pylint: disable=global-statement
    if APP:
        return APP

    APP = FastAPI(title="Haystack", debug=False, version=__version__, root_path="/", openapi_tags=OPENAPI_TAGS)
    APP.pipelines = load_pipelines(DEFAULT_PIPELINES)

    from haystack.v2.rest_api.routers import pipelines, about, files

    APP.include_router(pipelines.router, tags=["pipelines"])
    APP.include_router(files.router, tags=["files"])
    APP.include_router(about.router, tags=["about"])

    APP.add_exception_handler(HTTPException, http_error_handler)
    return APP
